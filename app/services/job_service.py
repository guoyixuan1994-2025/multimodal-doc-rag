from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.schemas.document import ParsedDocument


class ParseJobStore:
    """进程内解析任务状态表，并把状态快照写到本地 JSON 文件。

    当前项目是单机学习/演示版，所以先用内存字典 + JSON 快照。
    真实线上系统可以把这一层替换成 Redis、Celery、RQ 或数据库任务表。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.jobs_dir = self.settings.output_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._parse_lock = threading.Lock()

    @property
    def parse_lock(self) -> threading.Lock:
        """控制同一进程内的大文档解析串行执行，避免同时抢 GPU/CPU/Chroma 文件锁。"""

        return self._parse_lock

    def create(self, file_path: str, analysis_mode: str, collection_name: str) -> dict[str, Any]:
        """创建一个解析任务，返回给前端轮询用的 job_id。"""

        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "任务已创建，等待解析。",
            "file_path": file_path,
            "analysis_mode": analysis_mode,
            "collection_name": collection_name,
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
            "profile": None,
            "detail": {},
        }
        with self._lock:
            self._jobs[job_id] = job
            self._cancel_events[job_id] = threading.Event()
            self._persist(job)
        return job

    def update(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        """更新任务状态，并同步写入 JSON 快照。"""

        with self._lock:
            current = self._jobs.get(job_id, {"job_id": job_id})
            current.update(kwargs)
            current["updated_at"] = time.time()
            self._jobs[job_id] = current
            self._persist(current)
            return dict(current)

    def get(self, job_id: str) -> dict[str, Any] | None:
        """读取任务状态；内存没有时，从 JSON 快照恢复。"""

        with self._lock:
            if job_id in self._jobs:
                return dict(self._jobs[job_id])

        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None

        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        with self._lock:
            self._jobs[job_id] = job
            self._cancel_events.setdefault(job_id, threading.Event())
        return dict(job)

    def request_cancel(self, job_id: str) -> dict[str, Any] | None:
        """请求终止排队中或执行中的解析任务。"""

        job = self.get(job_id)
        if job is None:
            return None
        if job.get("status") in {"completed", "failed", "cancelled"}:
            return job

        with self._lock:
            cancel_event = self._cancel_events.setdefault(job_id, threading.Event())
            cancel_event.set()
            current = self._jobs[job_id]
            status = "cancelled" if current.get("status") == "queued" else "cancelling"
            current.update(
                status=status,
                message="任务已取消。" if status == "cancelled" else "正在终止解析任务...",
                updated_at=time.time(),
            )
            self._persist(current)
            return dict(current)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            event = self._cancel_events.get(job_id)
            return event.is_set() if event else False

    def mark_cancelled(self, job_id: str) -> dict[str, Any]:
        return self.update(job_id, status="cancelled", message="任务已取消。", error=None)

    def progress_callback(self, job_id: str):
        """给文档解析服务使用的进度回调。"""

        def _callback(progress: int, message: str, detail: dict | None = None) -> None:
            if self.is_cancel_requested(job_id):
                return
            self.update(
                job_id,
                status="running",
                progress=max(0, min(progress, 99)),
                message=message,
                detail=detail or {},
            )

        return _callback

    def _persist(self, job: dict[str, Any]) -> None:
        """用临时文件原子替换，降低 Windows 下写一半被读取的概率。"""

        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        path = self.jobs_dir / f"{job['job_id']}.json"
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)


def build_document_profile(doc: ParsedDocument) -> dict[str, Any]:
    """根据解析结果生成文档级、页面级、block 级路由画像。"""

    block_counts: dict[str, int] = {}
    page_map: dict[int, dict[str, Any]] = {}

    for block in doc.blocks:
        block_counts[block.block_type] = block_counts.get(block.block_type, 0) + 1
        page = block.page or 0
        page_info = page_map.setdefault(
            page,
            {
                "page": block.page,
                "block_types": {},
                "route": "text_layout",
            },
        )
        page_info["block_types"][block.block_type] = page_info["block_types"].get(block.block_type, 0) + 1

    image_blocks = block_counts.get("image_caption", 0)
    table_blocks = block_counts.get("table", 0)
    ocr_blocks = block_counts.get("ocr_text", 0)

    if image_blocks or ocr_blocks:
        doc_route = "multimodal"
    elif table_blocks:
        doc_route = "table_aware_text"
    else:
        doc_route = "text_layout"

    for page_info in page_map.values():
        types = page_info["block_types"]
        if types.get("image_caption"):
            page_info["route"] = "vlm_image_caption"
        elif types.get("ocr_text"):
            page_info["route"] = "ocr"
        elif types.get("table"):
            page_info["route"] = "table_extraction"

    return {
        "doc_id": doc.doc_id,
        "file_name": doc.file_name,
        "parser_mode": doc.parser_mode,
        "document_route": doc_route,
        "block_counts": block_counts,
        "page_routes": sorted(page_map.values(), key=lambda item: item["page"] or 0),
    }
