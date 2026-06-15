from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.api.web import router as web_router
from app.core.auth import require_api_key
from app.core.config import BASE_DIR, get_settings
from app.core.logging import setup_logging
from app.parsers.mineru_parser import ParseCancelledError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.document_service import DocumentService
from app.services.job_service import ParseJobStore, build_document_profile
from app.services.rag_service import RagService


setup_logging()
settings = get_settings()
app = FastAPI(title=settings.project_name)
app.include_router(web_router)

document_service = DocumentService(collection_name=settings.default_collection_name)
job_store = ParseJobStore()
rag_service: RagService | None = None
knowledge_base_lock = threading.RLock()
evaluation_lock = threading.Lock()
SUPPORTED_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
    ".md",
    ".markdown",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


def refresh_rag_service() -> None:
    """用当前内存中的 chunk 刷新检索问答服务。"""
    global rag_service
    rag_service = RagService(
        document_service.get_chunks(),
        collection_name=document_service.collection_name,
    )


def switch_collection(collection_name: str | None = None) -> None:
    """切换当前知识库，并恢复该 collection 中已经持久化的全部 chunk。"""
    document_service.set_collection(collection_name or settings.default_collection_name)


def ensure_rag_service(collection_name: str | None = None) -> RagService:
    """确保问答面向指定 collection 的整个知识库，而不是最近一次上传结果。"""
    with knowledge_base_lock:
        target_collection = collection_name or settings.default_collection_name
        if target_collection != document_service.collection_name or rag_service is None:
            switch_collection(target_collection)
            refresh_rag_service()
        if not document_service.get_chunks():
            raise HTTPException(status_code=400, detail="当前知识库为空，请先批量入库或上传文档。")
        assert rag_service is not None
        return rag_service


def load_sample_docs(collection_name: str | None = None, reset: bool = False) -> dict:
    """加载项目内置的轻量示例文档，主要用于冷启动和评估。"""
    sample_dir = BASE_DIR / "sample_docs"
    paths = sorted([path for path in sample_dir.iterdir() if path.suffix.lower() in {".md", ".txt"}])
    with knowledge_base_lock:
        switch_collection(collection_name)
        results = document_service.ingest_paths(paths, reset=reset)
        refresh_rag_service()
        return {
            "success": True,
            "documents": [result.model_dump() for result in results],
            "total_chunks": len(document_service.get_chunks()),
        }


def run_parse_job(
    job_id: str,
    file_path: str,
    analysis_mode: str,
    collection_name: str,
    reset: bool = False,
) -> None:
    """后台解析任务。

    当前版本使用单机串行锁，避免多个大文档同时争抢 GPU/CPU/Chroma 文件锁。
    """
    with job_store.parse_lock:
        try:
            if job_store.is_cancel_requested(job_id):
                job_store.mark_cancelled(job_id)
                return
            job_store.update(job_id, status="running", progress=5, message="任务开始。")
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在：{path}")

            with knowledge_base_lock:
                switch_collection(collection_name)
                results = document_service.ingest_paths(
                    [path],
                    reset=reset,
                    analysis_mode=analysis_mode,
                    progress_callback=job_store.progress_callback(job_id),
                    cancel_check=lambda: job_store.is_cancel_requested(job_id),
                )
                refresh_rag_service()
                profiles = [build_document_profile(doc) for doc in document_service.parsed_documents]
                payload = {
                    "success": True,
                    "documents": [result.model_dump() for result in results],
                    "total_chunks": len(document_service.get_chunks()),
                }
            job_store.update(
                job_id,
                status="completed",
                progress=100,
                message="解析入库完成。",
                result=payload,
                profile=profiles,
                error=None,
            )
        except ParseCancelledError:
            job_store.mark_cancelled(job_id)
        except Exception as exc:  # noqa: BLE001 - 任务错误要落到状态里给前端展示
            job_store.update(
                job_id,
                status="failed",
                progress=100,
                message="解析入库失败。",
                error=str(exc),
            )


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "name": settings.project_name,
        "status": "ok",
        "message": "Swagger 用于开发调试，真实用户入口建议使用 Streamlit 操作台或企业系统前端。",
    }


@app.post("/documents/ingest-sample")
def ingest_sample_docs(
    collection_name: str | None = None,
    reset: bool = False,
    _: None = Depends(require_api_key),
) -> dict:
    return load_sample_docs(collection_name=collection_name, reset=reset)


@app.post("/documents/ingest-path")
def ingest_path(
    file_path: str,
    analysis_mode: Literal["auto", "text", "ocr", "full"] = "auto",
    reset: bool = False,
    collection_name: str | None = None,
    _: None = Depends(require_api_key),
) -> dict:
    path = Path(file_path)
    try:
        with knowledge_base_lock:
            switch_collection(collection_name)
            results = document_service.ingest_paths([path], reset=reset, analysis_mode=analysis_mode)
            refresh_rag_service()
            profiles = [build_document_profile(doc) for doc in document_service.parsed_documents]
            total_chunks = len(document_service.get_chunks())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"文档解析或入库失败：{exc}") from exc

    return {
        "success": True,
        "documents": [result.model_dump() for result in results],
        "total_chunks": total_chunks,
        "profiles": profiles,
    }


@app.post("/documents/ingest-directory")
def ingest_directory(
    directory_path: str,
    analysis_mode: Literal["auto", "text", "ocr", "full"] = "auto",
    recursive: bool = True,
    reset: bool = False,
    collection_name: str | None = None,
    _: None = Depends(require_api_key),
) -> dict:
    """批量构建或增量更新知识库，适合已有资料目录一次性入库。"""
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在：{directory}")
    candidates = directory.rglob("*") if recursive else directory.glob("*")
    paths = sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
    )
    if not paths:
        raise HTTPException(status_code=400, detail="目录中没有可解析的支持格式文档。")
    try:
        with knowledge_base_lock:
            switch_collection(collection_name)
            results = document_service.ingest_paths(paths, reset=reset, analysis_mode=analysis_mode)
            refresh_rag_service()
            profiles = [build_document_profile(doc) for doc in document_service.parsed_documents]
            total_chunks = len(document_service.get_chunks())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"批量解析或入库失败：{exc}") from exc
    return {
        "success": True,
        "mode": "reset" if reset else "incremental",
        "file_count": len(paths),
        "documents": [result.model_dump() for result in results],
        "total_chunks": total_chunks,
        "profiles": profiles,
    }


@app.post("/documents/parse-jobs")
def create_parse_job(
    file_path: str,
    analysis_mode: Literal["auto", "text", "ocr", "full"] = "auto",
    collection_name: str | None = None,
    reset: bool = False,
    _: None = Depends(require_api_key),
) -> dict:
    collection = collection_name or settings.default_collection_name
    job = job_store.create(file_path=file_path, analysis_mode=analysis_mode, collection_name=collection)
    worker = threading.Thread(
        target=run_parse_job,
        args=(job["job_id"], file_path, analysis_mode, collection, reset),
        daemon=True,
        name=f"parse-{job['job_id']}",
    )
    worker.start()
    return job


@app.get("/documents/parse-jobs/{job_id}")
def get_parse_job(job_id: str, _: None = Depends(require_api_key)) -> dict:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return job


@app.post("/documents/parse-jobs/{job_id}/cancel")
def cancel_parse_job(job_id: str, _: None = Depends(require_api_key)) -> dict:
    job = job_store.request_cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return job


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _: None = Depends(require_api_key)) -> ChatResponse:
    service = ensure_rag_service(request.collection_name)
    return service.answer(request.question)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest, _: None = Depends(require_api_key)) -> StreamingResponse:
    service = ensure_rag_service(request.collection_name)
    return StreamingResponse(
        service.stream_answer_events(request.question),
        media_type="text/event-stream; charset=utf-8",
    )


@app.post("/eval/run")
def run_eval_api(isolated: bool = True, _: None = Depends(require_api_key)) -> dict:
    with evaluation_lock:
        return execute_eval(isolated)


def execute_eval(isolated: bool = True) -> dict:
    from app.services.eval_service import run_basic_eval, run_ragas_offline_eval, save_eval_outputs, save_ragas_outputs

    service_for_eval = rag_service
    if isolated:
        sample_dir = BASE_DIR / "sample_docs"
        paths = sorted([path for path in sample_dir.iterdir() if path.suffix.lower() in {".md", ".txt"}])
        eval_document_service = DocumentService(collection_name=settings.eval_collection_name)
        eval_document_service.ingest_paths(paths, reset=True)
        service_for_eval = RagService(
            eval_document_service.get_chunks(),
            collection_name=settings.eval_collection_name,
        )

    if service_for_eval is None:
        service_for_eval = ensure_rag_service()

    assert service_for_eval is not None
    df = run_basic_eval(service_for_eval)
    outputs = save_eval_outputs(df)
    ragas_df = run_ragas_offline_eval(df)
    ragas_outputs = save_ragas_outputs(ragas_df)
    return {
        "success": True,
        "isolated_collection": isolated,
        "basic_eval_csv": str(outputs["csv"]),
        "basic_eval_png": str(outputs["png"]),
        "ragas_csv": str(ragas_outputs["csv"]),
        "ragas_png": str(ragas_outputs["png"]),
        "metrics": {
            "hit": float(df["hit"].mean()),
            "precision_at_k": float(df["precision_at_k"].mean()),
            "recall_at_k": float(df["recall_at_k"].mean()),
            "mrr": float(df["mrr"].mean()),
        },
    }
