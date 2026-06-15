from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from app.parsers.markdown_parser import parse_markdown
from app.parsers.mineru_parser import parse_with_mineru
from app.parsers.txt_parser import parse_txt
from app.schemas.document import ParsedDocument


def build_doc_id(path: Path) -> str:
    # 同名文件视为同一逻辑文档：上传修订版时覆盖旧 chunk，而不是重复累积。
    logical_name = path.name.strip().casefold()
    digest = hashlib.md5(logical_name.encode("utf-8")).hexdigest()[:10]
    return f"doc-{digest}"


def parse_document(
    path: Path,
    doc_id: str | None = None,
    analysis_mode: str = "auto",
    cancel_check: Callable[[], bool] | None = None,
) -> ParsedDocument:
    doc_id = doc_id or build_doc_id(path)
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown"}:
        return parse_markdown(path, doc_id)
    if suffix in {".txt"}:
        return parse_txt(path, doc_id)
    if suffix in {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return parse_with_mineru(path, doc_id, analysis_mode=analysis_mode, cancel_check=cancel_check)

    raise ValueError(f"暂不支持的文件类型：{suffix}")
