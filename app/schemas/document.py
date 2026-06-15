from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


BlockType = Literal["text", "table", "formula", "ocr_text", "image_caption"]
MetadataValue = str | int | float | bool | None | list[float]


class ParsedBlock(BaseModel):
    # 解析层输出的最小结构。无论来自 PDF、Markdown、TXT 还是 OCR，先统一成 block。
    block_id: str
    block_type: BlockType = "text"
    content: str
    page: int | None = None
    title: str | None = None
    image_path: str | None = None
    bbox: list[float] | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    doc_id: str
    file_name: str
    file_type: str
    source_path: str
    parser_mode: str | None = None
    blocks: list[ParsedBlock]


class ChunkRecord(BaseModel):
    # 真正进入检索系统的是 chunk，不是原始文档。
    chunk_id: str
    doc_id: str
    content: str
    chunk_type: BlockType = "text"
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class DocumentIngestResult(BaseModel):
    doc_id: str
    file_name: str
    file_type: str
    parser_mode: str | None = None
    block_count: int
    chunk_count: int
    status: str = "indexed"


def infer_file_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"
