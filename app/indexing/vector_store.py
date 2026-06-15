from __future__ import annotations

import warnings
import json
from pathlib import Path
from typing import Any, cast, get_args

warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

try:
    from langchain_chroma import Chroma
except Exception:
    warnings.filterwarnings(
        "ignore",
        message="The class `Chroma` was deprecated.*",
        category=Warning,
    )
    from langchain_community.vectorstores import Chroma

from app.core.config import get_settings
from app.indexing.embeddings import create_embeddings
from app.schemas.document import BlockType, ChunkRecord


def reset_vector_store(collection_name: str | None = None) -> None:
    """
    清空当前 collection 中的旧 chunk，但保留 Chroma 的持久化目录。

    不能直接删除 chroma_store 目录：当 Web 服务已经打开向量库时，
    Windows 会锁住索引二进制文件，直接 rmtree 会触发 WinError 32。
    """
    store = open_vector_store(collection_name=collection_name)
    existing = store.get()
    ids = existing.get("ids", [])
    if ids:
        store.delete(ids=ids)


def delete_chunks_by_doc_id(doc_id: str, collection_name: str | None = None) -> int:
    """按逻辑文档 ID 删除旧版本 chunk，用于同名文档的增量覆盖更新。"""
    store = open_vector_store(collection_name=collection_name)
    existing = store.get(where={"doc_id": doc_id})
    ids = existing.get("ids", []) or []
    if ids:
        store.delete(ids=ids)
    return len(ids)


def build_vector_store(chunks: list[ChunkRecord], collection_name: str | None = None) -> Chroma:
    store = open_vector_store(collection_name=collection_name)
    texts = [chunk.content for chunk in chunks]
    metadatas = [
        sanitize_metadata(
            {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_type": chunk.chunk_type,
            }
        )
        for chunk in chunks
    ]
    ids = [chunk.chunk_id for chunk in chunks]

    if texts:
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return store


def load_chunks_from_vector_store(collection_name: str | None = None) -> list[ChunkRecord]:
    """从持久化 collection 恢复全部 chunk，保证服务重启后仍可全库检索。"""
    store = open_vector_store(collection_name=collection_name)
    existing = store.get(include=["documents", "metadatas"])
    ids = existing.get("ids", []) or []
    documents = existing.get("documents", []) or []
    metadatas = existing.get("metadatas", []) or []

    chunks: list[ChunkRecord] = []
    for index, stored_id in enumerate(ids):
        content = documents[index] if index < len(documents) else ""
        metadata = restore_metadata(metadatas[index] if index < len(metadatas) else {})
        doc_id = str(metadata.get("doc_id") or "")
        if not doc_id or not content:
            continue
        chunks.append(
            ChunkRecord(
                chunk_id=str(metadata.get("chunk_id") or stored_id),
                doc_id=doc_id,
                content=content,
                chunk_type=normalize_chunk_type(metadata.get("chunk_type")),
                metadata=metadata,
            )
        )
    return chunks


def open_vector_store(collection_name: str | None = None) -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=collection_name or settings.default_collection_name,
        embedding_function=create_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def sanitize_metadata(metadata: dict) -> dict:
    # Chroma metadata 只接受 str/int/float/bool/None 的简单值，复杂对象要转成字符串。
    clean: dict[str, str | int | float | bool | None] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, Path):
            clean[key] = str(value)
        elif isinstance(value, list):
            clean[key] = json.dumps(value, ensure_ascii=False)
        else:
            clean[key] = str(value)
    return clean


def restore_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """还原入库时被 JSON 序列化的定位信息，便于界面继续展示 bbox 等字段。"""
    restored = dict(metadata or {})
    for key in ("bbox",):
        value = restored.get(key)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                restored[key] = parsed
    return restored


def normalize_chunk_type(value: Any) -> BlockType:
    """旧数据缺少 chunk_type 时，按普通文本兼容恢复。"""
    valid_types = set(get_args(BlockType))
    if isinstance(value, str) and value in valid_types:
        return cast(BlockType, value)
    return "text"
