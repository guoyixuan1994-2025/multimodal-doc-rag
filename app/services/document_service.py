from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from app.chunking.recursive_chunker import chunk_document
from app.core.config import get_settings
from app.indexing.vector_store import (
    build_vector_store,
    delete_chunks_by_doc_id,
    load_chunks_from_vector_store,
    reset_vector_store,
)
from app.parsers.file_router import parse_document
from app.parsers.mineru_parser import ParseCancelledError
from app.schemas.document import ChunkRecord, DocumentIngestResult, ParsedDocument


ProgressCallback = Callable[[int, str, dict[str, Any] | None], None]
CancelCheck = Callable[[], bool]


class DocumentService:
    def __init__(self, collection_name: str | None = None) -> None:
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.default_collection_name
        self.chunks: list[ChunkRecord] = []
        self.parsed_documents: list[ParsedDocument] = []
        self.reload_from_store()

    def set_collection(self, collection_name: str | None) -> None:
        """切换业务知识库，并从持久化向量库加载该库全部已有内容。"""
        self.collection_name = collection_name or self.settings.default_collection_name
        self.reload_from_store()

    def reload_from_store(self) -> None:
        """检索范围以 collection 全量已有内容为准，而不是只看最近上传文档。"""
        self.chunks = load_chunks_from_vector_store(collection_name=self.collection_name)
        self.parsed_documents = []

    def ingest_paths(
        self,
        paths: list[Path],
        reset: bool = False,
        analysis_mode: str = "auto",
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> list[DocumentIngestResult]:
        def raise_if_cancelled() -> None:
            if cancel_check and cancel_check():
                raise ParseCancelledError("文档解析任务已取消。")

        raise_if_cancelled()
        self.reload_from_store()
        original_chunks = list(self.chunks)
        results: list[DocumentIngestResult] = []
        new_chunks: list[ChunkRecord] = []
        parsed_outputs: list[tuple[ParsedDocument, list[ChunkRecord]]] = []
        store_mutated = False

        try:
            raise_if_cancelled()
            if reset:
                if progress_callback:
                    progress_callback(10, "清空旧索引", {"collection_name": self.collection_name})
                reset_vector_store(collection_name=self.collection_name)
                store_mutated = True
                self.chunks = []
                self.parsed_documents = []

            for path in paths:
                raise_if_cancelled()
                if progress_callback:
                    progress_callback(20, f"解析文档：{path.name}", {"file_path": str(path), "analysis_mode": analysis_mode})
                parsed = parse_document(path, analysis_mode=analysis_mode, cancel_check=cancel_check)

                raise_if_cancelled()
                if progress_callback:
                    progress_callback(55, "生成 chunk", {"parser_mode": parsed.parser_mode, "block_count": len(parsed.blocks)})
                chunks = chunk_document(parsed)

                raise_if_cancelled()
                # 同一逻辑文档（同名修订版）先移除旧 chunk，再写入新版本。
                deleted_count = 0
                if not reset:
                    deleted_count = delete_chunks_by_doc_id(parsed.doc_id, collection_name=self.collection_name)
                    store_mutated = store_mutated or deleted_count > 0
                self.chunks = [item for item in self.chunks if item.doc_id != parsed.doc_id]
                self.parsed_documents = [item for item in self.parsed_documents if item.doc_id != parsed.doc_id]
                new_chunks = [item for item in new_chunks if item.doc_id != parsed.doc_id]
                if progress_callback and deleted_count:
                    progress_callback(65, f"替换旧版本：{path.name}", {"deleted_chunks": deleted_count})

                self.parsed_documents.append(parsed)
                self.chunks.extend(chunks)
                new_chunks.extend(chunks)
                parsed_outputs.append((parsed, chunks))
                results.append(
                    DocumentIngestResult(
                        doc_id=parsed.doc_id,
                        file_name=parsed.file_name,
                        file_type=parsed.file_type,
                        parser_mode=parsed.parser_mode,
                        block_count=len(parsed.blocks),
                        chunk_count=len(chunks),
                    )
                )

            raise_if_cancelled()
            if progress_callback:
                progress_callback(
                    75,
                    "写入向量数据库",
                    {"new_chunk_count": len(new_chunks), "total_chunks": len(self.chunks)},
                )
            store_mutated = True
            build_vector_store(new_chunks, collection_name=self.collection_name)
        except Exception:
            # An update is only valid after all new vectors are persisted; preserve the last healthy collection.
            if store_mutated:
                reset_vector_store(collection_name=self.collection_name)
                build_vector_store(original_chunks, collection_name=self.collection_name)
            self.reload_from_store()
            raise

        for parsed, chunks in parsed_outputs:
            self._save_parsed_document(parsed, chunks)

        if progress_callback:
            progress_callback(95, "刷新检索服务", {"total_chunks": len(self.chunks)})
        return results

    def get_chunks(self) -> list[ChunkRecord]:
        return self.chunks

    def _save_parsed_document(self, doc: ParsedDocument, chunks: list[ChunkRecord]) -> None:
        output = {
            "document": doc.model_dump(),
            "chunks": [chunk.model_dump() for chunk in chunks],
        }
        output_path = self.settings.parsed_dir / f"{doc.doc_id}.json"
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
