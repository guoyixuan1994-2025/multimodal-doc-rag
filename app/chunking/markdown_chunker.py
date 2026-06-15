from __future__ import annotations

from app.schemas.document import ChunkRecord, ParsedDocument


def chunk_markdown_document(doc: ParsedDocument) -> list[ChunkRecord]:
    # Markdown 已经按标题解析成 block，这里尽量保留标题语义，一个 block 先对应一个 chunk。
    chunks: list[ChunkRecord] = []
    for index, block in enumerate(doc.blocks, start=1):
        content = block.content.strip()
        if not content:
            continue
        chunk_id = f"{doc.doc_id}-chunk-{index}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                chunk_type=block.block_type,
                content=content,
                metadata={
                    "chunk_id": chunk_id,
                    "doc_id": doc.doc_id,
                    "file_name": doc.file_name,
                    "file_type": doc.file_type,
                    "page": block.page,
                    "title": block.title,
                    "chunk_type": block.block_type,
                    "source_path": doc.source_path,
                },
            )
        )
    return chunks
