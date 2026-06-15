from __future__ import annotations

from pathlib import Path

from app.schemas.document import ParsedBlock, ParsedDocument


def parse_txt(path: Path, doc_id: str) -> ParsedDocument:
    # TXT 没有结构信息，先作为一个大 block，后续交给递归分块器处理。
    text = path.read_text(encoding="utf-8")
    return ParsedDocument(
        doc_id=doc_id,
        file_name=path.name,
        file_type="txt",
        source_path=str(path),
        blocks=[
            ParsedBlock(
                block_id=f"{doc_id}-block-1",
                block_type="text",
                content=text.strip(),
                title=path.stem,
            )
        ],
    )
