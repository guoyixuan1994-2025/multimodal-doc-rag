from __future__ import annotations

from pathlib import Path

from app.schemas.document import ParsedBlock, ParsedDocument


def parse_markdown(path: Path, doc_id: str) -> ParsedDocument:
    # Markdown 天然有标题层级，先按标题粗分成 block，后续再用规则分块。
    text = path.read_text(encoding="utf-8")
    blocks: list[ParsedBlock] = []
    current_title = None
    current_lines: list[str] = []
    block_index = 0

    def flush_block() -> None:
        nonlocal block_index, current_lines
        content = "\n".join(line.strip() for line in current_lines if line.strip()).strip()
        if not content:
            current_lines = []
            return
        block_index += 1
        blocks.append(
            ParsedBlock(
                block_id=f"{doc_id}-block-{block_index}",
                block_type="text",
                content=content,
                title=current_title,
            )
        )
        current_lines = []

    for line in text.splitlines():
        if line.startswith("#"):
            flush_block()
            current_title = line.strip("# ").strip()
        current_lines.append(line)
    flush_block()

    return ParsedDocument(
        doc_id=doc_id,
        file_name=path.name,
        file_type="md",
        source_path=str(path),
        blocks=blocks,
    )
