from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunking.markdown_chunker import chunk_markdown_document
from app.core.config import get_settings
from app.schemas.document import ChunkRecord, ParsedDocument


# 有些 block 虽然字符很短，但语义价值很高，比如图片描述、公式、表格。
# 这些类型不能简单按最小字符数过滤掉。
KEEP_SHORT_BLOCK_TYPES = {
    "table",
    "formula",
    "image",
    "image_caption",
    "figure",
    "ocr_text",
}


# 政企文档里经常有很短但非常关键的指标行，例如 SLA 99.9%、端口 8080、响应 15 分钟。
# 这些短文本如果被 min_chunk_chars 过滤，会直接导致 RAG 查不到关键事实。
STRUCTURED_FACT_KEYWORDS = {
    "sla",
    "kpi",
    "qos",
    "cpu",
    "gpu",
    "qps",
    "tps",
    "api",
    "id",
    "gb",
    "mb",
    "tb",
    "ms",
    "指标",
    "比例",
    "成功率",
    "准确率",
    "召回率",
    "时延",
    "延迟",
    "耗时",
    "金额",
    "价格",
    "费用",
    "版本",
    "编号",
    "端口",
    "阈值",
    "容量",
    "并发",
    "响应",
    "分钟",
    "小时",
}

STRUCTURED_FACT_PATTERN = re.compile(
    r"(\d+(\.\d+)?\s*(%|ms|s|秒|分钟|小时|天|元|万|亿|GB|MB|TB|QPS|TPS)?)"
    r"|([A-Za-z]+[-_./]?\d+)"
    r"|(\d+[-_/]\d+[-_/]\d+)",
    re.IGNORECASE,
)


def chunk_document(doc: ParsedDocument) -> list[ChunkRecord]:
    # Markdown 文档已经按标题层级解析成 block，单独走 Markdown 规则切分。
    if doc.file_type in {"md", "markdown"}:
        return chunk_markdown_document(doc)
    return recursive_chunk_document(doc)


def normalize_chunk_text(text: str) -> str:
    """清理空白行，避免页眉、页脚、换行碎片污染 chunk。"""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def looks_like_structured_fact(text: str) -> bool:
    """识别短指标/短配置项，避免 SLA 99.9% 这类关键信息被过滤。"""
    normalized = text.strip()
    lowered = normalized.lower()
    has_keyword = any(keyword in lowered for keyword in STRUCTURED_FACT_KEYWORDS)
    has_value = bool(STRUCTURED_FACT_PATTERN.search(normalized))
    has_key_value_shape = bool(re.search(r"[:：]\s*\d", normalized))
    return (has_keyword and has_value) or has_key_value_shape


def should_keep_chunk(text: str, chunk_type: str, min_chunk_chars: int) -> bool:
    """过滤太短的低价值 chunk，同时保留表格、公式、图片描述、短指标等高价值块。"""
    normalized = text.strip()
    if not normalized:
        return False
    if chunk_type in KEEP_SHORT_BLOCK_TYPES:
        return True
    if looks_like_structured_fact(normalized):
        return True
    return len(normalized) >= min_chunk_chars


def recursive_chunk_document(doc: ParsedDocument) -> list[ChunkRecord]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n\n", "\n", "。", "；", " ", ""],
        add_start_index=True,
    )

    langchain_docs: list[Document] = []
    for block in doc.blocks:
        content = normalize_chunk_text(block.content)
        if not should_keep_chunk(content, block.block_type, settings.min_chunk_chars):
            continue
        langchain_docs.append(
            Document(
                page_content=content,
                metadata={
                    "doc_id": doc.doc_id,
                    "file_name": doc.file_name,
                    "file_type": doc.file_type,
                    "page": block.page,
                    "title": block.title,
                    "chunk_type": block.block_type,
                    "source_path": doc.source_path,
                    "image_path": block.image_path,
                    "bbox": block.bbox,
                    "block_id": block.block_id,
                    "parser_mode": doc.parser_mode,
                    **block.metadata,
                },
            )
        )

    split_docs = splitter.split_documents(langchain_docs)
    chunks: list[ChunkRecord] = []
    for split_doc in split_docs:
        chunk_type = split_doc.metadata.get("chunk_type") or "text"
        content = normalize_chunk_text(split_doc.page_content)
        if not should_keep_chunk(content, chunk_type, settings.min_chunk_chars):
            continue
        chunk_id = f"{doc.doc_id}-chunk-{len(chunks) + 1}"
        metadata = dict(split_doc.metadata)
        metadata["chunk_id"] = chunk_id
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                chunk_type=chunk_type,
                content=content,
                metadata=metadata,
            )
        )
    return chunks
