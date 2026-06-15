from __future__ import annotations

import re
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from app.generation.llm_client import create_chat_llm


REWRITE_PROMPT = """你是 RAG 检索 query 改写助手。请把用户问题改写成更适合检索知识库的 query。

当前知识库文件名：
{documents}

要求：
1. 保留用户原始专有名词、文件名、缩写、编号，不要删除。
2. 如果用户用中文提问，但资料可能是英文论文、英文报告或中英混排，请补充必要英文检索词。
3. 如果用户的关键词或缩写可能写错，请保留原词，并追加你推测的候选标准写法；例如“NEP”可能需要追加“NSP”，但不要直接回答问题。
4. 不要编造知识库不存在的事实，只输出一行检索 query，不要解释。

用户问题：
{question}
"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=2))
def rewrite_query(question: str, active_documents: list[str] | None = None) -> str:
    """Use the LLM only for query rewriting, never for answering."""

    llm = create_chat_llm()
    if llm is None:
        return rule_based_rewrite(question)
    documents = "\n".join(active_documents or []) or "unknown"
    response = llm.invoke(REWRITE_PROMPT.format(question=question, documents=documents))
    rewritten = response.content.strip().replace("\n", " ")
    return rewritten or question


def safe_rewrite_query(question: str, active_documents: list[str] | None = None) -> str:
    """Stable query rewrite entrypoint.

    The policy is intentionally layered:
    - deterministic document context binding;
    - deterministic bilingual expansion for common RAG / paper terms;
    - optional LLM rewrite for Chinese question + English corpus / acronyms / fuzzy wording.
    """

    active_documents = active_documents or []
    document_aware_query = apply_document_context(question, active_documents)
    rule_query = expand_bilingual_terms(rule_based_rewrite(document_aware_query), active_documents)

    if should_use_llm_rewrite(question, active_documents):
        try:
            llm_query = expand_bilingual_terms(rewrite_query(document_aware_query, active_documents), active_documents)
            return merge_queries(rule_query, llm_query)
        except Exception:
            return rule_query

    return rule_query


def apply_document_context(question: str, active_documents: list[str]) -> str:
    """Bind vague references such as 'this paper' to the only active document."""

    candidate_documents = _document_like_names(active_documents)
    if len(candidate_documents) != 1:
        return question

    doc_name = Path(candidate_documents[0]).stem
    references_current_document = any(
        phrase in question
        for phrase in (
            "这篇论文",
            "本文",
            "这份文档",
            "这个文档",
            "该文档",
            "这份报告",
            "这个报告",
        )
    )
    if references_current_document and doc_name.casefold() not in question.casefold():
        return f"{doc_name} {question}"
    return question


def _document_like_names(active_documents: list[str]) -> list[str]:
    """Prefer real document files when the collection also contains standalone images."""

    document_suffixes = {".pdf", ".doc", ".docx", ".txt", ".md", ".markdown"}
    document_like = [
        name
        for name in active_documents
        if Path(name).suffix.casefold() in document_suffixes
    ]
    return document_like or active_documents


def expand_bilingual_terms(query: str, active_documents: list[str]) -> str:
    """Add lightweight bilingual retrieval terms without answering the question."""

    expanded_terms: list[str] = []
    lower_query = query.lower()
    active_doc_text = " ".join(Path(name).stem.lower() for name in active_documents)

    def add_if_needed(markers: tuple[str, ...], terms: str) -> None:
        if any(marker.lower() in lower_query for marker in markers):
            expanded_terms.append(terms)

    add_if_needed(
        ("模型名称", "模型全称", "全称", "提出的模型", "模型叫什么", "叫什么"),
        "model name full name stands for",
    )
    add_if_needed(
        ("预训练任务", "预训练目标", "训练任务", "两大任务", "包含mlm", "包含nsp", "包含nep"),
        "pre-training tasks pre-training objectives masked language model masked LM MLM next sentence prediction NSP",
    )
    add_if_needed(
        ("掩码语言模型", "遮盖语言模型", "mlm"),
        "masked language model masked LM MLM",
    )
    add_if_needed(
        ("下一句预测", "句子预测", "nsp", "nep"),
        "next sentence prediction NSP",
    )
    add_if_needed(
        ("图说是", "图中", "图里", "流程图", "架构图", "图表示", "图表达", "图片内容"),
        "figure diagram image caption input representation architecture",
    )
    add_if_needed(
        ("输入表示", "输入表征", "embedding", "嵌入"),
        "input representation token embeddings segment embeddings position embeddings",
    )

    if ("bert" in lower_query or "bert" in active_doc_text) and any(
        marker in lower_query
        for marker in ("mlm", "nsp", "nep", "预训练", "训练任务", "两大任务", "task", "objective")
    ):
        expanded_terms.append(
            "BERT pre-training tasks pre-training objectives masked language model masked LM MLM "
            "next sentence prediction NSP Section 3.1"
        )

    if not expanded_terms:
        return query
    return merge_queries(query, " ".join(expanded_terms))


def should_use_llm_rewrite(question: str, active_documents: list[str]) -> bool:
    """Decide when the extra LLM rewrite cost is worth it."""

    if needs_llm_rewrite(question):
        return True
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", question))
    has_acronym = bool(re.search(r"\b[A-Z][A-Z0-9]{1,}\b", question))
    likely_english_docs = any(re.search(r"[A-Za-z]{3,}", Path(name).stem) for name in active_documents)
    return has_chinese and (has_acronym or likely_english_docs)


def needs_llm_rewrite(question: str) -> bool:
    """Detect vague or typo-prone natural language questions."""

    ambiguous_phrases = (
        "它",
        "那个",
        "这套",
        "这个功能",
        "啥",
        "咋",
        "怎么回事",
        "为什么要这样做",
        "我记着",
        "是不是",
        "好像",
    )
    return any(phrase in question for phrase in ambiguous_phrases)


def rule_based_rewrite(question: str) -> str:
    """Small deterministic cleanup before retrieval."""

    replacements = {
        "BM24": "BM25",
        "Regas": "RAGAS",
        "regas": "RAGAS",
        "NEP": "NEP NSP",
        "nep": "nep nsp",
    }
    rewritten = question
    for source, target in replacements.items():
        rewritten = rewritten.replace(source, target)
    return rewritten


def merge_queries(*queries: str) -> str:
    """Merge query strings while keeping order and avoiding repeated tokens."""

    merged_tokens: list[str] = []
    seen: set[str] = set()
    for query in queries:
        for token in query.split():
            key = token.casefold()
            if key in seen:
                continue
            merged_tokens.append(token)
            seen.add(key)
    return " ".join(merged_tokens).strip()
