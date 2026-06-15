from __future__ import annotations

from app.retrieval.reranker import RerankHit


def build_rag_prompt(question: str, rewritten_query: str, hits: list[RerankHit]) -> str:
    context_blocks: list[str] = []
    for hit in hits:
        metadata = hit.chunk.metadata
        context_blocks.append(
            "\n".join(
                [
                    f"[chunk_id={hit.chunk.chunk_id}]",
                    (
                        f"file={metadata.get('file_name')} | page={metadata.get('page')} "
                        f"| type={metadata.get('chunk_type')} | title={metadata.get('title')} "
                        f"| bbox={metadata.get('bbox')}"
                    ),
                    hit.chunk.content,
                ]
            )
        )

    context = "\n\n".join(context_blocks) if context_blocks else "(没有检索到相关资料)"
    return f"""你是政企文档智能检索问答助手。

要求：
1. 只能根据【资料】回答。
2. 如果资料中没有答案，必须回答“资料中没有相关信息。”。
3. 回答要简洁清楚。
4. 如果能回答，要指出依据来自哪些 chunk_id。

原始问题：{question}
检索问题：{rewritten_query}

【资料】
{context}

请输出最终答案：
"""
