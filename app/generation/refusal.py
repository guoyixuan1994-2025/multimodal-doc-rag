from __future__ import annotations

from app.core.config import get_settings
from app.retrieval.reranker import RerankHit


def should_refuse(hits: list[RerankHit]) -> bool:
    settings = get_settings()
    if not hits:
        return True
    return hits[0].score < settings.min_rerank_score
