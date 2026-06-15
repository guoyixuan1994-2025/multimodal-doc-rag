from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.indexing.bm25_store import BM25Store
from app.indexing.vector_store import open_vector_store
from app.schemas.document import ChunkRecord


@dataclass
class HybridHit:
    chunk: ChunkRecord
    score: float
    vector_score: float
    bm25_score: float


class HybridRetriever:
    def __init__(self, chunks: list[ChunkRecord], collection_name: str | None = None) -> None:
        self.settings = get_settings()
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.bm25_store = BM25Store(chunks)
        self.vector_store = open_vector_store(collection_name=collection_name)

    def retrieve(self, query: str) -> list[HybridHit]:
        vector_scores = self._vector_search(query)
        bm25_scores = self._bm25_search(query)

        all_ids = set(vector_scores) | set(bm25_scores)
        hits: list[HybridHit] = []
        for chunk_id in all_ids:
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            vector_score = vector_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            # 第一版固定权重：语义 0.6，关键词 0.4。后续可以通过评估集调参。
            final_score = 0.6 * vector_score + 0.4 * bm25_score
            hits.append(HybridHit(chunk=chunk, score=final_score, vector_score=vector_score, bm25_score=bm25_score))

        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def _vector_search(self, query: str) -> dict[str, float]:
        results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=self.settings.vector_top_k,
        )
        if not results:
            return {}
        return {
            str(doc.metadata.get("chunk_id")): float(score)
            for doc, score in results
            if doc.metadata.get("chunk_id")
        }

    def _bm25_search(self, query: str) -> dict[str, float]:
        hits = self.bm25_store.search(query, top_k=self.settings.bm25_top_k)
        if not hits:
            return {}
        max_score = max(hit.score for hit in hits) or 1.0
        return {hit.chunk.chunk_id: hit.score / max_score for hit in hits}
