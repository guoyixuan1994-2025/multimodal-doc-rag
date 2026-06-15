from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.schemas.document import ChunkRecord


@dataclass
class RerankHit:
    chunk: ChunkRecord
    score: float
    hybrid_score: float


class Reranker:
    """
    真实 BGE reranker。

    模型第一次加载会比较慢，后续会从配置的 Hugging Face 缓存目录读取。
    如果模型加载失败，自动回退到 hybrid_score，保证服务不中断。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    def rerank(self, query: str, candidates: list[tuple[ChunkRecord, float]], top_n: int) -> list[RerankHit]:
        if not candidates:
            return []

        model_pair = self._load_model()
        if model_pair is None:
            hits = [
                RerankHit(chunk=chunk, score=hybrid_score, hybrid_score=hybrid_score)
                for chunk, hybrid_score in candidates
            ]
            return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_n]

        tokenizer, model = model_pair
        try:
            import torch

            pairs = [(query, chunk.content) for chunk, _ in candidates]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            with torch.no_grad():
                outputs = model(**inputs)
            raw_scores = outputs.logits.squeeze(-1).tolist()
            scores = raw_scores if isinstance(raw_scores, list) else [float(raw_scores)]
        except Exception:
            # Some torch/transformers combinations can fail during inference
            # after loading successfully. Keep the RAG request available by
            # falling back to the already-computed hybrid retrieval scores.
            self._model = None
            return self._fallback_hits(candidates, top_n)

        hits = []
        for (chunk, hybrid_score), rerank_score in zip(candidates, scores):
            hits.append(
                RerankHit(
                    chunk=chunk,
                    score=float(rerank_score),
                    hybrid_score=hybrid_score,
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_n]

    @staticmethod
    def _fallback_hits(candidates: list[tuple[ChunkRecord, float]], top_n: int) -> list[RerankHit]:
        hits = [
            RerankHit(chunk=chunk, score=hybrid_score, hybrid_score=hybrid_score)
            for chunk, hybrid_score in candidates
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_n]

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.settings.reranker_model_name,
                cache_dir=self.settings.hf_cache_dir,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.reranker_model_name,
                cache_dir=self.settings.hf_cache_dir,
            )
            model.eval()
            self._model = (tokenizer, model)
            return self._model
        except Exception:
            self._model = None
            return None
