from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.schemas.document import ChunkRecord


def tokenize(text: str) -> list[str]:
    # 中文分词先用轻量规则：英文/数字保留词，中文做二元字符片段。
    lowered = text.lower()
    words = re.findall(r"[a-zA-Z0-9_]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]+", lowered)
    grams: list[str] = []
    for segment in chinese:
        grams.append(segment)
        grams.extend(segment[index : index + 2] for index in range(max(len(segment) - 1, 0)))
    return words + grams


@dataclass
class BM25Hit:
    chunk: ChunkRecord
    score: float


class BM25Store:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.tokenized_docs = [tokenize(chunk.content) for chunk in chunks]
        self.model = BM25Okapi(self.tokenized_docs) if chunks else None

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        if not self.model:
            return []
        scores = self.model.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [BM25Hit(chunk=self.chunks[index], score=float(score)) for index, score in ranked if score > 0]
