from __future__ import annotations

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    file_name: str | None = None
    page: int | None = None
    title: str | None = None
    chunk_type: str | None = None
    bbox: str | list[float] | None = None
    score: float | None = None
    rerank_score: float | None = None
    text: str


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None
    collection_name: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    question: str
    rewritten_query: str
    answer: str
    grounded: bool
    sources: list[SourceChunk] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class SearchResponse(BaseModel):
    request_id: str
    query: str
    rewritten_query: str | None = None
    hits: list[SourceChunk]
