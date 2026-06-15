from __future__ import annotations

from pydantic import BaseModel


class EvalCase(BaseModel):
    case_id: str
    question: str
    reference_answer: str
    expected_chunk_ids: list[str]
    should_refuse: bool = False


class EvalRow(BaseModel):
    case_id: str
    question: str
    answer: str
    reference_answer: str
    expected_chunk_ids: str
    hit_chunk_ids: str
    retrieved_contexts: str
    hit: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    grounded: bool
