from __future__ import annotations

import csv
import io
import sys
import types
from pathlib import Path

import pandas as pd
from langchain_core.embeddings import Embeddings

from app.core.config import BASE_DIR, get_settings
from app.generation.llm_client import create_chat_llm
from app.indexing.embeddings import create_embeddings
from app.parsers.file_router import build_doc_id
from app.schemas.eval import EvalCase, EvalRow
from app.services.rag_service import RagService


class RagasEmbeddingAdapter(Embeddings):
    """让 RAGAS 使用本地 BGE，同时避免将内部模型对象误记为模型名称。"""

    def __init__(self) -> None:
        self._embedding_client = create_embeddings()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedding_client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedding_client.embed_query(text)


def ensure_eval_import_stubs() -> None:
    # 当前 Windows 学习环境中，部分标准库 DLL 权限偶发影响 ragas/matplotlib 导入。
    # 这些 stub 只服务于本评估脚本，不会改动你的 Python 环境。
    if "lzma" not in sys.modules:
        module = types.ModuleType("lzma")
        module.LZMAError = RuntimeError
        module.FORMAT_AUTO = 0
        module.open = lambda *args, **kwargs: io.BytesIO()
        module.compress = lambda data, *args, **kwargs: data
        module.decompress = lambda data, *args, **kwargs: data
        sys.modules["lzma"] = module

    if "plistlib" not in sys.modules:
        module = types.ModuleType("plistlib")

        class InvalidFileException(Exception):
            pass

        module.InvalidFileException = InvalidFileException
        module.loads = lambda *args, **kwargs: []
        sys.modules["plistlib"] = module

    vertex_module = "langchain_community.chat_models.vertexai"
    if vertex_module not in sys.modules:
        module = types.ModuleType(vertex_module)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[vertex_module] = module


def default_eval_cases() -> list[EvalCase]:
    notes_doc_id = build_doc_id(BASE_DIR / "sample_docs" / "rag_project_notes.md")
    terms_doc_id = build_doc_id(BASE_DIR / "sample_docs" / "telecom_ai_terms.txt")
    return [
        EvalCase(
            case_id="case-1",
            question="这个系统的核心流程是什么？",
            reference_answer="系统流程包括文档解析、文本清洗、chunk 切分、embedding 入库、混合检索、reranker 精排和大模型生成。",
            expected_chunk_ids=[f"{notes_doc_id}-chunk-2"],
        ),
        EvalCase(
            case_id="case-2",
            question="为什么要用 BM25 和向量混合检索？",
            reference_answer="BM25 适合关键词和专有名词匹配，向量检索适合语义相近表达，混合检索可以同时利用两种信号。",
            expected_chunk_ids=[f"{notes_doc_id}-chunk-4", f"{terms_doc_id}-chunk-1"],
        ),
        EvalCase(
            case_id="case-3",
            question="RAGAS 可以评估什么？",
            reference_answer="RAGAS 可以评估 answer relevancy、faithfulness、context precision 和 context recall。",
            expected_chunk_ids=[f"{notes_doc_id}-chunk-5"],
        ),
        EvalCase(
            case_id="case-4",
            question="这个系统支持 Kubernetes 自动扩容吗？",
            reference_answer="资料中没有相关信息。",
            expected_chunk_ids=[],
            should_refuse=True,
        ),
    ]


def run_basic_eval(rag_service: RagService, cases: list[EvalCase] | None = None) -> pd.DataFrame:
    cases = cases or default_eval_cases()
    rows: list[EvalRow] = []

    for case in cases:
        response = rag_service.answer(case.question)
        hit_ids = [source.chunk_id for source in response.sources]
        retrieved_contexts = "\n\n".join(source.text for source in response.sources)
        metrics = retrieval_metrics(hit_ids, case.expected_chunk_ids)
        rows.append(
            EvalRow(
                case_id=case.case_id,
                question=case.question,
                answer=response.answer,
                reference_answer=case.reference_answer,
                expected_chunk_ids=";".join(case.expected_chunk_ids),
                hit_chunk_ids=";".join(hit_ids),
                retrieved_contexts=retrieved_contexts,
                hit=metrics["hit"],
                precision_at_k=metrics["precision_at_k"],
                recall_at_k=metrics["recall_at_k"],
                mrr=metrics["mrr"],
                grounded=response.grounded,
            )
        )
    return pd.DataFrame([row.model_dump() for row in rows])


def load_plotting_backend():
    """Use a non-interactive backend so report rendering is safe in API threads."""
    ensure_eval_import_stubs()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def run_ragas_offline_eval(df: pd.DataFrame) -> pd.DataFrame:
    # RAGAS 离线字符串指标：不依赖外部 LLM，适合先验证评估链路。
    ensure_eval_import_stubs()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics._string import ExactMatch, NonLLMStringSimilarity, StringPresence

    records = []
    for _, row in df.iterrows():
        contexts = [text for text in str(row.get("retrieved_contexts", "")).split("\n\n") if text.strip()]
        records.append(
            {
                "user_input": row["question"],
                "retrieved_contexts": contexts,
                "response": row["answer"],
                "reference": row["reference_answer"],
            }
        )

    dataset = Dataset.from_list(records)
    result = evaluate(
        dataset,
        metrics=[
            ExactMatch(),
            StringPresence(),
            NonLLMStringSimilarity(),
        ],
    )
    return result.to_pandas()


def run_ragas_online_eval(df: pd.DataFrame) -> pd.DataFrame | None:
    """用真实 DeepSeek 评估回答相关性、上下文质量和忠实性。"""
    ensure_eval_import_stubs()
    llm = create_chat_llm()
    if llm is None:
        return None

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness

    records = []
    for _, row in df.iterrows():
        contexts = [text for text in str(row.get("retrieved_contexts", "")).split("\n\n") if text.strip()]
        records.append(
            {
                "user_input": row["question"],
                "retrieved_contexts": contexts,
                "response": row["answer"],
                "reference": row["reference_answer"],
            }
        )
    result = evaluate(
        Dataset.from_list(records),
        metrics=[answer_relevancy, context_precision, context_recall, faithfulness],
        llm=llm,
        embeddings=RagasEmbeddingAdapter(),
    )
    return result.to_pandas()


def retrieval_metrics(hit_ids: list[str], expected_ids: list[str]) -> dict[str, float]:
    if not expected_ids:
        is_correct_refusal = 1.0 if not hit_ids else 0.0
        return {
            "hit": is_correct_refusal,
            "precision_at_k": is_correct_refusal,
            "recall_at_k": is_correct_refusal,
            "mrr": is_correct_refusal,
        }

    expected = set(expected_ids)
    hits = set(hit_ids)
    overlap = expected & hits
    first_rank = 0
    for index, chunk_id in enumerate(hit_ids, start=1):
        if chunk_id in expected:
            first_rank = index
            break
    return {
        "hit": 1.0 if overlap else 0.0,
        "precision_at_k": len(overlap) / max(len(hit_ids), 1),
        "recall_at_k": len(overlap) / max(len(expected), 1),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def save_eval_outputs(df: pd.DataFrame) -> dict[str, Path]:
    plt = load_plotting_backend()

    settings = get_settings()
    report_dir = settings.output_dir / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "basic_eval_result.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    metric_names = ["hit", "precision_at_k", "recall_at_k", "mrr"]
    means = df[metric_names].mean()
    png_path = report_dir / "basic_eval_metrics.png"

    plt.figure(figsize=(9, 5))
    bars = plt.bar(metric_names, [means[name] for name in metric_names], color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"])
    plt.ylim(0, 1.05)
    plt.title("RAG Retrieval Metrics")
    plt.ylabel("score")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    return {"csv": csv_path, "png": png_path}


def save_ragas_outputs(ragas_df: pd.DataFrame) -> dict[str, Path]:
    plt = load_plotting_backend()

    settings = get_settings()
    report_dir = settings.output_dir / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "ragas_offline_result.csv"
    ragas_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    metric_names = [
        name
        for name in ["exact_match", "string_present", "non_llm_string_similarity"]
        if name in ragas_df.columns
    ]
    png_path = report_dir / "ragas_offline_metrics.png"
    means = ragas_df[metric_names].mean() if metric_names else pd.Series(dtype=float)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(metric_names, [means[name] for name in metric_names], color=["#3b82f6", "#10b981", "#f59e0b"])
    plt.ylim(0, 1.05)
    plt.title("RAGAS Offline Metrics")
    plt.ylabel("score")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    return {"csv": csv_path, "png": png_path}


def save_ragas_online_outputs(ragas_df: pd.DataFrame) -> dict[str, Path]:
    """保存在线 RAGAS 经典指标和 matplotlib 可视化图。"""
    plt = load_plotting_backend()

    report_dir = get_settings().output_dir / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "ragas_online_result.csv"
    ragas_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    metric_names = [
        name
        for name in ["answer_relevancy", "context_precision", "context_recall", "faithfulness"]
        if name in ragas_df.columns
    ]
    means = ragas_df[metric_names].mean() if metric_names else pd.Series(dtype=float)
    png_path = report_dir / "ragas_online_metrics.png"
    plt.figure(figsize=(10, 5))
    bars = plt.bar(metric_names, [means[name] for name in metric_names], color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"])
    plt.ylim(0, 1.05)
    plt.title("RAGAS Online Metrics (DeepSeek)")
    plt.ylabel("score")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    return {"csv": csv_path, "png": png_path}
