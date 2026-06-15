from __future__ import annotations

from pathlib import Path

from app.core.config import BASE_DIR
from app.core.logging import setup_logging
from app.services.document_service import DocumentService
from app.services.eval_service import (
    run_basic_eval,
    run_ragas_offline_eval,
    run_ragas_online_eval,
    save_eval_outputs,
    save_ragas_online_outputs,
    save_ragas_outputs,
)
from app.services.rag_service import RagService


def main() -> None:
    setup_logging()
    sample_dir = BASE_DIR / "sample_docs"
    paths = sorted([path for path in sample_dir.iterdir() if path.suffix.lower() in {".md", ".txt"}])

    document_service = DocumentService()
    document_service.ingest_paths(paths, reset=True)
    rag_service = RagService(document_service.get_chunks())

    df = run_basic_eval(rag_service)
    outputs = save_eval_outputs(df)
    ragas_df = run_ragas_offline_eval(df)
    ragas_outputs = save_ragas_outputs(ragas_df)
    online_df = run_ragas_online_eval(df)
    online_outputs = save_ragas_online_outputs(online_df) if online_df is not None else None

    print("=== 评估完成 ===")
    print(df.to_string(index=False))
    print("\n平均指标：")
    for name in ["hit", "precision_at_k", "recall_at_k", "mrr"]:
        print(f"- {name}: {df[name].mean():.4f}")
    print(f"\nCSV：{outputs['csv']}")
    print(f"图表：{outputs['png']}")
    print(f"RAGAS CSV：{ragas_outputs['csv']}")
    print(f"RAGAS 图表：{ragas_outputs['png']}")
    if online_outputs is None:
        print("RAGAS 在线指标：未检测到 DeepSeek key，本轮跳过。")
    else:
        print(f"RAGAS 在线 CSV：{online_outputs['csv']}")
        print(f"RAGAS 在线图表：{online_outputs['png']}")


if __name__ == "__main__":
    main()
