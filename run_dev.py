from __future__ import annotations

from app.core.config import BASE_DIR, get_settings
from app.core.logging import setup_logging
from app.services.document_service import DocumentService
from app.services.rag_service import RagService


def main() -> None:
    """用内置资料验证完整问答链路，不依赖浏览器操作。"""
    setup_logging()
    settings = get_settings()

    print("=== 政企多模态文档解析智能 RAG 检索问答系统：开发验证 ===")
    print(f"项目目录：{BASE_DIR}")
    print(f"输出目录：{settings.output_dir}")
    print(f"Embedding 模型：{settings.embedding_model_name}")
    print(f"Reranker 模型：{settings.reranker_model_name}")
    print(f"DeepSeek model：{settings.llm_model}")

    sample_dir = BASE_DIR / "sample_docs"
    paths = sorted(path for path in sample_dir.iterdir() if path.suffix.lower() in {".md", ".txt"})

    print("\nStep 1：解析并入库样例文档")
    document_service = DocumentService()
    results = document_service.ingest_paths(paths, reset=True)
    for result in results:
        print(
            f"- {result.file_name} | type={result.file_type} | "
            f"blocks={result.block_count} | chunks={result.chunk_count}"
        )
    print(f"总 chunk 数：{len(document_service.get_chunks())}")

    print("\nStep 2：RAG 问答验证")
    rag_service = RagService(document_service.get_chunks())
    questions = [
        "这个系统的核心流程是什么？",
        "为什么要用 BM25 和向量混合检索？",
        "RAGAS 可以评估什么？",
        "这个系统支持 Kubernetes 自动扩容吗？",
    ]

    for question in questions:
        print("\n" + "-" * 80)
        print(f"问题：{question}")
        response = rag_service.answer(question)
        print(f"改写后 query：{response.rewritten_query}")
        print(f"是否基于资料：{response.grounded}")
        print(f"回答：{response.answer}")
        print("引用来源：")
        if not response.sources:
            print("  无")
        for source in response.sources:
            print(
                f"  - {source.chunk_id} | file={source.file_name} | "
                f"score={source.score:.4f} | rerank={source.rerank_score:.4f}"
            )

    print("\n你需要理解的结论：")
    print("1. 主链路已跑通：解析 -> 分块 -> embedding -> Chroma -> 混合检索 -> 精排 -> 生成/拒答。")
    print("2. PDF / Office / 图片会进入真实 MinerU，执行版面分析、OCR、表格/公式与图片理解。")
    print("3. 精排使用本地真实 BGE reranker，不依赖在线 reranker 接口。")
    print("4. 评估脚本提供检索指标、RAGAS 指标和 matplotlib 图表；Web Demo 提供用户入口。")


if __name__ == "__main__":
    main()
