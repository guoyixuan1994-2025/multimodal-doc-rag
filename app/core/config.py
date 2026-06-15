from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


def load_project_env() -> None:
    load_dotenv(BASE_DIR / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    project_name: str = "政企多模态文档解析智能 RAG 检索问答系统"
    output_dir: Path = Path(os.getenv("PROJECT_OUTPUT_DIR", str(BASE_DIR / "data")))
    app_api_key: str | None = os.getenv("APP_API_KEY")
    default_collection_name: str = os.getenv("DEFAULT_COLLECTION_NAME", "enterprise_multimodal_doc_rag")
    eval_collection_name: str = os.getenv("EVAL_COLLECTION_NAME", "enterprise_multimodal_doc_rag_eval")

    llm_api_key: str | None = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    llm_base_url: str = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    llm_model: str = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))

    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    fastembed_cache_dir: str = os.getenv("FASTEMBED_CACHE_DIR", str(BASE_DIR / ".cache" / "fastembed"))
    hf_cache_dir: str = os.getenv("HF_CACHE_DIR", str(BASE_DIR / ".cache" / "huggingface"))
    modelscope_cache_dir: str = os.getenv("MODELSCOPE_CACHE_DIR", str(BASE_DIR / ".cache" / "modelscope"))

    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "5"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "5"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "3"))
    min_rerank_score: float = float(os.getenv("MIN_RERANK_SCORE", "0.2"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    min_chunk_chars: int = int(os.getenv("MIN_CHUNK_CHARS", "40"))
    parse_job_poll_interval_seconds: float = float(os.getenv("PARSE_JOB_POLL_INTERVAL_SECONDS", "1"))
    mineru_backend: str = os.getenv("MINERU_BACKEND", "hybrid-auto-engine")
    mineru_method: str = os.getenv("MINERU_METHOD", "auto")
    mineru_formula: bool = os.getenv("MINERU_FORMULA", "true").lower() == "true"
    mineru_table: bool = os.getenv("MINERU_TABLE", "true").lower() == "true"
    mineru_image_analysis: bool = os.getenv("MINERU_IMAGE_ANALYSIS", "true").lower() == "true"

    @property
    def raw_docs_dir(self) -> Path:
        return self.output_dir / "raw_docs"

    @property
    def parsed_dir(self) -> Path:
        return self.output_dir / "parsed"

    @property
    def chroma_dir(self) -> Path:
        return self.output_dir / "chroma_store"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"


@lru_cache
def get_settings() -> Settings:
    load_project_env()
    settings = Settings()
    settings.llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    settings.llm_base_url = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or settings.llm_base_url
    settings.llm_model = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or settings.llm_model
    settings.llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", str(settings.llm_timeout_seconds)))
    settings.output_dir = Path(os.getenv("PROJECT_OUTPUT_DIR", str(settings.output_dir)))
    settings.app_api_key = os.getenv("APP_API_KEY", settings.app_api_key)
    settings.default_collection_name = os.getenv("DEFAULT_COLLECTION_NAME", settings.default_collection_name)
    settings.eval_collection_name = os.getenv("EVAL_COLLECTION_NAME", settings.eval_collection_name)
    settings.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", settings.embedding_model_name)
    settings.reranker_model_name = os.getenv("RERANKER_MODEL_NAME", settings.reranker_model_name)
    settings.fastembed_cache_dir = os.getenv("FASTEMBED_CACHE_DIR", settings.fastembed_cache_dir)
    settings.hf_cache_dir = os.getenv("HF_CACHE_DIR", settings.hf_cache_dir)
    settings.modelscope_cache_dir = os.getenv("MODELSCOPE_CACHE_DIR", settings.modelscope_cache_dir)
    settings.mineru_backend = os.getenv("MINERU_BACKEND", settings.mineru_backend)
    settings.mineru_method = os.getenv("MINERU_METHOD", settings.mineru_method)
    settings.mineru_formula = os.getenv("MINERU_FORMULA", str(settings.mineru_formula)).lower() == "true"
    settings.mineru_table = os.getenv("MINERU_TABLE", str(settings.mineru_table)).lower() == "true"
    settings.mineru_image_analysis = os.getenv("MINERU_IMAGE_ANALYSIS", str(settings.mineru_image_analysis)).lower() == "true"
    settings.chunk_size = int(os.getenv("CHUNK_SIZE", str(settings.chunk_size)))
    settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(settings.chunk_overlap)))
    settings.min_chunk_chars = int(os.getenv("MIN_CHUNK_CHARS", str(settings.min_chunk_chars)))
    settings.parse_job_poll_interval_seconds = float(
        os.getenv("PARSE_JOB_POLL_INTERVAL_SECONDS", str(settings.parse_job_poll_interval_seconds))
    )
    settings.raw_docs_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    Path(settings.modelscope_cache_dir).mkdir(parents=True, exist_ok=True)
    return settings
