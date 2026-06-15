from __future__ import annotations

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from app.core.config import get_settings


def create_embeddings() -> FastEmbedEmbeddings:
    settings = get_settings()
    return FastEmbedEmbeddings(
        model_name=settings.embedding_model_name,
        cache_dir=settings.fastembed_cache_dir,
    )
