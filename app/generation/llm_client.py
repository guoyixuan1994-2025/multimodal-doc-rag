from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import get_settings


def create_chat_llm() -> ChatOpenAI | None:
    settings = get_settings()
    if not settings.llm_api_key:
        return None
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )
