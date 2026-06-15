from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """简单 API Key 权限层。

    本地学习时默认不开启；上线时配置 APP_API_KEY 后，所有关键接口都要求请求头：
    X-API-Key: <your-key>
    """
    expected = get_settings().app_api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少或错误的 X-API-Key。",
        )
