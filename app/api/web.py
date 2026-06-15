from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import HTMLResponse

from app.core.auth import require_api_key
from app.core.config import BASE_DIR, get_settings


router = APIRouter()


@router.get("/web", response_class=HTMLResponse)
def web_demo() -> str:
    html_path = BASE_DIR / "web_demo" / "index.html"
    return html_path.read_text(encoding="utf-8")


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), _: None = Depends(require_api_key)) -> dict:
    settings = get_settings()
    target_path = settings.raw_docs_dir / Path(file.filename or "uploaded_file").name
    content = await file.read()
    target_path.write_bytes(content)
    return {
        "success": True,
        "file_name": target_path.name,
        "file_path": str(target_path),
        "message": "文件已上传。可调用 /documents/ingest-path 传入 file_path 解析入库。",
    }
