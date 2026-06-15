from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader

from app.core.config import get_settings
from app.schemas.document import ParsedBlock, ParsedDocument


class ParseCancelledError(RuntimeError):
    """Raised when a user cancels a running document parse."""


def stop_process_tree(process: subprocess.Popen) -> None:
    """Stop MinerU and any model-worker child processes it started."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def parse_with_mineru(
    path: Path,
    doc_id: str,
    analysis_mode: str = "auto",
    cancel_check: Callable[[], bool] | None = None,
) -> ParsedDocument:
    """
    MinerU 真实解析入口。

    MinerU 会对 PDF / 图片 / Office 文档执行版面分析、OCR、表格/公式与图片理解，
    并产出 Markdown/JSON 等结构化文件。
    当前 RAG 入库读取 Markdown 主内容并转成统一 ParsedBlock；MinerU 产出的 JSON
    和图片等中间结果会保留在输出目录，后续可继续绑定 bbox、表格或图片引用。

    为了控制真实项目里的等待时间，文字型 PDF 默认使用快速文本版面解析；
    扫描件、图片或用户明确指定 full/ocr 时，才启用重型 OCR/VLM 解析。
    """

    output_root = get_settings().parsed_dir / "mineru_outputs" / doc_id
    settings = get_settings()
    output_root.mkdir(parents=True, exist_ok=True)
    profile = choose_parse_profile(path, analysis_mode)

    mineru_cli = resolve_mineru_cli()
    mineru_exe = Path(mineru_cli or Path(sys.executable).parent / "Scripts" / "mineru.exe")
    if mineru_cli is None:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            return parse_image_without_mineru(path, doc_id)
        raise RuntimeError(f"未找到 MinerU 命令行工具：{mineru_exe}")

    command = [
        str(mineru_exe),
        "-p",
        str(path),
        "-o",
        str(output_root),
        "-b",
        profile["backend"],
        "-m",
        profile["method"],
        "-l",
        "ch",
        "-f",
        str(settings.mineru_formula).lower(),
        "-t",
        str(settings.mineru_table).lower(),
    ]
    # MinerU 与模型下载工具都使用统一缓存目录，避免模型散落到系统盘。
    env = os.environ.copy()
    env["HF_HOME"] = settings.hf_cache_dir
    env["HF_HUB_CACHE"] = str(Path(settings.hf_cache_dir) / "hub")
    env["MODELSCOPE_CACHE"] = settings.modelscope_cache_dir
    if cancel_check and cancel_check():
        raise ParseCancelledError("文档解析任务已取消。")

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="ignore") as stdout_file:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="ignore") as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=env,
                creationflags=creationflags,
            )
            while process.poll() is None:
                if cancel_check and cancel_check():
                    stop_process_tree(process)
                    raise ParseCancelledError("文档解析任务已取消。")
                time.sleep(0.2)
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
            returncode = process.returncode

    if returncode != 0:
        raise RuntimeError(
            "MinerU 解析失败。\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    content_list_path = find_content_list_json(output_root)
    if content_list_path is not None:
        blocks = content_list_to_blocks(content_list_path, doc_id)
        if profile["method"] == "ocr":
            for block in blocks:
                if block.block_type == "text":
                    block.block_type = "ocr_text"
        if blocks:
            return ParsedDocument(
                doc_id=doc_id,
                file_name=path.name,
                file_type=path.suffix.lower().lstrip("."),
                source_path=str(path),
                parser_mode=profile["label"],
                blocks=blocks,
            )

    md_files = sorted(output_root.rglob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not md_files:
        raise RuntimeError(f"MinerU 已运行，但没有在 {output_root} 下找到 Markdown 输出。")

    markdown_path = md_files[0]
    markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    blocks = markdown_to_blocks(markdown_text, doc_id)
    if profile["method"] == "ocr":
        for block in blocks:
            if block.block_type == "text":
                block.block_type = "ocr_text"

    return ParsedDocument(
        doc_id=doc_id,
        file_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        source_path=str(path),
        parser_mode=profile["label"],
        blocks=blocks,
    )


def resolve_mineru_cli() -> str | None:
    """Resolve MinerU CLI on Windows virtualenvs and Linux containers."""

    configured = os.getenv("MINERU_CLI")
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return str(candidate)
        resolved = shutil.which(configured)
        if resolved:
            return resolved

    for command_name in ("mineru", "magic-pdf"):
        resolved = shutil.which(command_name)
        if resolved:
            return resolved

    windows_candidate = Path(sys.executable).parent / "Scripts" / "mineru.exe"
    if windows_candidate.exists():
        return str(windows_candidate)
    return None


def parse_image_without_mineru(path: Path, doc_id: str) -> ParsedDocument:
    """Lightweight fallback for Docker images when MinerU is not installed."""

    width = height = None
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        pass

    size_text = f"{width}x{height}" if width and height else "unknown"
    content = (
        f"图片文件：{path.name}\n"
        f"图片尺寸：{size_text}\n"
        "图片解析说明：当前运行环境未安装 MinerU，"
        "系统已将图片作为 image_caption 类型入库，但未执行 OCR/VLM 内容理解。"
    )
    return ParsedDocument(
        doc_id=doc_id,
        file_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        source_path=str(path),
        parser_mode="image_metadata_fallback",
        blocks=[
            ParsedBlock(
                block_id=f"{doc_id}-block-1",
                block_type="image_caption",
                content=content,
                page=1,
                title=path.stem,
                metadata={
                    "parser_warning": "mineru_cli_not_found",
                    "width": width,
                    "height": height,
                },
            )
        ],
    )


def find_content_list_json(output_root: Path) -> Path | None:
    """优先读取 MinerU 的结构化 content_list，而不是只读最终 Markdown。"""
    candidates = [
        path
        for path in output_root.rglob("*content_list.json")
        if "content_list_v2" not in path.name
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def content_list_to_blocks(content_list_path: Path, doc_id: str) -> list[ParsedBlock]:
    data = json.loads(content_list_path.read_text(encoding="utf-8", errors="ignore"))
    blocks: list[ParsedBlock] = []
    current_title: str | None = None

    for index, item in enumerate(data, start=1):
        block = content_item_to_block(item, doc_id, index, current_title, content_list_path)
        if block is None:
            continue
        if item.get("type") == "text" and item.get("text_level"):
            current_title = clean_text(item.get("text"))
            block.title = current_title
        blocks.append(block)

    return blocks


def content_item_to_block(
    item: dict[str, Any],
    doc_id: str,
    index: int,
    current_title: str | None,
    content_list_path: Path,
) -> ParsedBlock | None:
    item_type = str(item.get("type") or "text")
    page_idx = item.get("page_idx")
    page = int(page_idx) + 1 if isinstance(page_idx, int) else None
    bbox = normalize_bbox(item.get("bbox"))
    metadata: dict[str, str | int | float | None] = {
        "element_type": item_type,
        "sub_type": item.get("sub_type"),
        "text_level": item.get("text_level"),
    }

    block_type = "text"
    image_path = None
    content = ""

    if item_type == "text":
        content = clean_text(item.get("text"))
        if not content:
            return None
        if item.get("text_level"):
            level = max(1, min(int(item.get("text_level") or 1), 6))
            content = f"{'#' * level} {content}"

    elif item_type == "image":
        block_type = "image_caption"
        image_path = resolve_relative_output_path(content_list_path, item.get("img_path"))
        caption = clean_text(item.get("content")) or clean_text(item.get("text"))
        caption_lines = []
        if caption:
            caption_lines.append(f"图片描述：{caption}")
        for label, values in (("图片标题", item.get("image_caption")), ("图片脚注", item.get("image_footnote"))):
            normalized = normalize_list_text(values)
            if normalized:
                caption_lines.append(f"{label}：{normalized}")
        content = "\n".join(caption_lines).strip()
        if not content:
            content = "MinerU 识别到图片区域，但未识别到可读文字标签或图片说明。"

    elif item_type == "table":
        block_type = "table"
        table_content = (
            clean_text(item.get("table_body"))
            or clean_text(item.get("table_html"))
            or clean_text(item.get("text"))
            or clean_text(item.get("content"))
        )
        table_caption = normalize_list_text(item.get("table_caption"))
        table_footnote = normalize_list_text(item.get("table_footnote"))
        parts = []
        if table_caption:
            parts.append(f"表格标题：{table_caption}")
        if table_content:
            parts.append(f"表格内容：\n{table_content}")
        if table_footnote:
            parts.append(f"表格脚注：{table_footnote}")
        content = "\n".join(parts).strip()
        if not content:
            return None

    elif item_type in {"equation", "formula", "interline_equation"}:
        block_type = "formula"
        latex = clean_text(item.get("latex") or item.get("text") or item.get("content"))
        content = f"公式：{latex}" if latex else ""
        if not content:
            return None

    else:
        content = clean_text(item.get("text") or item.get("content"))
        if not content:
            return None

    return ParsedBlock(
        block_id=f"{doc_id}-block-{index}",
        block_type=block_type,
        content=content,
        page=page,
        title=current_title,
        image_path=image_path,
        bbox=bbox,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def resolve_relative_output_path(content_list_path: Path, relative_path: Any) -> str | None:
    if not relative_path:
        return None
    candidate = content_list_path.parent / str(relative_path)
    return str(candidate)


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    normalized: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            normalized.append(float(item))
    return normalized or None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_list_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return "；".join(clean_text(item) for item in value if clean_text(item))
    return clean_text(value)


def choose_parse_profile(path: Path, analysis_mode: str) -> dict[str, str | bool]:
    settings = get_settings()
    full_backend = settings.mineru_backend or "hybrid-auto-engine"
    """根据文件是否已有可读取文字层，选择速度和能力更匹配的 MinerU 路径。"""
    if analysis_mode == "text":
        return {
            "backend": "pipeline",
            "method": "txt",
            "image_analysis": False,
            "label": "mineru_text_layout_fast",
        }
    if analysis_mode == "ocr":
        return {
            "backend": full_backend,
            "method": "ocr",
            "image_analysis": True,
            "label": "mineru_ocr_full",
        }
    if analysis_mode == "full":
        return {
            "backend": full_backend,
            "method": "auto",
            "image_analysis": True,
            "label": "mineru_multimodal_full",
        }

    if path.suffix.lower() == ".pdf" and pdf_has_text_layer(path):
        return {
            "backend": "pipeline",
            "method": "txt",
            "image_analysis": False,
            "label": "mineru_text_layout_fast",
        }
    return {
        "backend": full_backend,
        "method": "auto",
        "image_analysis": True,
        "label": "mineru_multimodal_full",
    }


def pdf_has_text_layer(path: Path) -> bool:
    """抽样读取 PDF 前三页；文字足够多时，避免对整篇论文做无必要的 OCR。"""
    try:
        reader = PdfReader(str(path))
        sampled_text = "".join((page.extract_text() or "") for page in reader.pages[:3])
        return len(sampled_text.strip()) >= 200
    except Exception:
        return False


def markdown_to_blocks(markdown_text: str, doc_id: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    current_title = None
    current_lines: list[str] = []
    block_index = 0

    def flush_block() -> None:
        nonlocal block_index, current_lines
        content = "\n".join(line.strip() for line in current_lines if line.strip()).strip()
        if not content:
            current_lines = []
            return
        block_index += 1
        block_type = "image_caption" if "<summary>natural_image</summary>" in content else "text"
        blocks.append(
            ParsedBlock(
                block_id=f"{doc_id}-block-{block_index}",
                block_type=block_type,
                content=content,
                title=current_title,
            )
        )
        current_lines = []

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush_block()
            current_title = stripped.strip("# ").strip()
        current_lines.append(line)
    flush_block()

    if not blocks and markdown_text.strip():
        blocks.append(
            ParsedBlock(
                block_id=f"{doc_id}-block-1",
                block_type="text",
                content=markdown_text.strip(),
            )
        )
    return blocks
