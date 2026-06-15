from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8090")
PROJECT_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="政企多模态文档 RAG 控制台",
    page_icon="",
    layout="wide",
)


def init_state() -> None:
    defaults = {
        "uploaded_file_path": "",
        "last_job": None,
        "last_profile": None,
        "last_result": None,
        "chat_answer": "",
        "chat_sources": [],
        "eval_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def api_headers() -> dict[str, str]:
    api_key = st.session_state.get("api_key", "").strip()
    if not api_key:
        return {}
    if api_key.lower().startswith("sk-"):
        raise RuntimeError("请填写服务端 APP_API_KEY，不要在页面中填写或展示 LLM_API_KEY。")
    return {"X-API-Key": api_key}


def api_url(path: str) -> str:
    return f"{st.session_state.api_base.rstrip('/')}{path}"


def request_json(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    headers = kwargs.pop("headers", {})
    merged_headers = {**api_headers(), **headers}
    response = requests.request(method, api_url(path), headers=merged_headers, timeout=600, **kwargs)
    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text}
    if not response.ok:
        raise RuntimeError(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def parse_sse_line_buffer(buffer: str) -> tuple[list[dict[str, Any]], str]:
    events: list[dict[str, Any]] = []
    while "\n\n" in buffer:
        block, buffer = buffer.split("\n\n", 1)
        event_name = "message"
        payload = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                payload += line.removeprefix("data:").strip()
        if payload:
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"text": payload}
            events.append({"event": event_name, "data": data})
    return events, buffer


def upload_file(uploaded_file) -> dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    return request_json("POST", "/documents/upload", files=files)


def create_parse_job(
    file_path: str,
    analysis_mode: str,
    collection_name: str,
    reset: bool = False,
) -> dict[str, Any]:
    params = {
        "file_path": file_path,
        "analysis_mode": analysis_mode,
        "reset": str(reset).lower(),
    }
    if collection_name.strip():
        params["collection_name"] = collection_name.strip()
    return request_json("POST", "/documents/parse-jobs", params=params)


def poll_job(job_id: str) -> dict[str, Any]:
    return request_json("GET", f"/documents/parse-jobs/{job_id}")


def cancel_parse_job(job_id: str) -> dict[str, Any]:
    return request_json("POST", f"/documents/parse-jobs/{job_id}/cancel")


def ingest_directory(
    directory_path: str,
    analysis_mode: str,
    collection_name: str,
    recursive: bool = True,
    reset: bool = False,
) -> dict[str, Any]:
    params = {
        "directory_path": directory_path,
        "analysis_mode": analysis_mode,
        "recursive": str(recursive).lower(),
        "reset": str(reset).lower(),
    }
    if collection_name.strip():
        params["collection_name"] = collection_name.strip()
    return request_json("POST", "/documents/ingest-directory", params=params)


def render_json_text(data: Any) -> None:
    """Render debug payloads without Streamlit's lazy-loaded JSON component."""

    text = json.dumps(data, ensure_ascii=False, indent=2).replace("```", "` ` `")
    st.markdown(f"```json\n{text}\n```")


def render_simple_table(headers: list[str], rows: list[list[Any]]) -> None:
    def clean(value: Any) -> str:
        return str(value if value is not None else "-").replace("|", "\\|").replace("\n", " ")

    table = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    table.extend(f"| {' | '.join(clean(value) for value in row)} |" for row in rows)
    st.markdown("\n".join(table))


def show_profile(profile: Any) -> None:
    if not profile:
        st.info("当前任务没有返回路由画像。")
        return

    profiles = profile if isinstance(profile, list) else [profile]
    for doc_profile in profiles:
        with st.expander(f"文档路由画像：{doc_profile.get('file_name', 'unknown')}", expanded=True):
            render_simple_table(
                ["文档级路由", "页面数量", "block 类型数"],
                [[
                    doc_profile.get("document_route", "-"),
                    len(doc_profile.get("page_routes", [])),
                    len(doc_profile.get("block_counts", {})),
                ]],
            )

            st.write("block 统计")
            block_counts = doc_profile.get("block_counts", {})
            render_simple_table(["类型", "数量"], [[key, value] for key, value in block_counts.items()])

            st.write("页面级路由")
            page_routes = doc_profile.get("page_routes", [])
            if page_routes:
                render_simple_table(
                    ["page", "block_types", "route"],
                    [
                        [row.get("page"), row.get("block_types"), row.get("route")]
                        for row in page_routes
                    ],
                )
            else:
                st.caption("没有页面级路由信息。")


def render_parse_job_status() -> None:
    job = st.session_state.last_job
    if not job or not job.get("job_id"):
        return

    job = poll_job(job["job_id"])
    st.session_state.last_job = job
    status = job.get("status", "queued")
    progress = int(job.get("progress", 0))

    st.progress(min(max(progress, 0), 100))
    st.caption(f"{status} | {progress}% | {job.get('message', '')}")

    if status in {"queued", "running", "cancelling"}:
        st.info("解析在后端执行中。页面停止或切换不会终止任务；需要中断时请点击“取消当前解析任务”。")
        col_refresh, col_cancel = st.columns([1, 1])
        with col_refresh:
            if st.button("刷新解析状态", use_container_width=True):
                st.rerun()
        with col_cancel:
            if st.button("取消当前解析任务", type="secondary", use_container_width=True):
                st.session_state.last_job = cancel_parse_job(job["job_id"])
                st.rerun()
        return

    if status == "completed":
        st.success("解析入库完成")
        st.session_state.last_profile = job.get("profile")
        st.session_state.last_result = job.get("result", job)
    elif status == "cancelled":
        st.warning("解析任务已取消，可以立即提交其他文档。")
    elif status == "failed":
        st.error(job.get("error", "解析失败"))
        st.session_state.last_result = job


def stream_chat(
    question: str,
    collection_name: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    payload: dict[str, Any] = {"question": question}
    if collection_name.strip():
        payload["collection_name"] = collection_name.strip()
    response = requests.post(
        api_url("/chat/stream"),
        headers={**api_headers(), "Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=600,
    )
    if not response.ok:
        raise RuntimeError(response.text)

    answer = ""
    sources: list[dict[str, Any]] = []
    final_payload: dict[str, Any] | None = None
    buffer = ""

    answer_box = st.empty()
    status_box = st.empty()

    for raw_chunk in response.iter_content(chunk_size=None):
        if not raw_chunk:
            continue
        buffer += raw_chunk.decode("utf-8", errors="ignore")
        events, buffer = parse_sse_line_buffer(buffer)
        for item in events:
            event_name = item["event"]
            data = item["data"]
            if event_name == "status":
                status_box.info(data.get("message", "处理中..."))
            elif event_name == "token":
                answer += data.get("text", "")
                answer_box.markdown(answer or "正在生成...")
            elif event_name == "done":
                final_payload = data
                answer = data.get("answer", answer)
                sources = data.get("sources", [])
                answer_box.markdown(answer)
                status_box.success("回答完成")
            elif event_name == "error":
                raise RuntimeError(data.get("message", "流式回答失败"))

    return answer, sources, final_payload


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("没有引用来源。")
        return
    for idx, source in enumerate(sources, start=1):
        label = (
            f"{idx}. {source.get('file_name', '-')}"
            f" | {source.get('chunk_id', '-')}"
            f" | page={source.get('page')}"
        )
        with st.expander(label, expanded=idx == 1):
            render_simple_table(
                ["向量/混合分", "rerank 分", "chunk 类型", "标题"],
                [[
                    f"{source.get('score', 0):.4f}",
                    f"{source.get('rerank_score', 0):.4f}",
                    source.get("chunk_type", "-"),
                    source.get("title") or "-",
                ]],
            )
            st.write("片段内容")
            st.markdown(f"<pre>{escape(str(source.get('text', '')))}</pre>", unsafe_allow_html=True)


def main() -> None:
    init_state()

    st.title("政企多模态文档解析智能 RAG 检索问答系统")
    st.caption("Streamlit 内部操作台：上传文档、异步解析、路由画像、流式问答、评估结果。")

    with st.sidebar:
        st.header("服务配置")
        st.text_input("FastAPI 地址", value=DEFAULT_API_BASE, key="api_base")
        st.text_input(
            "服务 API Key（仅填写 APP_API_KEY）",
            type="password",
            key="api_key",
            help="用于访问本项目 FastAPI。禁止填写 DeepSeek/OpenAI 等 LLM_API_KEY。",
        )
        if st.session_state.api_key.strip().lower().startswith("sk-"):
            st.error("检测到疑似 LLM_API_KEY，请立即清空。这里只能填写 APP_API_KEY。")
        st.selectbox(
            "解析模式",
            options=["auto", "text", "ocr", "full"],
            index=0,
            key="analysis_mode",
            help="auto 自动路由；text 偏原生文本；ocr 偏扫描件；full 会启用更完整的多模态解析。",
        )
        st.text_input("collection 名称（留空使用后端默认业务库）", key="collection_name")
        st.divider()
        st.caption("启动方式")
        st.code(
            "python -m streamlit run streamlit_app.py",
            language="powershell",
        )

    tab_upload, tab_chat, tab_eval = st.tabs(["文档解析入库", "知识库问答", "评估与报表"])

    with tab_upload:
        st.subheader("1. 上传并异步解析")
        reset_upload = st.checkbox(
            "上传前清空当前 collection",
            value=False,
            help="默认关闭。关闭时为增量入库；同名文档会按 doc_id 覆盖旧版本。",
        )
        uploaded_file = st.file_uploader(
            "选择 PDF / Word / TXT / Markdown / 图片等文件",
            type=["pdf", "docx", "doc", "txt", "md", "png", "jpg", "jpeg", "bmp", "webp"],
        )
        col_upload, col_sample = st.columns([1, 1])

        with col_upload:
            run_parse = st.button("上传并开始解析", type="primary", use_container_width=True)
        with col_sample:
            if st.button("解析 sample_docs 示例", use_container_width=True):
                with st.spinner("正在解析示例文档..."):
                    params = {"reset": str(reset_upload).lower()}
                    if st.session_state.collection_name.strip():
                        params["collection_name"] = st.session_state.collection_name.strip()
                    result = request_json("POST", "/documents/ingest-sample", params=params)
                    st.session_state.last_job = {"status": "completed", "result": result}
                    st.session_state.last_result = result
                    st.session_state.last_profile = None
                    st.success("示例文档已入库")

        if run_parse:
            if uploaded_file is None:
                st.warning("请先选择一个文件。")
            else:
                with st.spinner("正在上传文件并创建后台解析任务..."):
                    upload_result = upload_file(uploaded_file)
                    file_path = upload_result["file_path"]
                    st.session_state.uploaded_file_path = file_path
                    st.session_state.last_job = create_parse_job(
                        file_path=file_path,
                        analysis_mode=st.session_state.analysis_mode,
                        collection_name=st.session_state.collection_name,
                        reset=reset_upload,
                    )
                    st.session_state.last_result = None
                    st.session_state.last_profile = None
                st.success("任务已创建，解析将在后端继续执行。")

        render_parse_job_status()

        if st.session_state.last_result:
            with st.expander("最近一次入库结果", expanded=False):
                render_json_text(st.session_state.last_result)

        if st.session_state.last_profile:
            st.subheader("最近一次文档路由画像")
            show_profile(st.session_state.last_profile)

        st.divider()
        st.subheader("2. 批量目录初始化 / 增量更新")
        directory_path = st.text_input(
            "目录路径",
            value=str(PROJECT_DIR / "sample_docs"),
            help="适合首次批量铺底库，也适合后续按目录增量更新。",
        )
        recursive = st.checkbox("递归扫描子目录", value=True)
        reset_directory = st.checkbox(
            "目录入库前清空当前 collection",
            value=False,
            help="首次重建知识库可开启；日常更新建议关闭。",
        )
        if st.button("批量导入目录", use_container_width=True):
            if not directory_path.strip():
                st.warning("请输入要导入的目录路径。")
            else:
                with st.spinner("正在批量解析目录并写入知识库..."):
                    result = ingest_directory(
                        directory_path=directory_path.strip(),
                        analysis_mode=st.session_state.analysis_mode,
                        collection_name=st.session_state.collection_name,
                        recursive=recursive,
                        reset=reset_directory,
                    )
                st.session_state.last_job = {"status": "completed", "result": result}
                st.session_state.last_result = result
                st.session_state.last_profile = result.get("profiles")
                st.success(
                    f"目录导入完成：{result.get('file_count', 0)} 个文件，"
                    f"当前知识库共 {result.get('total_chunks', 0)} 个 chunk。"
                )

    with tab_chat:
        active_collection = st.session_state.collection_name.strip() or "后端默认业务库"
        st.subheader("3. 全库流式检索问答")
        st.caption(f"当前提问范围：{active_collection}")
        question = st.text_area(
            "输入问题",
            value="这份报告中的 SLA 指标是多少？",
            height=120,
        )
        if st.button("流式提问", type="primary"):
            if not question.strip():
                st.warning("请输入问题。")
            else:
                with st.container():
                    answer, sources, payload = stream_chat(
                        question.strip(),
                        st.session_state.collection_name,
                    )
                st.session_state.chat_answer = answer
                st.session_state.chat_sources = sources
                if payload:
                    with st.expander("完整响应 JSON", expanded=False):
                        render_json_text(payload)

        if st.session_state.chat_answer:
            st.subheader("引用来源")
            render_sources(st.session_state.chat_sources)

    with tab_eval:
        st.subheader("3. 评估与指标")
        st.caption("默认使用隔离评估库，避免污染当前业务知识库。")
        isolated = st.checkbox("使用隔离评估 collection", value=True)
        if st.button("运行评估", type="primary"):
            with st.spinner("正在运行基础检索评估与 RAGAS 离线评估..."):
                result = request_json("POST", "/eval/run", params={"isolated": str(isolated).lower()})
                st.session_state.eval_result = result
                st.success("评估完成")

        if st.session_state.eval_result:
            result = st.session_state.eval_result
            render_json_text(result)
            metrics = result.get("metrics", {})
            if metrics:
                render_simple_table(["指标", "结果"], [[key, f"{value:.4f}"] for key, value in metrics.items()])

            for key in ["basic_eval_png", "ragas_png"]:
                path = result.get(key)
                if path and Path(path).exists():
                    st.image(path, caption=key, use_container_width=True)


if __name__ == "__main__":
    main()
