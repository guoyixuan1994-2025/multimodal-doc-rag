from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_old_record_factory = logging.getLogRecordFactory()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    # 用 LogRecordFactory 给所有日志记录补 request_id。
    # 这样第三方库 httpx/openai/langchain 打出来的日志也不会缺字段。
    def record_factory(*args, **kwargs):
        record = _old_record_factory(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    logging.setLogRecordFactory(record_factory)

    # 简化版结构化日志：每条日志都带 request_id，方便后续排查某次问答的完整链路。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [request_id=%(request_id)s] %(name)s - %(message)s",
    )


def new_request_id() -> str:
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    request_id_var.set(request_id)
    return request_id
