"""日志：级别与格式由环境驱动，每条带 trace_id，便于把一次请求的日志串起来。

之前全项目靠 print——没有级别、没有出处，线上出问题时无从下手。uvicorn 自带
的 handler 不经过这里，所以访问日志格式不变，应用日志统一走本模块。
"""

from __future__ import annotations

import contextvars
import logging
import sys
from uuid import uuid4

from app.config import settings

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class _TraceFormatter(logging.Formatter):
    """把 ContextVar 里的 trace_id 注入 record，格式串才能引用 %(trace)s。"""

    def format(self, record: logging.LogRecord) -> str:
        record.trace = _trace_id.get()
        return super().format(record)


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    # uvicorn --reload 等场景会重复导入，避免叠加 handler 导致每行打印两次
    for handler in list(root.handlers):
        if getattr(handler, "_app_handler", False):
            root.removeHandler(handler)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_TraceFormatter(fmt=settings.log_format, datefmt="%H:%M:%S"))
        handler._app_handler = True
        root.addHandler(handler)

    # 三方库的 INFO 太吵，只保留它们自己的告警
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "pymilvus", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_trace(value: str | None = None) -> str:
    _trace_id.set(value or uuid4().hex[:8])
    return _trace_id.get()


def trace_id() -> str:
    return _trace_id.get()
