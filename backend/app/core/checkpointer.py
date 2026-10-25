"""LangGraph 检查点存储。

interrupt / resume 依赖 checkpointer：把执行状态落到 PostgreSQL，
进程重启后待审批的会话依然能接着恢复——这是"可暂停可恢复"真正的落点。
PG 不可用时退回内存实现，只影响持久性，不影响开发联调。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings
from app.core.log import get_logger

_saver: Any = None
_lock = asyncio.Lock()


def _psycopg_dsn() -> str:
    """SQLAlchemy 的 asyncpg 方言前缀对 psycopg 无意义，必须剥掉。"""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def get_checkpointer() -> Any:
    global _saver
    if _saver is not None:
        return _saver

    async with _lock:
        if _saver is not None:
            return _saver

        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        pool = AsyncConnectionPool(
            conninfo=_psycopg_dsn(),
            open=False,
            min_size=1,
            max_size=5,
            name="cogniflow-checkpoints",
            # 与 SQLAlchemy 侧保持一致：锁 UTC。注意 psycopg 不认 asyncpg 的
            # server_settings，设置会话参数要用 libpq 的 options。
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": "-c TimeZone=UTC"},
        )
        try:
            await pool.open(wait=True, timeout=10.0)
            saver = AsyncPostgresSaver(pool)
            await saver.setup()  # 幂等建 checkpoint 相关表
        except Exception as exc:
            get_logger(__name__).warning("PostgreSQL 不可用，退回内存检查点（重启即丢失）: %s", exc)
            await pool.close()
            from langgraph.checkpoint.memory import InMemorySaver

            _saver = InMemorySaver()
            return _saver

        get_logger(__name__).info("AsyncPostgresSaver ready (tables ensured)")
        _saver = saver
        return _saver


def thread_id(agent_name: str, conversation_id: int) -> str:
    """一个会话在一个 Agent 上一条线程。带上 Agent 名，避免不同图共用 thread 串状态。"""
    return f"{agent_name}:{conversation_id}"
