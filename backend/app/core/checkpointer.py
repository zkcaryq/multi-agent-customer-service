"""LangGraph 检查点存储。

interrupt / resume 依赖 checkpointer：把执行状态落到 PostgreSQL，
进程重启后待审批的会话依然能接着恢复——这是"可暂停可恢复"真正的落点。
检查点初始化失败会退回内存；只保留同进程图恢复能力。业务会话/审批仍依赖 PG，整条接口未必可用。
"""

from __future__ import annotations

import asyncio                          # 只用它的 Lock，做首建串行化
from typing import Any                  # 返回类型写 Any 而非具体 Saver：Postgres 版与内存版是两个类，没有共同基类可依赖

from app.config import settings          # database_url 走配置，不在这里拼串
from app.core.log import get_logger

# 进程级缓存，四个领域图共用 saver，但用不同 thread_id 隔离状态。
_saver: Any = None
# 避免同进程首次并发请求重复创建连接池；不是跨进程锁或审批互斥锁。
_lock = asyncio.Lock()


def _psycopg_dsn() -> str:
    """SQLAlchemy 的 asyncpg 方言前缀对 psycopg 无意义，必须剥掉。"""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def get_checkpointer() -> Any:
    global _saver                         # 进程级单例：模块变量而不是实例属性，四个域 Agent 拿到的是同一个池
    # 锁外快路径与锁内二次检查配合；退回内存后也会缓存，不会每轮自动重连 PG。
    if _saver is not None:
        return _saver

    # 只有初始化阶段串行；不负责把同会话的多个 run/resume 请求串行化。
    async with _lock:
        # 锁外快路径与锁内二次检查配合；退回内存后也会缓存，不会每轮自动重连 PG。
        if _saver is not None:
            return _saver

        # 三个 import 放在函数里而不是文件头：psycopg / langgraph-checkpoint-postgres 只在真要建
        # Postgres 检查点时才需要，测试与降级路径可以在没装好这些包的环境里继续导入本模块。
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        pool = AsyncConnectionPool(
            conninfo=_psycopg_dsn(),
            # 构造池时不立即连接，下面 await pool.open 在当前异步事件循环里显式打开。
            open=False,
            # 至少保留一个连接；这是检查点独立连接池，不是 SQLAlchemy 业务连接池。
            min_size=1,
            # 限制检查点连接数；更多并发会等待池，不意味着最多五个会话。
            max_size=5,
            name="ma-cs-checkpoints",
            # 与 SQLAlchemy 侧保持一致：锁 UTC。注意 psycopg 不认 asyncpg 的
            # server_settings，设置会话参数要用 libpq 的 options。
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": "-c TimeZone=UTC"},
        )
        try:
            # 等待初始连接就绪，失败后才选择内存降级。
            await pool.open(wait=True, timeout=10.0)
            # Saver 只是池的包装：它自己不建连接，所以必须在 pool 打开之后构造。
            saver = AsyncPostgresSaver(pool)
            await saver.setup()  # 幂等建 checkpoint 相关表
        except Exception as exc:
            # 兜住所有异常是刻意的：连接超时、DDL 权限不足、表结构冲突都应该降级而不是让整条接口 500。
            get_logger(__name__).warning("PostgreSQL 不可用，退回内存检查点（重启即丢失）: %s", exc)
            # 初始化异常时释放已创建的连接池，避免退回内存后还遗留连接。
            await pool.close()
            from langgraph.checkpoint.memory import InMemorySaver

            # 进程重启和多 worker 切换都会丢失/找不到这份暂停状态。
            _saver = InMemorySaver()
            return _saver

        get_logger(__name__).info("AsyncPostgresSaver ready (tables ensured)")
        # 只缓存初始化成功的 saver；应用当前没有单独的 saver 生命周期关闭入口。
        _saver = saver
        return _saver


def thread_id(agent_name: str, conversation_id: int) -> str:
    """一个会话在一个 Agent 上一条线程。带上 Agent 名，避免不同图共用 thread 串状态。"""
    return f"{agent_name}:{conversation_id}"
