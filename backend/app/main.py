import os
import sys
from contextlib import asynccontextmanager

if sys.platform == "win32":
    # psycopg3 的异步实现不接受 Windows 默认的 ProactorEventLoop，
    # 而 LangGraph 的 Postgres checkpointer 用的是 psycopg3 异步连接池。
    # 这里设一次是给"直接 import app"的用法兜底；用 uvicorn 命令行启动时循环
    # 已经先建好了，得走 backend/run.py。
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.websocket import router as ws_router
from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.core.log import get_logger, set_trace, setup_logging, trace_id

setup_logging()
log = get_logger("cogniflow.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, load BM25 index, ensure Milvus collection exists."""
    from app.providers import describe, preflight

    log.info("providers: %s", describe())
    for issue in preflight():
        # 不阻塞启动——评测和自检要在无凭据时也能跑，但必须把"哪家模型现在是空的"说在最前面
        log.warning("配置不完整: %s", issue)

    import asyncio

    if "Proactor" in type(asyncio.get_running_loop()).__name__:
        log.error(
            "运行在 ProactorEventLoop 上，psycopg3 用不了，Postgres 检查点会退回内存版——"
            "审批/中断状态重启即丢。请改用 `python run.py` 启动"
        )

    # Init database tables
    try:
        from app.models.engine import init_db
        await init_db()
        log.info("database tables ensured")
    except Exception as e:
        log.warning("DB init failed, persistence disabled: %s", e)

    # Load RAG indexes
    try:
        from app.rag.bm25_store import bm25_store
        from app.rag.ingestion import BM25_INDEX_PATH
        from app.rag.vector_store import vector_store

        if os.path.exists(BM25_INDEX_PATH) and not bm25_store.is_ready:
            bm25_store.load(BM25_INDEX_PATH)
            log.info("BM25 index loaded: %d docs", len(bm25_store))

        if vector_store.is_ready:
            vector_store.load()
            log.info("Milvus collection loaded")
        else:
            vector_store.create_collection()
            log.info("Milvus collection created")
    except Exception as e:
        log.warning("RAG index load failed: %s", e)

    # 工具注册属于运行时能力，不应取决于用户先访问了哪个 Agent。
    from app.agents import load_all_agents
    from app.tools.registry import tool_registry

    load_all_agents()
    log.info("runtime tools registered: %d", len(tool_registry.list_all()))

    yield


app = FastAPI(
    title="CogniFlow — 智能客服",
    description="电商智能客服系统 — 售前/售中/售后/投诉四领域 Agent 服务",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(ws_router, prefix="/ws")


@app.get("/health")
async def health():
    from app.rag.vector_store import vector_store
    from app.rag.bm25_store import bm25_store

    db_ok = False
    try:
        from app.models.engine import engine as db_engine
        async with db_engine.connect() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        db_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "agent": "multi",
        "milvus": vector_store.is_ready,
        "bm25": bm25_store.is_ready,
        "bm25_count": len(bm25_store),
        "database": db_ok,
    }
