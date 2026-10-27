"""FastAPI 应用组装：启动钩子（建表/索引/工具）、中间件、路由挂载、健康检查。"""

import os                                      # 只用到 os.path.exists：启动时判断 BM25 索引文件在不在
import sys                                     # 只用到 sys.platform：判断是否需要换事件循环策略
from contextlib import asynccontextmanager      # 把 startup / yield / shutdown 写成 async with 形态的 lifespan

if sys.platform == "win32":
    # psycopg3 的异步实现不接受 Windows 默认的 ProactorEventLoop，
    # 而 LangGraph 的 Postgres checkpointer 用的是 psycopg3 异步连接池。
    # 这里设一次是给"直接 import app"的用法兜底；用 uvicorn 命令行启动时循环
    # 已经先建好了，得走 backend/run.py。
    import asyncio                              # 只在本分支导入：非 Windows 的 asyncio 没有 WindowsSelectorEventLoopPolicy，放顶层会 AttributeError

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())   # 必须早于下面所有 app.* 导入：策略影响后续任何 asyncio.run()/get_event_loop() 造出的循环类型

from fastapi import FastAPI, Request            # Request 只给 trace 中间件读请求头用
from fastapi.middleware.cors import CORSMiddleware   # 跨域策略集中在这里，四个 router 不各写一份

from app.api.chat import router as chat_router   # 四个 router 在导入期即完成定义；这里只是取对象，挂载在文件末尾
from app.api.websocket import router as ws_router
from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.core.log import get_logger, set_trace, setup_logging, trace_id   # set_trace 写 ContextVar，trace_id 读回（回写响应头要用最终值）

setup_logging()                                 # 要早于任何日志输出：它按 _app_handler 标记清理旧 handler，reload/重复导入不会把每行打两遍
log = get_logger("app.startup")                 # 名字只是 logger 字段，便于从 uvicorn 噪声里筛启动阶段这几行


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, load BM25 index, ensure Milvus collection exists."""
    from app.providers import describe, preflight   # 延迟到启动时导：settings 在 import 那一刻固化环境，这里只是读它、不建连接

    log.info("providers: %s", describe())           # describe() 只拼配置文本（含向量库形态），不发网络请求；放第一行，排障先看"现在用的是哪家模型"
    for issue in preflight():                       # 返回的是"选了这个 provider 但对应变量为空"的清单；配置齐全时循环体一次都不走
        # 不阻塞启动——评测和自检要在无凭据时也能跑，但必须把"哪家模型现在是空的"说在最前面
        log.warning("配置不完整: %s", issue)         # 只 warning：缺凭据的后果推迟到首次调用抛 ProviderError，由接口层翻成 503，不在启动期炸

    import asyncio                                  # 函数内再导一次：模块顶部那句只在 win32 分支执行，这里要无条件可用

    if "Proactor" in type(asyncio.get_running_loop()).__name__:   # 按类名字符串判，避免在非 Windows 顶层导入只存在于 Windows 的策略类；取"正在跑的循环"而不是策略，能抓到 uvicorn 已经建好的那个实例
        log.error(
            "运行在 ProactorEventLoop 上，psycopg3 用不了，Postgres 检查点会退回内存版——"
            "审批/中断状态重启即丢。请改用 `python run.py` 启动"
        )                                           # 这是提前预告而非检测：检查点要到首次 Agent 运行才建池，届时失败只会留一条 warning 日志

    # Init database tables
    try:
        from app.models.engine import init_db       # 引擎与建表函数晚绑：PG 未就绪时，导入这一段本身不该拦住启动
        await init_db()                             # create_all 幂等建表，并补 seed agents 四行 + 引导后台账号（agents 是 conversations.agent_id 的外键目标）
        log.info("database tables ensured")
    except Exception as e:
        log.warning("DB init failed, persistence disabled: %s", e)   # 降级不撤销：engine 还在，会话/审批写入会在各自接口里失败（会话持久化那条路返回 503）

    # Load RAG indexes
    try:
        from app.rag.bm25_store import bm25_store      # 进程内单例：多 worker 时每个进程各加载一份，彼此不同步
        from app.rag.ingestion import BM25_INDEX_PATH   # 固定读 data/.index/bm25.pkl，产物由入库脚本写；路径在这里只读不拼
        from app.rag.vector_store import vector_store   # 构造时只解析 uri/是否远端，不代表已连通；首次 client 访问才建连接

        if os.path.exists(BM25_INDEX_PATH) and not bm25_store.is_ready:   # 两个条件都不能省：文件缺失时 load 抛 FileNotFoundError；已就绪再 load 等于白重建一遍词频统计
            bm25_store.load(BM25_INDEX_PATH)            # pickle 反序列化 docs/corpus 后重建 BM25Okapi；只吃本项目产物，格式变了要重新入库
            log.info("BM25 index loaded: %d docs", len(bm25_store))   # __len__ 返回文档条数，与 is_ready 判据里的 _docs 同一口径

        if vector_store.is_ready:                       # property 在本地缓存位为假时会真访问 has_collection，可能直接抛异常跳到 except
            vector_store.load()                         # 把集合载入 Milvus 内存：search 前必需；内部再判一次 has_collection，重复调用无害
            log.info("Milvus collection loaded")
        else:
            vector_store.create_collection()            # 首次启动建 schema；没传 drop_existing，所以不会顺手把既有数据删掉
            log.info("Milvus collection created")
    except Exception as e:
        # 已知限制：BM25 与 Milvus 挤在同一个 try，前一步抛错会整段跳过，Milvus 分支不再执行
        log.warning("RAG index load failed: %s", e)     # 只 warning 不阻塞：检索节点自己会捕获异常降级为空结果，缺索引不该拒绝启动

    # 工具注册属于运行时能力，不应取决于用户先访问了哪个 Agent。
    from app.agents import load_all_agents              # 四个 Agent 单例的构造入口；agent_by_name 是函数内 import，不主动调用就没人注册
    from app.tools.registry import tool_registry        # 只为把注册总数打进日志，不参与任何决策

    load_all_agents()                                   # 构造单例时 register_tools() 并同步建图，把首次请求的注册成本挪到启动阶段
    log.info("runtime tools registered: %d", len(tool_registry.list_all()))   # 全局总数，未按 agent_id 拆分：某域注册成空列表也看不出来，而空 schema 会让该域静默退化为纯问答

    yield                                               # 上面是 startup、以下是 shutdown；本文件没有 shutdown 语句，检查点连接池与记忆后端都缺显式关闭入口


app = FastAPI(
    title="基于 Multi-Agent 的智能客服业务平台",       # 只进 OpenAPI/文档页，运行时逻辑不读它
    description="电商智能客服系统 — 售前/售中/售后/投诉四领域 Agent 服务",
    version="0.2.0",                                 # 手写的文档版本号；/health 的返回值里没有版本，两处不同步
    lifespan=lifespan,                               # 传函数对象而不是调用结果：@asynccontextmanager 已把协程函数包成上下文管理器工厂
)


@app.middleware("http")                              # 等价于 add_middleware(BaseHTTPMiddleware, dispatch=...)；注册早于下面的 CORS，因此排在更内层
async def trace_middleware(request: Request, call_next):
    """每个请求一个 trace_id，之后同一请求的日志能串成一条链。

    客户端可通过 X-Trace-Id 透传；没传就生成 8 位随机串，并回写到响应头。
    chat 接口在确定会话号后会把它切成 conv-<id>，便于按会话检索。
    """
    set_trace(request.headers.get("X-Trace-Id"))     # 头缺失即传 None，由 set_trace 自己生成；ContextVar 按请求 task 隔离，并发请求不串号
    response = await call_next(request)              # 抛出的业务异常不在这里捕获，交给 ServerErrorMiddleware 转 500（那种响应不带 X-Trace-Id）
    response.headers["X-Trace-Id"] = trace_id()      # 回读而不是复用进入时的值：chat 内部把 trace 改成 conv-<会话号>，响应头要给出最终值
    return response


app.add_middleware(                                  # add_middleware 是 insert(0)：后注册的在外层，所以 CORS 包住 trace 中间件
    CORSMiddleware,
    allow_origins=["*"],                             # 通配配上下面的凭据开关，带 Cookie 的请求会被 Starlette 改成回显具体 Origin 并加 Vary: Origin
    allow_credentials=True,                          # 于是任何站点都能带凭据打后台接口；投产前应收敛成来源白名单（这里按局域网演示放宽）
    allow_methods=["*"],                             # 展开成固定方法全集，不是"任意字符串都放行"
    allow_headers=["*"],                             # 预检按请求头回显，等于不锁自定义头（X-Trace-Id 才能直接透传）
)

app.include_router(auth_router, prefix="/api/v1")    # 四条都挂 /api/v1，路径不重叠；这里的先后只在同路径同方法时才有意义
app.include_router(chat_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(ws_router, prefix="/ws")          # WS 升级请求不在 CORSMiddleware 的 http 分支里，跨域限制得服务端自己校验 Origin


@app.get("/health")                                  # 没声明 response_model：schemas 里的 HealthResponse 无人引用，字段也对不上（它带 version，这里不返回）
async def health():
    from app.rag.vector_store import vector_store     # 每次探测现读单例状态，不缓存，便于看到运行期入库后就绪位翻转
    from app.rag.bm25_store import bm25_store

    db_ok = False                                    # 先按"不通"初始化：探测失败就走这个值，不做"未测即通过"的乐观默认
    try:
        from app.models.engine import engine as db_engine
        async with db_engine.connect() as conn:       # 借一条连接就归还：不碰业务表，避免健康检查给数据库加负载
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")   # 行内动态导入，等同顶部 from sqlalchemy import text；SQLAlchemy 2 不接受裸 SQL 字符串
            )
        db_ok = True
    except Exception:
        pass                                          # 吞掉全部异常换 /health 不 500，代价是响应里看不出失败原因（要去翻日志）

    return {
        "status": "ok",                               # 恒定返回 ok，不随下面三个探针变化：当存活探针可用，当就绪探针会误判健康
        "agent": "multi",
        "milvus": vector_store.is_ready,              # 不在 try 里：就绪位为假时它会建 client 并访问远端，Milvus 不可达时 /health 直接 500
        "bm25": bm25_store.is_ready,                  # 判据是"索引对象已建且文档数 > 0"，空文件加载出来的索引不算就绪
        "bm25_count": len(bm25_store),                # 与启动那行日志同口径，运维可对比是否被重建或被清空
        "database": db_ok,
    }
