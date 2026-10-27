"""Database engine and session management."""

from __future__ import annotations

# 只用 SQLAlchemy 的异步 API：同步 engine 会在 FastAPI 协程里阻塞整个事件循环
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import settings            # DATABASE_URL 在 settings 实例化时就已经固化
from app.core.log import get_logger        # 只有 seed 真的写了行才打日志
from app.models.database import Base       # 全部 32 张表注册在这一个 Base.metadata 上

# Async engine — use asyncpg driver
# 强制会话时区 UTC：容器里的 PG 默认 Asia/Shanghai，而本库所有 datetime 列都是
# naive TIMESTAMP。不锁时区的话 func.now() 写进去的是 +8 的墙上时间，与 Python
# 侧写的 naive UTC 混在一起，超时判定、日界统计全会错 8 小时。
engine = create_async_engine(
    settings.database_url,      # 必须是 asyncpg 这类异步驱动 URL；填同步驱动要到首次 init_db 建连接才报错，不在 import 期
    echo=False,                 # 不打印 SQL：每轮对话都写消息正文，开 echo 会把用户内容灌进日志
    pool_size=5,                # 每进程常驻 5 条连接
    max_overflow=10,            # 突发再借 10 条、用完即还；单进程上限 15，多 worker 部署要按 worker 数乘（PG max_connections 常见 100）
    connect_args={"server_settings": {"TimeZone": "UTC"}},   # 每条新连接都锁 UTC，覆盖容器 PG 的会话默认时区
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    # commit 后不让属性过期：后台接口常在 `async with` 块外继续读行字段（admin.decide_approval 就是），
    expire_on_commit=False,     # 否则读 row.conversation_id 会变成"重新 SELECT"或直接 DetachedInstanceError
)


async def init_db() -> None:
    """Create all tables if they don't exist (idempotent), then seed agents and the bootstrap admin."""
    # begin() 把整段 DDL 放进一个事务：中途失败整体回滚，不会留下"半套表"
    async with engine.begin() as conn:
        # create_all 是同步 API，只能在异步连接上 run_sync；只建缺失的表，不给已存在的表补列（无 Alembic，改模型要手工 ALTER）
        await conn.run_sync(Base.metadata.create_all)
        # 它也不建视图：schema.sql 里的 knowledge_document_stats / conversation_satisfaction 只有手工跑过 schema.sql 才存在
    await seed_agents()
    # 后台所有接口都要求已登录，没有可登录账号面板就完全打不开，所以引导账号
    # 与 agents 一样属于启动必备数据。
    await seed_admin()


# conversations.agent_id 有外键指向 agents，且后台面板按 agent 分组统计，
# 所以这四行必须存在，否则每笔会话写入都会 FK 违规。
DEFAULT_AGENTS = (
    # 元组顺序 = 插入顺序 = 自增 id 顺序，后台 list_agents 按 id 排序，所以这就是面板上的岗位顺序
    ("pre_sales", "售前客服", "pre_sales"),
    ("during_sales", "售中客服", "during_sales"),
    ("after_sales", "售后客服", "after_sales"),
    ("complaint", "投诉受理", "complaint"),
)


async def seed_agents() -> int:
    """按名称幂等补齐四个业务 Agent，model 取当前配置的岗位模型。"""
    # 函数内 import：模块顶层只依赖 Base，避免 models 层与 agents 层互相引用
    from sqlalchemy import select

    from app.models.database import Agent

    async with async_session_factory() as session:
        existing = {
            # 只取 name 单列拼集合，行少但比 select(Agent) 省；判重键与 agents.name 的唯一约束一致
            row for (row,) in (await session.execute(select(Agent.name))).all()
        }
        created = 0
        for name, display_name, role in DEFAULT_AGENTS:
            if name in existing:
                # 已存在就整行不动：display_name / model 后来改了都不会被同步回去，这是有意的（DB 不是配置的唯一来源）
                continue
            session.add(Agent(name=name, display_name=display_name, model=settings.model_for(role)))
            # model 列存的是 seed 那一刻解析出的模型快照；改 LLM_MODEL_* 后需删行重建才刷新，后台展示并不读这一列
            created += 1
        await session.commit()   # 四行一次提交：中途失败回滚，不会留下"只建了两个 Agent"的半成品
        if created:
            get_logger(__name__).info("seeded %d agents", created)
        return created           # 返回新建行数（0 表示库已就绪），init_db 不看这个值，只便于自检断言


async def get_session() -> AsyncSession:
    """Yield an async session for dependency injection."""
    async with async_session_factory() as session:
        # 生成器不主动 commit：请求结束退出即回滚，写操作得由调用方自己 commit（业务代码目前基本自己开 session，因为要跨 await 复用）
        yield session


async def seed_admin() -> int:
    """引导一个可登录的后台账号（幂等：已存在就不动口令）。

    口令来自 ADMIN_BOOTSTRAP_PASSWORD；只在首次创建时写入哈希，
    之后改密码要走数据库/管理接口，否则重启会把人的改动冲掉。
    """
    from sqlalchemy import select

    from app.config import settings      # 顶层已导入同一个 settings，这里再取一次只为把函数依赖写全
    from app.core.auth import hash_password
    from app.models.database import AdminUser

    if not settings.bootstrap_admin_password:
        # 口令没有代码默认值：宁可不建号也不造一个弱口令后门，此时后台完全无法登录
        get_logger(__name__).warning(
            "ADMIN_BOOTSTRAP_PASSWORD 未配置，跳过后台管理员自动创建"
        )
        return 0

    async with async_session_factory() as session:
        existing = (await session.execute(
            select(AdminUser).where(AdminUser.username == settings.bootstrap_admin_user)
        )).scalar_one_or_none()      # 按 username 查（该列 unique），scalar_one_or_none 保证 0/1 行
        if existing is not None:
            return 0                  # 连 password_hash 都不碰：人工改过口令后重启不会被冲掉
        session.add(AdminUser(
            username=settings.bootstrap_admin_user,
            password_hash=hash_password(settings.bootstrap_admin_password),   # 只存 pbkdf2 摘要，明文不落库
            role="admin",             # 固定 admin：属于 DECIDER_ROLES，能审批高危操作（要只读账号请手工建 operator）
            status="active",          # 必须是 active，否则 require_staff 每次请求都 401
        ))
        await session.commit()
        get_logger(__name__).info("seeded admin user: %s", settings.bootstrap_admin_user)   # 只打用户名，不打口令
        return 1
