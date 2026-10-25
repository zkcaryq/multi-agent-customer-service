"""Database engine and session management."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import settings
from app.core.log import get_logger
from app.models.database import Base

# Async engine — use asyncpg driver
# 强制会话时区 UTC：容器里的 PG 默认 Asia/Shanghai，而本库所有 datetime 列都是
# naive TIMESTAMP。不锁时区的话 func.now() 写进去的是 +8 的墙上时间，与 Python
# 侧写的 naive UTC 混在一起，超时判定、日界统计全会错 8 小时。
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    connect_args={"server_settings": {"TimeZone": "UTC"}},
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables if they don't exist (idempotent), then seed agents and the bootstrap admin."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_agents()
    # 后台所有接口都要求已登录，没有可登录账号面板就完全打不开，所以引导账号
    # 与 agents 一样属于启动必备数据。
    await seed_admin()


# conversations.agent_id 有外键指向 agents，且后台面板按 agent 分组统计，
# 所以这四行必须存在，否则每笔会话写入都会 FK 违规。
DEFAULT_AGENTS = (
    ("pre_sales", "售前客服", "pre_sales"),
    ("during_sales", "售中客服", "during_sales"),
    ("after_sales", "售后客服", "after_sales"),
    ("complaint", "投诉受理", "complaint"),
)


async def seed_agents() -> int:
    """按名称幂等补齐四个业务 Agent，model 取当前配置的岗位模型。"""
    from sqlalchemy import select

    from app.models.database import Agent

    async with async_session_factory() as session:
        existing = {
            row for (row,) in (await session.execute(select(Agent.name))).all()
        }
        created = 0
        for name, display_name, role in DEFAULT_AGENTS:
            if name in existing:
                continue
            session.add(Agent(name=name, display_name=display_name, model=settings.model_for(role)))
            created += 1
        await session.commit()
        if created:
            get_logger(__name__).info("seeded %d agents", created)
        return created


async def get_session() -> AsyncSession:
    """Yield an async session for dependency injection."""
    async with async_session_factory() as session:
        yield session


async def seed_admin() -> int:
    """引导一个可登录的后台账号（幂等：已存在就不动口令）。

    口令来自 ADMIN_BOOTSTRAP_PASSWORD；只在首次创建时写入哈希，
    之后改密码要走数据库/管理接口，否则重启会把人的改动冲掉。
    """
    from sqlalchemy import select

    from app.config import settings
    from app.core.auth import hash_password
    from app.models.database import AdminUser

    if not settings.bootstrap_admin_password:
        get_logger(__name__).warning(
            "ADMIN_BOOTSTRAP_PASSWORD 未配置，跳过后台管理员自动创建"
        )
        return 0

    async with async_session_factory() as session:
        existing = (await session.execute(
            select(AdminUser).where(AdminUser.username == settings.bootstrap_admin_user)
        )).scalar_one_or_none()
        if existing is not None:
            return 0
        session.add(AdminUser(
            username=settings.bootstrap_admin_user,
            password_hash=hash_password(settings.bootstrap_admin_password),
            role="admin",
            status="active",
        ))
        await session.commit()
        get_logger(__name__).info("seeded admin user: %s", settings.bootstrap_admin_user)
        return 1
