"""CRM 侧只读工具：客户画像、会员权益、历史消费与服务记录。

简历里"联动订单、CRM、工单多套内部系统"的 CRM 部分落在这里。
只暴露只读能力，且 user_id 由运行时上下文注入，模型无法指定查谁。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import func, select

from app.models.database import (
    Conversation,
    HumanHandoff,
    Order,
    Refund,
    User,
    UserAttribute,
    UserTag,
)
from app.models.engine import async_session_factory
from app.tools.base import ToolDefinition, ToolResult

# CRM 的本地适配实现：演示环境直接查询项目 PostgreSQL。生产环境可保持工具契约不变，
# 将下面的 handler 替换为企业 CRM 的 HTTP/MCP 客户端。
from app.tools.registry import tool_registry


def _num(v: Any) -> float:
    return float(v) if isinstance(v, Decimal) else float(v or 0)


async def _load_user(session, user_id: str) -> tuple[User | None, list, list]:
    user = (await session.execute(
        select(User).where(User.external_id == user_id)
    )).scalar_one_or_none()
    if user is None:
        return None, [], []
    attrs = (await session.execute(
        select(UserAttribute).where(UserAttribute.user_id == user.id)
    )).scalars().all()
    tags = (await session.execute(
        select(UserTag).where(UserTag.user_id == user.id)
    )).scalars().all()
    return user, list(attrs), list(tags)


async def _query_customer_profile(user_id: str = "") -> ToolResult:
    """会员等级、积分、标签 + 累计消费、订单数、未完成退款。"""
    async with async_session_factory() as session:
        user, attrs, tags = await _load_user(session, user_id)
        if user is None:
            return ToolResult(success=False, error="未识别当前用户，请重新登录")

        order_count, spent = (await session.execute(
            select(func.count(Order.id), func.coalesce(func.sum(Order.amount), 0)).where(
                Order.user_id == user.id, Order.status != "unpaid"
            )
        )).one()
        open_refunds = (await session.execute(
            select(func.count(Refund.id)).where(
                Refund.user_id == user.id,
                Refund.status.notin_(["refunded", "rejected", "cancelled"]),
            )
        )).scalar() or 0

        return ToolResult(success=True, data={
            "customer_id": user.external_id,
            "nickname": user.nickname,
            "attributes": {a.attr_key: a.attr_value for a in attrs},
            "tags": sorted({t.tag for t in tags}),
            "paid_order_count": int(order_count),
            "lifetime_spend": _num(spent),
            "open_refund_count": int(open_refunds),
        })


async def _query_service_history(user_id: str = "") -> ToolResult:
    """近 30 天会话量、转人工次数与最近几次转人工原因。

    human_handoffs 没有 user_id，要经 conversations 关联回用户。
    """
    from datetime import datetime, timedelta, timezone

    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)

    async with async_session_factory() as session:
        user, _attrs, _tags = await _load_user(session, user_id)
        if user is None:
            return ToolResult(success=False, error="未识别当前用户，请重新登录")

        convs = (await session.execute(
            select(func.count(Conversation.id)).where(
                Conversation.user_id == user.id, Conversation.created_at >= since
            )
        )).scalar() or 0

        handoff_count = (await session.execute(
            select(func.count(HumanHandoff.id))
            .join(Conversation, Conversation.id == HumanHandoff.conversation_id)
            .where(Conversation.user_id == user.id, HumanHandoff.created_at >= since)
        )).scalar() or 0

        recent = (await session.execute(
            select(HumanHandoff.reason, HumanHandoff.status)
            .join(Conversation, Conversation.id == HumanHandoff.conversation_id)
            .where(Conversation.user_id == user.id)
            .order_by(HumanHandoff.id.desc()).limit(3)
        )).all()

        return ToolResult(success=True, data={
            "conversations_30d": int(convs),
            "handoff_count_30d": int(handoff_count),
            "recent_handoffs": [{"reason": r, "status": s} for r, s in recent],
        })


def register_crm_tools() -> None:
    tool_registry.register(
        ToolDefinition(
            name="query_customer_profile",
            display_name="查询客户画像",
            description="查询当前用户的会员等级、积分、标签、累计消费与未完成退款",
            parameters=[],
            context_params=["user_id"],
        ),
        _query_customer_profile,
        agent_id=None,  # 三个业务 Agent 都可用
    )

    tool_registry.register(
        ToolDefinition(
            name="query_service_history",
            display_name="查询服务历史",
            description="查询当前用户近 30 天的咨询量与转人工记录",
            parameters=[],
            context_params=["user_id"],
        ),
        _query_service_history,
        agent_id=None,
    )
