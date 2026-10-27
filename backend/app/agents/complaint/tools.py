"""投诉受理工具：定级、建单（human_handoffs）、查询处理进度。

工单落在 human_handoffs 表——它本来就是"转人工工单"的建模，
带 status/claimed_by/resolution_note，不另起一套工单表。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.core.audit import record_audit
from app.models.database import Agent, Conversation, HumanHandoff, User
from app.models.engine import async_session_factory
from app.tools.base import ToolDefinition, ToolParameterDef, ToolResult
from app.tools.registry import tool_registry

# 定级规则：命中越靠前的关键词级别越高，直接决定 SLA
_SEVERITY_RULES: list[tuple[str, list[str], int]] = [
    ("high", ["315", "工商", "市场监管", "消协", "曝光", "媒体", "起诉", "律师", "欺诈", "造假", "假货", "安全"], 2),
    ("medium", ["投诉", "维权", "赔偿", "态度差", "拖延", "拒不", "多次", "再次联系"], 24),
    ("low", ["建议", "反馈", "不太满意", "希望改进"], 72),
]

_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("质量问题", ["质量", "坏了", "破损", "故障", "假货", "三无", "异味"]),
    ("物流问题", ["快递", "物流", "发货", "签收", "丢件"]),
    ("服务态度", ["态度", "客服不理", "辱骂", "推诿"]),
    ("退款纠纷", ["退款", "不给退", "未到账", "拒退"]),
]

_SEVERITY_LABEL = {"high": "高危", "medium": "一般", "low": "轻微"}


def classify_complaint(text: str) -> tuple[str, str, int]:
    """返回 (severity, category, sla_hours)。规则可解释，比让模型直接打分更好审计。"""
    content = text or ""
    # 未命中规则默认一般投诉、24 小时；只是 SLA 字段建议，没有自动计时调度。
    severity, sla = "medium", 24
    for level, keywords, hours in _SEVERITY_RULES:
        # 顺序优先，命中第一档就 break，不累计多条规则的分数。
        if any(k in content for k in keywords):
            severity, sla = level, hours
            break

    category = "其他投诉"
    for name, keywords in _CATEGORY_RULES:
        # 顺序优先，命中第一档就 break，不累计多条规则的分数。
        if any(k in content for k in keywords):
            category = name
            break
    return severity, category, sla


async def _create_service_ticket(
    detail: str = "", order_id: str = "", user_id: str = "", conversation_id: int = 0,
) -> ToolResult:
    """建转人工工单。同一会话重复调用会各建一张单，由 Agent 侧决定是否已建单。"""
    if not conversation_id:
        return ToolResult(success=False, error="缺少会话上下文，无法建单")

    severity, category, sla_hours = classify_complaint(f"{detail} {order_id}")

    async with async_session_factory() as session:
        # 已有会话直接取用，当前函数未再次验证 user_id 归属，依赖入口保证上下文正确。
        conv = (await session.get(Conversation, conversation_id))
        if conv is None:
            # 自检/脚本直连时会话可能还不存在，先补一条归属当前用户的会话
            user = (await session.execute(
                select(User).where(User.external_id == user_id)
            )).scalar_one_or_none()
            if user is None:
                return ToolResult(success=False, error="未识别当前用户，无法建单")
            # agent_id 有外键约束，必须按名字查真实主键，不能硬编码
            agent_row = (await session.execute(
                select(Agent.id).where(Agent.name == "complaint").limit(1)
            )).scalar_one_or_none()
            conv = Conversation(user_id=user.id, agent_id=agent_row, status="open",
                                title=category, source="complaint_agent")
            session.add(conv)
            # 为脚本补建会话时先取得主键，工单外键再引用；原请求 conversation_id 不会自动同步改变。
            await session.flush()

        # 每次调用插新行，没有动作幂等键；图重放或网络重试可能重复建工单。
        ticket = HumanHandoff(
            conversation_id=conv.id,
            reason=category,
            summary=detail or "用户投诉，需人工跟进",
            intent_agent="complaint",
            status="pending",
        )
        session.add(ticket)
        await session.commit()
        # 读取数据库生成的 ID，才能构造展示用 TS-编号。
        await session.refresh(ticket)
        # 格式化为至少六位数，不是额外工单表的独立业务编号列。
        ticket_no = f"TS-{ticket.id:06d}"

    await record_audit(
        action="complaint_ticket_created",
        entity_type="human_handoff",
        entity_id=ticket.id,
        summary=f"{_SEVERITY_LABEL[severity]}投诉建单 {ticket_no}（{category}）",
        detail={"severity": severity, "category": category, "sla_hours": sla_hours,
                "order_id": order_id, "user_id": user_id},
    )

    return ToolResult(success=True, data={
        "ticket_id": ticket_no,
        "severity": severity,
        "severity_label": _SEVERITY_LABEL[severity],
        "category": category,
        "sla_hours": sla_hours,
        "status": ticket.status,
        # 数据库工单存在与真人已接单是两件事；代码没有保证经理会在 SLA 内联系的调度系统。
        "next_step": "已生成人工工单，客服经理将在 SLA 内联系您",
    })


async def _query_my_complaints(user_id: str = "", conversation_id: int = 0) -> ToolResult:
    """查当前用户未关闭的投诉工单。"""
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(HumanHandoff)
            .join(Conversation, Conversation.id == HumanHandoff.conversation_id)
            .join(User, User.id == Conversation.user_id)
            .where(User.external_id == user_id, HumanHandoff.status != "resolved")
            .order_by(HumanHandoff.id.desc())
        )).scalars().all()

    if not rows:
        return ToolResult(success=True, data={"tickets": [], "message": "当前没有未处理完的投诉工单"})

    return ToolResult(success=True, data={"tickets": [
        {
            "ticket_id": f"TS-{h.id:06d}",
            "category": h.reason,
            "status": h.status,
            "created_at": h.created_at.strftime("%Y-%m-%d %H:%M") if h.created_at else None,
            "resolution_note": h.resolution_note,
        } for h in rows
    ]})


def register_complaint_tools(agent_id: int = 4) -> None:
    from app.tools.crm import register_crm_tools
    register_crm_tools()

    tool_registry.register(
        ToolDefinition(
            name="create_service_ticket",
            display_name="创建投诉工单",
            description="为用户投诉创建人工跟进工单，系统自动定级并给出 SLA",
            parameters=[
                ToolParameterDef("detail", "string", "投诉内容摘要", is_required=True),
                ToolParameterDef("order_id", "string", "相关订单号（可选）"),
            ],
            context_params=["user_id", "conversation_id"],
        ),
        _create_service_ticket,
        agent_id=agent_id,
    )

    tool_registry.register(
        ToolDefinition(
            name="query_my_complaints",
            display_name="查询我的投诉工单",
            description="查询当前用户尚未处理完的投诉工单及状态",
            parameters=[],
            context_params=["user_id", "conversation_id"],
        ),
        _query_my_complaints,
        agent_id=agent_id,
    )
