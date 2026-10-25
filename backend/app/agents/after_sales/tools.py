"""After-sales agent tools — 查订单、查物流、发起退款、查退款进度。

全部打到真实业务表（orders / order_items / refunds），
`user_id` 由运行时上下文注入而不是模型给参数，所以查不到别人的单。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.audit import record_audit
from app.models.database import Order, Refund, User
from app.models.engine import async_session_factory
from app.tools.base import ToolDefinition, ToolParameterDef, ToolResult
from app.tools.registry import tool_registry

# 退货时限：无理由 7 天（签收起算），质量问题 15 天
RETURN_WINDOW_DAYS = {"no_reason": 7, "quality": 15}
_QUALITY_HINTS = ("质量", "坏了", "破损", "故障", "没法用", "不能用", "缺件")

_POLICY = {
    "no_reason": "自签收之日起 7 天内可无理由退货，运费由用户承担",
    "quality": "质量问题 15 天内可退货/换货，运费由商家承担",
    "not_applicable": ["已拆封的消耗品（如食品、化妆品）", "定制商品", "已激活的数码产品（如手机、电脑）"],
    "refund_timeline": "审核通过后 3-5 个工作日到账",
}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _money(value: Any) -> float:
    """Numeric 列取出来是 Decimal，必须转 float：Decimal 既不能 json.dumps，
    也进不了 LangGraph 的 msgpack checkpoint。"""
    return float(value) if isinstance(value, Decimal) else float(value or 0)


def _order_data(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_no,
        "status": order.status,
        "amount": _money(order.amount),
        "items": [
            {"name": i.sku_name, "qty": i.qty, "price": _money(i.price)} for i in order.items
        ],
        "created_at": order.created_at.strftime("%Y-%m-%d %H:%M") if order.created_at else None,
        "payment_method": order.payment_method,
        "shipping_address": order.shipping_address,
    }


def _logistics_data(order: Order) -> dict[str, Any]:
    try:
        events = json.loads(order.logistics_events or "[]")
    except json.JSONDecodeError:
        events = []
    return {
        "order_id": order.order_no,
        "carrier": order.carrier,
        "tracking_no": order.tracking_no,
        "status": order.logistics_status,
        "status_detail": events[-1]["status"] if events else None,
        "estimated_delivery": order.estimated_delivery,
        "history": events,
    }


async def _find_own_order(order_id: str, user_id: str) -> tuple[Order | None, str | None]:
    """按订单号 + 当前登录用户取单。查不到统一回同一句话，不透露订单是否存在。"""
    async with async_session_factory() as session:
        row = (await session.execute(
            select(Order)
            .join(User, User.id == Order.user_id)
            .where(Order.order_no == order_id, User.external_id == user_id)
        )).scalar_one_or_none()
        if row is None:
            return None, f"未找到订单 {order_id}，或该订单不属于当前账号，请确认订单号"
        await session.refresh(row)  # items 是 relationship，需要关联数据可用
        return row, None


async def _check_logistics(order_id: str = "", user_id: str = "") -> ToolResult:
    order, err = await _find_own_order(order_id, user_id)
    if err:
        return ToolResult(success=False, error=err)
    if not order.tracking_no:
        return ToolResult(
            success=False,
            error=f"订单 {order_id} 尚未发货，暂无物流信息",
            data={"order_id": order_id, "status": order.status},
        )
    return ToolResult(success=True, data=_logistics_data(order))


async def _query_order(order_id: str = "", user_id: str = "") -> ToolResult:
    order, err = await _find_own_order(order_id, user_id)
    if err:
        return ToolResult(success=False, error=err)
    return ToolResult(success=True, data=_order_data(order))


def _refund_window(order: Order, reason: str) -> tuple[bool, str]:
    """返回 (是否在时限内, 说明)。"""
    start = order.completed_at or order.shipped_at or order.paid_at
    if start is None:
        return False, "订单尚未支付，无需退款"

    quality = any(k in (reason or "") for k in _QUALITY_HINTS)
    limit = RETURN_WINDOW_DAYS["quality" if quality else "no_reason"]
    used = (_now() - start).days
    if used > limit:
        return False, (
            f"已超过{'质量' if quality else '无理由'}退货期限（签收后 {used} 天，限 {limit} 天），请转人工处理"
        )
    return True, f"在{'质量' if quality else '无理由'}退货期限内（{used}/{limit} 天）"


async def _request_refund(order_id: str = "", reason: str = "", user_id: str = "") -> ToolResult:
    order, err = await _find_own_order(order_id, user_id)
    if err:
        return ToolResult(success=False, error=err)

    ok, note = _refund_window(order, reason)
    if not ok:
        return ToolResult(success=False, error=note, data={"order_id": order_id, "policy": _POLICY})

    refund = Refund(
        refund_no=f"REF{_now().strftime('%Y%m%d%H%M%S')}{order.id:03d}",
        order_id=order.id,
        user_id=order.user_id,
        amount=order.amount,
        reason=reason or "未填写",
        status="pending_review",
    )
    try:
        async with async_session_factory() as session:
            session.add(refund)
            # _find_own_order 的 session 已关闭，这里的 order 是游离对象，
            # 直接改字段不会被持久化，必须在新 session 里取回再更新
            db_order = await session.get(Order, order.id)
            db_order.status = "refunding"
            await session.commit()
    except IntegrityError:
        # 同一订单已有未终态退款单：恢复/重试导致的重复提交在这里天然幂等
        async with async_session_factory() as session:
            existing = (await session.execute(
                select(Refund).where(Refund.order_id == order.id).order_by(Refund.id.desc())
            )).scalars().first()
        return ToolResult(
            success=True,
            data={
                "refund_id": existing.refund_no if existing else None,
                "order_id": order_id,
                "status": existing.status if existing else "pending_review",
                "status_detail": "该订单已有退款申请正在处理中，未重复提交",
                "idempotent": True,
            },
        )

    await record_audit(
        action="refund_submitted",
        entity_type="refund",
        entity_id=refund.id,
        summary=f"用户 {user_id} 对订单 {order_id} 发起退款 {note}",
        detail={"order_id": order_id, "amount": _money(refund.amount), "reason": reason},
    )

    return ToolResult(success=True, data={
        "refund_id": refund.refund_no,
        "order_id": order_id,
        "amount": _money(refund.amount),
        "reason": refund.reason,
        "status": refund.status,
        "status_detail": "退款申请已提交，等待商家审核",
        "eligibility": note,
        "estimated_refund": _POLICY["refund_timeline"],
        "policy": _POLICY,
        "created_at": refund.created_at.strftime("%Y-%m-%d %H:%M:%S") if refund.created_at else None,
    })


async def _check_refund_status(refund_id: str = "", user_id: str = "") -> ToolResult:
    async with async_session_factory() as session:
        refund = (await session.execute(
            select(Refund)
            .join(User, User.id == Refund.user_id)
            .where(Refund.refund_no == refund_id, User.external_id == user_id)
        )).scalar_one_or_none()

    if refund is None:
        return ToolResult(success=False, error=f"未找到退款单 {refund_id}，或不属于当前账号")

    return ToolResult(success=True, data={
        "refund_id": refund.refund_no,
        "order_id": refund.order.order_no,
        "amount": _money(refund.amount),
        "reason": refund.reason,
        "status": refund.status,
        "status_detail": {
            "pending_review": "退款申请已提交，等待商家审核",
            "approved": "审核通过，等待退款到账",
            "refunded": "退款已到账",
            "rejected": "审核未通过",
        }.get(refund.status, refund.status),
        "approval_note": refund.approval_note,
        "created_at": refund.created_at.strftime("%Y-%m-%d %H:%M:%S") if refund.created_at else None,
        "estimated_refund": _POLICY["refund_timeline"],
    })


# ── Register tools ────────────────────────────────────────────────────────────

def register_after_sales_tools(agent_id: int = 3) -> None:
    from app.tools.web_search import register_web_search
    register_web_search(agent_id=None)

    from app.tools.crm import register_crm_tools
    register_crm_tools()

    tool_registry.register(
        ToolDefinition(
            name="check_logistics",
            display_name="查询物流",
            description="查询订单物流状态、快递公司、运单号和配送进度",
            parameters=[
                ToolParameterDef("order_id", "string", "订单编号", is_required=True),
            ],
            context_params=["user_id"],
        ),
        _check_logistics,
        agent_id=agent_id,
    )

    tool_registry.register(
        ToolDefinition(
            name="query_order",
            display_name="查询订单",
            description="查询订单详细信息：状态、金额、商品、收货地址",
            parameters=[
                ToolParameterDef("order_id", "string", "订单编号", is_required=True),
            ],
            context_params=["user_id"],
        ),
        _query_order,
        agent_id=agent_id,
    )

    tool_registry.register(
        ToolDefinition(
            name="request_refund",
            display_name="发起退款",
            description="为已支付订单发起退款/退货申请，需填写退款原因",
            parameters=[
                ToolParameterDef("order_id", "string", "订单编号", is_required=True),
                ToolParameterDef("reason", "string", "退款原因（如：质量问题、不想要了等）", is_required=True),
            ],
            context_params=["user_id"],
        ),
        _request_refund,
        agent_id=agent_id,
    )

    tool_registry.register(
        ToolDefinition(
            name="check_refund_status",
            display_name="查询退款进度",
            description="查询退款申请的处理进度和预计到账时间",
            parameters=[
                ToolParameterDef("refund_id", "string", "退款单号", is_required=True),
            ],
            context_params=["user_id"],
        ),
        _check_refund_status,
        agent_id=agent_id,
    )
