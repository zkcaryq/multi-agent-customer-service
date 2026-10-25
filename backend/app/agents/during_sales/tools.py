"""售中 Agent 工具：查询待支付订单、支付状态、修改地址和申请发票。

订单能力统一查询 PostgreSQL 业务表，不再使用进程内演示数据。
user_id 由运行时上下文注入，模型不能指定或查询其他用户的订单。
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select

from app.core.audit import record_audit
from app.models.database import Order, User
from app.models.engine import async_session_factory
from app.tools.base import ToolDefinition, ToolParameterDef, ToolResult
from app.tools.registry import tool_registry


_PAYMENT_METHODS = [
    {"method": "支付宝", "supported": True, "limit_single": 50000},
    {"method": "微信支付", "supported": True, "limit_single": 20000},
    {"method": "银行卡", "supported": True, "limit_single": 100000},
    {"method": "花呗分期", "supported": True, "limit_single": 30000, "installments": [3, 6, 12]},
]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _money(value: Any) -> float:
    return float(value) if isinstance(value, Decimal) else float(value or 0)


def _order_data(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_no,
        "status": order.status,
        "amount": _money(order.amount),
        "items": [
            {"name": item.sku_name, "qty": item.qty, "price": _money(item.price)}
            for item in order.items
        ],
        "created_at": order.created_at.strftime("%Y-%m-%d %H:%M") if order.created_at else None,
        "payment_method": order.payment_method,
        "shipping_address": order.shipping_address,
    }


async def _find_own_order(session, order_id: str, user_id: str) -> Order | None:
    return (await session.execute(
        select(Order)
        .join(User, User.id == Order.user_id)
        .where(Order.order_no == order_id, User.external_id == user_id)
    )).scalar_one_or_none()


async def _query_unpaid_order(user_id: str = "") -> ToolResult:
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(Order)
            .join(User, User.id == Order.user_id)
            .where(
                User.external_id == user_id,
                Order.status.in_(("unpaid", "payment_failed")),
            )
            .order_by(Order.created_at.desc())
        )).scalars().all()

    orders = [_order_data(order) for order in rows]
    return ToolResult(success=True, data={
        "orders": orders,
        "count": len(orders),
        "tip": "请在订单关闭前完成支付" if orders else "当前没有待支付订单",
    })


async def _check_payment_status(order_id: str = "", user_id: str = "") -> ToolResult:
    async with async_session_factory() as session:
        order = await _find_own_order(session, order_id, user_id)
        if order is None:
            return ToolResult(success=False, error=f"未找到订单 {order_id}，或该订单不属于当前账号")
        return ToolResult(success=True, data={
            "order_id": order.order_no,
            "status": order.status,
            "amount": _money(order.amount),
            "payment_method": order.payment_method,
            "available_methods": _PAYMENT_METHODS,
        })


async def _modify_address(
    order_id: str = "", new_address: str = "", user_id: str = "",
) -> ToolResult:
    async with async_session_factory() as session:
        order = await _find_own_order(session, order_id, user_id)
        if order is None:
            return ToolResult(success=False, error=f"未找到订单 {order_id}，或该订单不属于当前账号")
        if order.status != "paid" or order.shipped_at is not None:
            return ToolResult(
                success=False,
                error="仅已支付且未发货的订单支持修改地址",
                data={"order_id": order_id, "current_status": order.status},
            )
        order_db_id = order.id
        old_address = order.shipping_address or ""
        order.shipping_address = new_address
        order.updated_at = _now()
        await session.commit()

    await record_audit(
        action="shipping_address_changed",
        entity_type="order",
        entity_id=order_db_id,
        summary=f"用户 {user_id} 修改订单 {order_id} 收货地址",
        detail={"order_id": order_id, "old_address": old_address, "new_address": new_address},
    )
    return ToolResult(success=True, data={
        "order_id": order_id,
        "old_address": old_address,
        "new_address": new_address,
        "updated_at": _now().isoformat(sep=" "),
    })


async def _apply_invoice(
    order_id: str = "", invoice_type: str = "电子发票", title: str = "个人",
    user_id: str = "",
) -> ToolResult:
    async with async_session_factory() as session:
        order = await _find_own_order(session, order_id, user_id)
        if order is None:
            return ToolResult(success=False, error=f"未找到订单 {order_id}，或该订单不属于当前账号")
        if order.status == "unpaid":
            return ToolResult(
                success=False,
                error="仅已支付订单可申请发票",
                data={"order_id": order_id, "current_status": order.status},
            )
        order_db_id = order.id
        amount = _money(order.amount)

    requested_at = _now()
    request_id = f"INV{requested_at.strftime('%Y%m%d%H%M%S')}{order_db_id:03d}"
    await record_audit(
        action="invoice_requested",
        entity_type="order",
        entity_id=order_db_id,
        summary=f"用户 {user_id} 为订单 {order_id} 申请{invoice_type}",
        detail={"request_id": request_id, "order_id": order_id, "title": title},
    )
    return ToolResult(success=True, data={
        "request_id": request_id,
        "order_id": order_id,
        "amount": amount,
        "type": invoice_type,
        "title": title,
        "status": "pending",
        "status_detail": "发票申请已受理，等待财务系统开具",
        "created_at": requested_at.isoformat(sep=" "),
    })


def register_during_sales_tools(agent_id: int = 2) -> None:
    from app.tools.web_search import register_web_search
    register_web_search(agent_id=None)

    from app.tools.crm import register_crm_tools
    register_crm_tools()

    tool_registry.register(
        ToolDefinition(
            name="query_unpaid_order",
            display_name="查询未支付订单",
            description="查询当前用户未支付或支付失败的订单列表",
            parameters=[],
            context_params=["user_id"],
        ),
        _query_unpaid_order,
        agent_id=agent_id,
    )

    tool_registry.register(
        ToolDefinition(
            name="check_payment_status",
            display_name="查询支付状态",
            description="查询当前用户指定订单的支付状态及可用支付方式",
            parameters=[ToolParameterDef("order_id", "string", "订单编号", is_required=True)],
            context_params=["user_id"],
        ),
        _check_payment_status,
        agent_id=agent_id,
    )

    tool_registry.register(
        ToolDefinition(
            name="modify_address",
            display_name="修改收货地址",
            description="修改当前用户已支付未发货订单的收货地址，执行前需要人工确认",
            parameters=[
                ToolParameterDef("order_id", "string", "订单编号", is_required=True),
                ToolParameterDef("new_address", "string", "新收货地址", is_required=True),
            ],
            context_params=["user_id"],
        ),
        _modify_address,
        agent_id=agent_id,
    )

    tool_registry.register(
        ToolDefinition(
            name="apply_invoice",
            display_name="申请发票",
            description="为当前用户已支付订单提交发票申请",
            parameters=[
                ToolParameterDef("order_id", "string", "订单编号", is_required=True),
                ToolParameterDef("invoice_type", "string", "发票类型：电子发票 / 纸质发票"),
                ToolParameterDef("title", "string", "发票抬头：个人 / 企业名称"),
            ],
            context_params=["user_id"],
        ),
        _apply_invoice,
        agent_id=agent_id,
    )
