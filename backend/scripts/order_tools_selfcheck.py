"""订单/退款工具自检：真实数据库、越权防护、退款时限、幂等、审计留痕。

前置：先跑 `python scripts/seed_orders.py`
跑法：backend 目录下 `python scripts/order_tools_selfcheck.py`
"""

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.agents.after_sales.tools import register_after_sales_tools  # noqa: E402
from app.models.database import AuditLog, Order, Refund  # noqa: E402
from app.models.engine import async_session_factory  # noqa: E402
from app.tools.executor import invoke_tool  # noqa: E402
from app.tools.registry import tool_registry  # noqa: E402

PASS = 0
TOTAL = 0
OK_ORDER = "ORD-AIRPODS-PAID"        # shipped，在 7/15 天窗口内
STALE_ORDER = "ORD-CASE-DELIVERED"   # completed 20 天前，超无理由窗口
UNPAID_ORDER = "ORD-IPHONE-UNPAID"
OTHER_ORDER = "ORD-OTHER-USER"       # 属于 demo_user2

CTX = {"user_id": "user_001"}


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, TOTAL
    TOTAL += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label} {detail}")
    PASS += 1 if ok else 0


async def _reset_fixture() -> None:
    """把演示订单回到"未申请退款"的初始态，自检可反复执行。"""
    from sqlalchemy import delete

    async with async_session_factory() as s:
        ids = (await s.execute(select(Order.id).where(Order.order_no == OK_ORDER))).scalars().all()
        if ids:
            await s.execute(delete(Refund).where(Refund.order_id.in_(ids)))
        order = (await s.execute(select(Order).where(Order.order_no == OK_ORDER))).scalar_one()
        order.status = "shipped"
        await s.commit()


async def main() -> int:
    register_after_sales_tools()
    await _reset_fixture()

    # 1) 给模型的 function schema 里不能出现 user_id
    schema = tool_registry.to_openai_tools(agent_id=3)
    refund_schema = next(s for s in schema if s["function"]["name"] == "request_refund")
    props = refund_schema["function"]["parameters"]["properties"]
    check("可信参数不进 function schema", "user_id" not in props, f"→ {sorted(props)}")

    # 2) 模型伪造 user_id 被拦
    r = await invoke_tool("query_order", {"order_id": OTHER_ORDER, "user_id": "demo_user2"}, context=CTX)
    blocked = (not r["success"]) and "系统注入" in r["error"]
    check("模型伪造 user_id 被拒", blocked, f"→ {r['error'][:70]}")

    # 3) 查自己的订单成功
    r = await invoke_tool("query_order", {"order_id": OK_ORDER}, context=CTX)
    check("查询本人订单", r["success"] and r["data"]["amount"] == 1899.0, f"→ {r['data']['items'] if r['success'] else r['error']}")

    # 4) 查别人的订单查不到（且不透露是否存在）
    r = await invoke_tool("query_order", {"order_id": OTHER_ORDER}, context=CTX)
    check("越权查询按不存在处理", not r["success"] and "不属于当前账号" in r["error"], f"→ {r['error']}")

    # 5) 物流读的是库里的轨迹 JSON
    r = await invoke_tool("check_logistics", {"order_id": OK_ORDER}, context=CTX)
    ok = r["success"] and r["data"]["carrier"] == "顺丰速运" and len(r["data"]["history"]) == 3
    check("物流与轨迹来自数据库", ok, f"→ {r['data']['status'] if r['success'] else r['error']}")

    # 6) 未支付订单不发退款
    r = await invoke_tool("request_refund", {"order_id": UNPAID_ORDER, "reason": "不想要了"}, context=CTX)
    check("未支付订单拒绝退款", not r["success"] and "尚未支付" in r["error"], f"→ {r['error']}")

    # 7) 超时限拒绝
    r = await invoke_tool("request_refund", {"order_id": STALE_ORDER, "reason": "不想要了"}, context=CTX)
    check("超 7 天无理由窗口被拒", not r["success"] and "期限" in r["error"], f"→ {r['error']}")

    # 8) 窗口内成功建单 + 订单状态流转 + 审计落库
    r = await invoke_tool("request_refund", {"order_id": OK_ORDER, "reason": "商品与描述不符"}, context=CTX)
    refund_no = r["data"]["refund_id"] if r["success"] else None
    check("窗口内退款受理", r["success"] and refund_no, f"→ {refund_no} / {r.get('error','')}")

    async with async_session_factory() as s:
        order = (await s.execute(select(Order).where(Order.order_no == OK_ORDER))).scalar_one()
        audits = (await s.execute(select(AuditLog.action, AuditLog.summary))).all()
    check("订单状态转为 refunding", order.status == "refunding", f"→ {order.status}")
    check("退款动作写入 audit_log", any(a == "refund_submitted" for a, _ in audits),
          f"→ {[(a, sm[:18]) for a, sm in audits][-2:]}")

    # 9) 重复提交幂等：不再产生第二张未终态退款单
    r2 = await invoke_tool("request_refund", {"order_id": OK_ORDER, "reason": "重复点击"}, context=CTX)
    async with async_session_factory() as s:
        open_count = len((await s.execute(
            select(Refund).where(Refund.order_id == order.id, Refund.status.notin_(["refunded", "rejected", "cancelled"]))
        )).scalars().all())
    check("重复提交被幂等吸收", r2["success"] and open_count == 1 and r2["data"].get("refund_id") == refund_no,
          f"→ 未终态退款单 {open_count} 张")

    # 10) 退款进度按单号查，且限本人
    r = await invoke_tool("check_refund_status", {"refund_id": refund_no}, context=CTX)
    check("查询退款进度", r["success"] and r["data"]["refund_id"] == refund_no, f"→ {r['data']['status'] if r['success'] else r['error']}")
    r = await invoke_tool("check_refund_status", {"refund_id": refund_no}, context={"user_id": "demo_user2"})
    check("退款进度越权保护", not r["success"], f"→ {r['error']}")

    # 11) CRM：画像参数不由模型提供，画像内容来自 users/user_attributes/user_tags
    crm_schema = next(s for s in tool_registry.to_openai_tools(agent_id=3)
                      if s["function"]["name"] == "query_customer_profile")
    check("CRM 工具无可填参数（只认上下文）",
          crm_schema["function"]["parameters"]["properties"] == {},
          f"→ {crm_schema['function']['parameters']}")

    r = await invoke_tool("query_customer_profile", {}, context=CTX)
    d = r["data"] if r["success"] else {}
    check("客户画像拼出等级/标签/消费", r["success"]
          and d["attributes"]["member_level"] == "PLUS会员"
          and "高价值" in d["tags"]
          and d["lifetime_spend"] == 2198.0
          and d["open_refund_count"] == 1,
          f"→ {d}")

    r = await invoke_tool("query_service_history", {}, context=CTX)
    check("服务历史可查", r["success"] and "conversations_30d" in r["data"],
          f"→ {r['data'] if r['success'] else r['error']}")

    r = await invoke_tool("query_customer_profile", {}, context={"user_id": "ghost"})
    check("陌生用户不泄露画像", not r["success"] and "未识别" in r["error"], f"→ {r['error']}")

    print(f"\n{PASS}/{TOTAL} 通过")
    return 0 if PASS == TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
