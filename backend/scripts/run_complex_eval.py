"""复杂任务成功率评测：多轮对话 + 高危审批 + 数据库终态断言。

跑法（backend 目录下）：
    uv run python scripts/run_complex_eval.py

与 run_evaluation.py 的区别：那个评"单问单答"（意图/检索），这个评"一件事有没有办成"——
工具是否被调用且成功、参数对不对、审批有没有走、落库终态对不对。

判定全部是规则断言，不引入 LLM 当裁判，所以同一结果可复现、可解释。
每条用例使用专属 ORD-EVAL-* 订单，跑前重置退款/地址，避免
uq_refunds_open_per_order 让幂等分支影响结论。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "eval" / "complex_tasks.json"

# 用例专属订单：状态按各用例需要设定，跑前统一重置
EVAL_ORDERS: dict[str, tuple[str, str]] = {
    "ORD-EVAL-REFUND":    ("user_001", "shipped"),
    "ORD-EVAL-REJECT":    ("user_001", "shipped"),
    "ORD-EVAL-IDEM":      ("user_001", "shipped"),
    "ORD-EVAL-QUERY":     ("user_001", "shipped"),
    "ORD-EVAL-LOGISTICS": ("user_001", "shipped"),
    "ORD-EVAL-ADDRESS":   ("user_001", "paid"),      # 未发货才能改地址
    "ORD-EVAL-COMBO":     ("user_001", "shipped"),
    "ORD-EVAL-OTHER":     ("demo_user2", "paid"),    # 越权用例：属于别人
}

DEFAULT_ADDRESS = "北京市朝阳区评测路 1 号"


def _check(ok: bool, name: str, detail: str = "") -> tuple[str, bool, str]:
    return name, ok, detail


async def prepare_eval_data() -> None:
    """建评测用户/订单，并把退款、地址、订单状态重置到用例前提。"""
    from sqlalchemy import delete, select

    from app.core.timeutil import utcnow_naive
    from app.models.database import Order, OrderItem, Refund, User
    from app.models.engine import async_session_factory

    now = utcnow_naive()
    async with async_session_factory() as session:
        users: dict[str, User] = {}
        for external_id in {"user_001", "demo_user2"}:
            user = (await session.execute(
                select(User).where(User.external_id == external_id)
            )).scalar_one_or_none()
            if user is None:
                user = User(external_id=external_id, nickname=external_id,
                            phone=f"1380000{abs(hash(external_id)) % 10000:04d}")
                session.add(user)
            users[external_id] = user
        await session.flush()

        for order_no, (external_id, status) in EVAL_ORDERS.items():
            order = (await session.execute(
                select(Order).where(Order.order_no == order_no)
            )).scalar_one_or_none()
            if order is None:
                order = Order(order_no=order_no)
                session.add(order)
            order.user_id = users[external_id].id
            order.status = status
            order.amount = 399.00
            order.payment_method = "微信支付"
            order.shipping_address = DEFAULT_ADDRESS
            order.created_at = now - timedelta(days=10)
            order.paid_at = now - timedelta(days=9)
            order.shipped_at = None if status == "paid" else now - timedelta(days=2)
            order.completed_at = None
            order.carrier = None if status == "paid" else "顺丰速运"
            order.tracking_no = None if status == "paid" else f"SF{order_no[-9:]}0001"
            order.logistics_status = None if status == "paid" else "运输中"
            order.estimated_delivery = None if status == "paid" else "今天 18:00 前"
            order.logistics_events = None if status == "paid" else json.dumps(
                [{"time": "2026-10-25 10:00", "status": "运输中", "desc": "快件已到达【杭州转运中心】"}],
                ensure_ascii=False,
            )
            await session.flush()

            # 重置退款单与明细，保证反复跑结果一致
            await session.execute(delete(Refund).where(Refund.order_id == order.id))
            items = (await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )).scalars().all()
            if not items:
                session.add(OrderItem(order_id=order.id, sku_name="AirPods Pro 2", qty=1, price=399.00))
        await session.commit()


async def cleanup_eval_data() -> None:
    """删掉评测订单及其退款/明细。

    这些订单金额会计入用户画像（lifetime_spend / paid_order_count），
    不清理会让 order_tools_selfcheck 里写死的数字失衡——评测数据不该污染演示库。
    """
    from sqlalchemy import delete, select

    from app.models.database import Order, OrderItem, Refund
    from app.models.engine import async_session_factory

    async with async_session_factory() as session:
        ids = (await session.execute(
            select(Order.id).where(Order.order_no.in_(tuple(EVAL_ORDERS)))
        )).scalars().all()
        if ids:
            await session.execute(delete(Refund).where(Refund.order_id.in_(ids)))
            await session.execute(delete(OrderItem).where(OrderItem.order_id.in_(ids)))
            await session.execute(delete(Order).where(Order.id.in_(ids)))
        await session.commit()


async def _db_snapshot(order_id: str | None, conversation_id: int) -> dict:
    from sqlalchemy import func, select

    from app.models.database import HumanHandoff, Order, Refund
    from app.models.engine import async_session_factory

    async with async_session_factory() as session:
        refunds = None
        status = None
        address = None
        if order_id:
            order = (await session.execute(
                select(Order).where(Order.order_no == order_id)
            )).scalar_one_or_none()
            if order is not None:
                refunds = (await session.execute(
                    select(func.count()).select_from(Refund).where(Refund.order_id == order.id)
                )).scalar()
                status = order.status
                address = order.shipping_address
        handoffs = (await session.execute(
            select(func.count()).select_from(HumanHandoff)
            .where(HumanHandoff.conversation_id == conversation_id)
        )).scalar()
    return {"refunds": refunds, "status": status, "address": address, "handoffs": handoffs}


def _tool_calls(responses: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for body in responses:
        trace = body.get("trace") or {}
        calls.extend(trace.get("tool_call_results") or [])
    return calls


def _final_reply(responses: list[dict]) -> str:
    for body in reversed(responses):
        if body.get("reply"):
            return body["reply"]
    return ""


async def run_case(client, case: dict) -> tuple[bool, list[tuple[str, bool, str]], dict]:
    expect = case["expect"]
    conversation_id = 0
    responses: list[dict] = []

    for turn in case["turns"]:
        resp = await client.post("/api/v1/chat", json={
            "conversation_id": conversation_id,
            "user_id": case["user_id"],
            "message": turn,
        })
        if resp.status_code != 200:
            return False, [_check(False, "chat_http", f"HTTP {resp.status_code}: {resp.text[:120]}")], {}
        body = resp.json()
        conversation_id = body.get("conversation_id") or conversation_id
        responses.append(body)

        if body.get("pending_confirmation") and "approve" in expect:
            pending = body["pending_confirmation"]
            confirmed = await client.post("/api/v1/chat/confirm", json={
                "conversation_id": pending["conversation_id"],
                "agent": pending["agent"],
                "approved": expect["approve"],
                "note": "complex-eval",
            })
            if confirmed.status_code == 409:
                # 同一会话里第二次高危操作：审批表按 (agent, conversation) 唯一，
                # 上一次已处理过就会被挡回 409。语义是"不会重复执行"，属正确终态，
                # 交给后面的数据库断言裁决，这里不当失败。
                continue
            if confirmed.status_code != 200:
                return False, [_check(False, "confirm_http",
                                      f"HTTP {confirmed.status_code}: {confirmed.text[:120]}")], {}
            responses.append(confirmed.json())

    calls = _tool_calls(responses)
    reply = _final_reply(responses)
    snapshot = await _db_snapshot(case.get("order_id"), conversation_id)
    results: list[tuple[str, bool, str]] = []

    if "tools_all" in expect:
        names = {c.get("tool") for c in calls if c.get("success")}
        missing = [t for t in expect["tools_all"] if t not in names]
        results.append(_check(not missing, "tools_all", f"缺少 {missing}" if missing else ""))
    if "tools_any" in expect:
        names = {c.get("tool") for c in calls if c.get("success")}
        results.append(_check(bool(names & set(expect["tools_any"])), "tools_any",
                              f"实际调用 {sorted(names)}"))
    if "tools_none" in expect:
        hit = {c.get("tool") for c in calls} & set(expect["tools_none"])
        results.append(_check(not hit, "tools_none", f"不该调用 {sorted(hit)}"))
    for tool, expected in (expect.get("tool_success") or {}).items():
        entries = [c for c in calls if c.get("tool") == tool]
        ok = bool(entries) and entries[-1].get("success") is expected
        results.append(_check(ok, f"tool_success[{tool}]",
                              f"实际 {[c.get('success') for c in entries] or '未调用'}"))

    # 只约束"被调用时的成功率"：模型可能改用别的工具达到同样效果，硬绑工具名会把
    # 正确的行为判成失败（越权查询就是这种情况）。
    for tool, expected in (expect.get("tool_success_if_called") or {}).items():
        entries = [c for c in calls if c.get("tool") == tool]
        ok = all(c.get("success") is expected for c in entries)
        results.append(_check(ok, f"tool_success_if_called[{tool}]",
                              f"实际 {[c.get('success') for c in entries]}"))

    for tool, wanted in (expect.get("arguments") or {}).items():
        entries = [c for c in calls if c.get("tool") == tool]
        actual = entries[-1].get("arguments") if entries else {}
        ok = all(str(actual.get(k)) == str(v) for k, v in wanted.items())
        results.append(_check(ok, f"arguments[{tool}]", f"实际 {actual}"))

    if "requires_approval" in expect:
        got = any(r.get("pending_confirmation") for r in responses)
        results.append(_check(got is expect["requires_approval"], "requires_approval",
                              f"期望 {expect['requires_approval']}，实际 {got}"))
    if "should_handoff" in expect:
        got = bool(responses[-1].get("handoff"))
        results.append(_check(got is expect["should_handoff"], "should_handoff",
                              f"期望 {expect['should_handoff']}，实际 {got}"))
    if "must_contain_any" in expect:
        ok = any(k in reply for k in expect["must_contain_any"])
        results.append(_check(ok, "must_contain_any",
                              f"回复未包含任一 {expect['must_contain_any']}"))
    if "must_not_contain" in expect:
        leaked = [k for k in expect["must_not_contain"] if k in reply]
        results.append(_check(not leaked, "must_not_contain", f"回复泄露 {leaked}"))
    if "consultations_any" in expect:
        got = {
            c.get("agent")
            for body in responses
            for c in ((body.get("trace") or {}).get("coordination") or {}).get("consultations") or []
        }
        results.append(_check(bool(got), "consultations_any", f"实际 {sorted(got) or '无辅助 Agent'}"))

    if "db_refunds" in expect:
        results.append(_check(snapshot["refunds"] == expect["db_refunds"], "db_refunds",
                              f"期望 {expect['db_refunds']}，实际 {snapshot['refunds']}"))
    if "db_handoffs_delta" in expect:
        results.append(_check(snapshot["handoffs"] == expect["db_handoffs_delta"], "db_handoffs",
                              f"期望 {expect['db_handoffs_delta']}，实际 {snapshot['handoffs']}"))
    if "db_order_status" in expect:
        results.append(_check(snapshot["status"] == expect["db_order_status"], "db_order_status",
                              f"期望 {expect['db_order_status']}，实际 {snapshot['status']}"))
    if "db_address_contains" in expect:
        ok = expect["db_address_contains"] in (snapshot["address"] or "")
        results.append(_check(ok, "db_address", f"实际 {snapshot['address']!r}"))

    return all(ok for _n, ok, _d in results), results, {
        "reply": reply, "calls": calls, "conversation_id": conversation_id,
    }


async def main() -> int:
    import httpx

    from app.agents import load_all_agents
    from app.models.engine import init_db

    cases = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    only = {int(x) for x in os.getenv("EVAL_ONLY", "").split(",") if x.strip().isdigit()}
    if only:
        cases = [c for c in cases if c["id"] in only]
    debug = os.getenv("EVAL_DEBUG", "") not in ("", "0")

    await init_db()
    load_all_agents()

    # 检索索引尽量加载；加载不了也不影响工具类断言（会打印提示）
    try:
        from app.rag.bm25_store import bm25_store
        from app.rag.ingestion import BM25_INDEX_PATH
        from app.rag.vector_store import vector_store

        if os.path.exists(BM25_INDEX_PATH) and not bm25_store.is_ready:
            bm25_store.load(BM25_INDEX_PATH)
        if vector_store.is_ready:
            vector_store.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 检索索引未就绪，RAG 类断言可能受影响: {exc}")

    await prepare_eval_data()

    from app.main import app

    passed = 0
    by_category: dict[str, list[bool]] = {}
    failures: list[str] = []

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=180) as client:
            for index, case in enumerate(cases, 1):
                ok, checks, info = await run_case(client, case)
                passed += int(ok)
                by_category.setdefault(case["category"], []).append(ok)
                mark = "PASS" if ok else "FAIL"
                print(f"[{index:2d}] {mark}  {case['category']:6}  {case['turns'][0][:34]}")
                if debug:
                    calls = [(c.get("tool"), c.get("success")) for c in info.get("calls", [])]
                    print(f"        tools={calls}  conv={info.get('conversation_id')}")
                    print(f"        reply={info.get('reply', '')[:180]!r}")
                if not ok:
                    for name, check_ok, detail in checks:
                        if not check_ok:
                            print(f"        ↳ {name}: {detail}")
                            failures.append(f"#{case['id']} {name}")
    finally:
        await cleanup_eval_data()

    total = len(cases)
    print("\n按类别：")
    for category, flags in by_category.items():
        print(f"  {category:8} {sum(flags)}/{len(flags)}")

    rate = passed / total * 100 if total else 0.0
    print(f"\n复杂任务成功率: {passed}/{total} = {rate:.1f}%")
    if failures:
        print("失败断言：" + ", ".join(failures))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
