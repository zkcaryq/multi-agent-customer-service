"""审批队列自检：登记 → 后台可见 → 批准恢复 / 超时自动拒绝 → 审计与评测接口。

前置：`python scripts/seed_orders.py`
跑法：backend 目录下 `python scripts/approval_queue_selfcheck.py`

实现要点：全程跑在同一个事件循环里（httpx.ASGITransport 直接驱动 ASGI 应用）。
不要用 TestClient——它自带一个 portal 循环，与脚本自身的 asyncpg 连接池不是
同一个 loop，会报 "another operation is in progress"。
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["RERANK_PROVIDER"] = "noop"
os.environ["CHAT_API_KEY"] = "stub-not-used"
os.environ["MILVUS_LITE_PATH"] = (Path(tempfile.mkdtemp()) / "approval.db").as_posix()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.providers import registry  # noqa: E402
from app.providers.base import ChatResult, ToolCall  # noqa: E402

PASS = 0
TOTAL = 0
USER = "user_001"
ORDER = "ORD-AIRPODS-PAID"


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, TOTAL
    TOTAL += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label} {detail}")
    PASS += 1 if ok else 0


class StubChat:
    """决策阶段固定要求发起退款（高危工具 → 必然进审批）。"""

    async def chat(self, *, role, messages, temperature=0.7, max_tokens=2048,
                   json_mode=False, tools=None):
        if tools:
            return ChatResult(content="", model="stub", provider="stub",
                              tool_calls=[ToolCall(id="c1", name="request_refund",
                                                   arguments={"order_id": ORDER, "reason": "商品与描述不符"})])
        refused = any("人工审批未通过" in str(m.get("content", "")) for m in messages)
        content = ("很抱歉，本次退款申请暂未通过审核，已为您转人工跟进。" if refused
                   else "已为您受理退款申请，进度可在个人中心查看。")
        return ChatResult(content=content, model="stub", prompt_tokens=6,
                          completion_tokens=6, provider="stub")


registry.override_chat(StubChat())


async def _list(client, status: str) -> list[dict]:
    resp = await client.get(f"/api/v1/admin/approvals?status={status}")
    resp.raise_for_status()
    return resp.json()["items"]


async def _pending_of(client, conv_id: int) -> dict | None:
    return next((i for i in await _list(client, "pending") if i["conversation_id"] == conv_id), None)


async def _decided_of(client, conv_id: int) -> dict | None:
    for st in ("approved", "rejected", "expired"):
        for i in await _list(client, st):
            if i["conversation_id"] == conv_id:
                return i
    return None


async def _login(client) -> bool:
    """走真实登录链路拿令牌：账号口令都来自配置，脚本不写死。"""
    anon = await client.get("/api/v1/admin/approvals")
    check("未登录访问后台返回 401", anon.status_code == 401, f"→ HTTP {anon.status_code}")

    wrong = await client.post("/api/v1/auth/login",
                              json={"username": settings.bootstrap_admin_user, "password": "wrong-password"})
    check("口令错误不签发令牌", wrong.status_code == 401, f"→ HTTP {wrong.status_code}")

    resp = await client.post("/api/v1/auth/login", json={
        "username": settings.bootstrap_admin_user, "password": settings.bootstrap_admin_password})
    if resp.status_code != 200:
        check("引导账号可登录", False, f"→ HTTP {resp.status_code} {resp.text[:80]}")
        return False
    check("引导账号可登录", True, f"→ role={resp.json()['user']['role']}")

    client.headers["Authorization"] = f"Bearer {resp.json()['access_token']}"
    me = await client.get("/api/v1/auth/me")
    check("令牌可取回当前坐席", me.status_code == 200
          and me.json()["username"] == settings.bootstrap_admin_user, f"→ HTTP {me.status_code}")

    forged = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    check("伪造令牌被拒", forged.status_code == 401, f"→ HTTP {forged.status_code}")
    return True


async def run_scenarios(client) -> None:
    # 会话号以服务端返回为准：传入的 id 不存在时会新建并返回真实主键
    seed_a, seed_b = 6101, 6102

    # ── 1) 触发审批 ────────────────────────────────────────────────────
    r = await client.post("/api/v1/chat", json={
        "conversation_id": seed_a, "user_id": USER, "message": f"给 {ORDER} 退款"})
    assert r.status_code == 200, r.text
    conv_a = r.json()["conversation_id"]
    pending = r.json()["pending_confirmation"]
    check("高危操作进入审批", bool(pending) and pending["actions"][0]["tool"] == "request_refund")

    row = await _pending_of(client, conv_a)
    check("审批队列已登记且带 user_id", row is not None and row["user_id"] == USER,
          f"→ {row and row['status']}")
    check("确认单含参数（审批人看得见动哪个订单）",
          bool(row) and row["actions"][0]["arguments"].get("order_id") == ORDER)

    # ── 2) 后台批准 → 工具执行 ─────────────────────────────────────────
    dec = await client.post(f"/api/v1/admin/approvals/{row['id']}/decide",
                      json={"approved": True, "note": "主管同意"})
    body = dec.json()
    check("后台审批通过后恢复执行", dec.status_code == 200 and "受理退款" in body.get("reply", ""),
          f"→ HTTP {dec.status_code}")

    done = await _decided_of(client, conv_a)
    approver = settings.bootstrap_admin_user
    check("结论回写 approved + 审批人取自令牌", done is not None and done["status"] == "approved"
          and done["decided_by"] == approver, f"→ {done and (done['status'], done['decided_by'])}")

    tool_results = body.get("trace", {}).get("tool_call_results", [])
    check("退款工具确实执行", any(t["tool"] == "request_refund" and t["success"] for t in tool_results),
          f"→ {[ (t['tool'], t['success']) for t in tool_results ]}")

    audits = (await client.get("/api/v1/admin/audit-logs")).json()["items"]
    kinds = {a["action"] for a in audits}
    check("审计接口含审批与退款痕迹", {"approval_decided", "refund_submitted"} <= kinds,
          f"→ {sorted(kinds)[:6]}")

    # ── 3) 超时自动拒绝（用配置项触发，不改动数据库）──────────────────
    r2 = await client.post("/api/v1/chat", json={
        "conversation_id": seed_b, "user_id": USER, "message": f"给 {ORDER} 退款"})
    assert r2.json()["pending_confirmation"], "第二通会话没进审批"
    conv_b = r2.json()["conversation_id"]
    row_b = await _pending_of(client, conv_b)

    saved_timeout = settings.approval_timeout_s
    settings.approval_timeout_s = 0  # 模拟审批窗口已过
    try:
        dup = await client.post(f"/api/v1/admin/approvals/{row_b['id']}/decide",
                          json={"approved": True, "note": "晚到的批准"})
    finally:
        settings.approval_timeout_s = saved_timeout

    b2 = dup.json()
    expired_view = await _list(client, "expired")
    row_b2 = next((i for i in expired_view if i["conversation_id"] == conv_b), None)
    check("超时后批准被降级为自动拒绝",
          row_b2 is not None and b2.get("handoff_reason") == "approval_rejected"
          and row_b2["decided_by"] == "system",
          f"→ expired 队列 {len(expired_view)} 条, conv_b={conv_b}, decided_by={row_b2 and row_b2['decided_by']}")
    check("超时路径没有执行高危工具", not b2.get("trace", {}).get("tool_call_results"),
          f"→ {b2.get('trace', {}).get('tool_call_results')}")
    check("超时话术不声称已办理", "受理" not in b2.get("reply", ""), f"→ {b2.get('reply', '')[:22]!r}")

    # ── 4) 重复审批幂等 ───────────────────────────────────────────────
    again = await client.post(f"/api/v1/admin/approvals/{row['id']}/decide",
                        json={"approved": False, "note": "重复点击"})
    check("重复审批返回 409", again.status_code == 409, f"→ {again.json().get('detail')}")


async def main() -> int:
    import httpx

    from app.main import app
    from app.models.engine import init_db
    from app.rag.ingestion import ingest

    await init_db()
    ingest(rebuild=True)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        if not await _login(client):
            print(f"\n{PASS}/{TOTAL} 通过（登录未通过，后续场景跳过）")
            return 1

        await run_scenarios(client)

        runs = (await client.get("/api/v1/admin/evaluations")).json()["items"]
        check("评测批次接口可读", len(runs) >= 1 and runs[0]["intent_accuracy"] is not None,
              f"→ {len(runs)} 批，最新 intent={runs[0]['intent_accuracy'] if runs else '-'}%")
        if runs:
            detail = (await client.get(f"/api/v1/admin/evaluations/{runs[0]['id']}?only_wrong=true")).json()
            check("badcase 复盘接口可用", "cases" in detail and detail["run"]["id"] == runs[0]["id"],
                  f"→ 分错 {len(detail.get('cases', []))} 条")

        tickets = (await client.get("/api/v1/admin/tickets")).json()["items"]
        check("工单列表接口可用", isinstance(tickets, list), f"→ {len(tickets)} 张")
        if tickets:
            res = await client.post(f"/api/v1/admin/tickets/{tickets[0]['id']}/resolve",
                              json={"note": "已电话沟通并补发", "status": "resolved"})
            check("工单可回填处理结论", res.status_code == 200
                  and res.json().get("ticket_status") == "resolved", f"→ HTTP {res.status_code}")

    print(f"\n{PASS}/{TOTAL} 通过")
    return 0 if PASS == TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
