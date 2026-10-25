"""投诉 Agent 自检：路由分流、定级规则、建单落库、进度查询、越权隔离。

前置：`python scripts/seed_orders.py`
跑法：backend 目录下 `python scripts/complaint_selfcheck.py`
"""

import asyncio
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os_env = {
    "EMBEDDING_PROVIDER": "hash",
    "RERANK_PROVIDER": "noop",
    "CHAT_API_KEY": "stub-not-used",
    "MILVUS_LITE_PATH": (Path(tempfile.mkdtemp()) / "complaint.db").as_posix(),
}
import os  # noqa: E402

os.environ.update(os_env)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.agents.complaint.tools import classify_complaint, register_complaint_tools  # noqa: E402
from app.agents.router import _classify_keyword, agent_by_name  # noqa: E402
from app.models.database import AuditLog, HumanHandoff  # noqa: E402
from app.models.engine import async_session_factory  # noqa: E402
from app.providers import registry  # noqa: E402
from app.providers.base import ChatResult, ToolCall  # noqa: E402
from app.tools.executor import invoke_tool  # noqa: E402

PASS = 0
TOTAL = 0
CTX = {"user_id": "user_001", "conversation_id": 8801}


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, TOTAL
    TOTAL += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label} {detail}")
    PASS += 1 if ok else 0


class StubChat:
    """第一次决策要求建单，之后固定安抚话术。"""

    def __init__(self) -> None:
        self.decided = False

    async def chat(self, *, role, messages, temperature=0.7, max_tokens=2048,
                   json_mode=False, tools=None):
        if role == "intent":
            content = '{"intent_name":"complaint","target_agent":"complaint","confidence":0.96}'
        elif tools and not self.decided:
            self.decided = True
            return ChatResult(content="", model="stub", provider="stub",
                              tool_calls=[ToolCall(id="t1", name="create_service_ticket",
                                                   arguments={"detail": "买的耳机三次故障，怀疑假货，要求赔偿",
                                                              "order_id": "ORD-AIRPODS-PAID"})])
        else:
            content = "非常抱歉给您带来困扰，已为您建立人工工单，客服经理会在 2 小时内联系您。"
        return ChatResult(content=content, model="stub", prompt_tokens=8, completion_tokens=8, provider="stub")


stub = StubChat()
registry.override_chat(stub)


def pending_of(state: dict):
    interrupts = state.get("__interrupt__") or []
    return getattr(interrupts[0], "value", None) if interrupts else None


async def main() -> int:
    # 建表 + seed 四个 Agent（conversations.agent_id 有外键，投诉域必须先存在）
    from app.models.engine import init_db

    await init_db()
    register_complaint_tools()

    # 1) 关键词分流：投诉走 complaint，常规售后仍走 after_sales
    hi = _classify_keyword("我要打12315投诉你们，还要曝光")
    normal = _classify_keyword("快递到哪了")
    check("投诉诉求分流到 complaint", hi.target_agent == "complaint", f"→ {hi.intent_name}/{hi.target_agent}")
    check("常规物流仍走 after_sales", normal.target_agent == "after_sales", f"→ {normal.target_agent}")
    check("router 能取到 complaint 实例", agent_by_name("complaint").AGENT_NAME == "complaint")

    # 2) 定级规则可解释
    cases = [
        ("商品是假货我要起诉", "high", "质量问题"),
        ("客服态度差，我要投诉", "medium", "服务态度"),
        ("只是提个建议希望改进", "low", "其他投诉"),
    ]
    got = [(classify_complaint(t)[:2]) for t, _s, _c in cases]
    check("三级定级 + 归类正确",
          got == [("high", "质量问题"), ("medium", "服务态度"), ("low", "其他投诉")], f"→ {got}")

    # 3) 建单落库 + 审计
    r = await invoke_tool("create_service_ticket",
                          {"detail": "买的耳机三次故障，怀疑假货，要求赔偿", "order_id": "ORD-AIRPODS-PAID"},
                          context=CTX)
    ticket_id = r["data"]["ticket_id"] if r["success"] else None
    check("高危投诉建单成功", r["success"] and ticket_id and r["data"]["severity"] == "high"
          and r["data"]["sla_hours"] == 2, f"→ {ticket_id} {r.get('error','')}")

    async with async_session_factory() as s:
        h = (await s.execute(select(HumanHandoff).where(HumanHandoff.reason == "质量问题"))).scalars().first()
        audits = (await s.execute(
            select(AuditLog.action).where(AuditLog.action == "complaint_ticket_created")
        )).scalars().all()
    check("工单写入 human_handoffs", h is not None and h.status == "pending" and h.intent_agent == "complaint",
          f"→ id={getattr(h, 'id', None)} status={getattr(h, 'status', None)}")
    check("建单写入 audit_log", len(audits) >= 1, f"→ {len(audits)} 条")

    # 4) 进度查询与越权隔离
    r = await invoke_tool("query_my_complaints", {}, context=CTX)
    mine = [t["ticket_id"] for t in r["data"]["tickets"]] if r["success"] else []
    check("本人可查未关闭工单", r["success"] and ticket_id in mine, f"→ {mine}")

    r = await invoke_tool("query_my_complaints", {}, context={"user_id": "demo_user2", "conversation_id": 8802})
    other = [t["ticket_id"] for t in r["data"]["tickets"]] if r["success"] else []
    check("查不到别人的工单", r["success"] and ticket_id not in other, f"→ {other}")

    # 5) Agent 全链路：建单前暂停审批，批准后建单并转人工
    agent = agent_by_name("complaint")
    stub.decided = False
    state = await agent.run(conversation_id=8802, user_query="我买的东西是假货，我要投诉赔偿", user_id="user_001")
    check("投诉建单进入人工审批", bool(pending_of(state)), f"→ {pending_of(state)}")
    state = await agent.resume(8802, {"approved": True, "note": "同意建单", "approver": "selfcheck"})
    check("审批后投诉 Agent 全链路跑通",
          state["handoff"] and state["handoff_reason"] == "complaint_ticket_created"
          and bool(state["final_response"]),
          f"→ reason={state['handoff_reason']} reply={state['final_response'][:18]!r}")

    print(f"\n{PASS}/{TOTAL} 通过")
    return 0 if PASS == TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
