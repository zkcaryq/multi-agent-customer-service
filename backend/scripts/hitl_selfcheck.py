"""HITL 暂停/恢复自检：用真实 PostgreSQL checkpointer 跑通 interrupt → resume。

跑法：backend 目录下 `python scripts/hitl_selfcheck.py`
不依赖大模型（chat 用桩），但依赖路由上的 PostgreSQL。
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    # 同 app/main.py：psycopg3 异步需要 Selector 事件循环
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["RERANK_PROVIDER"] = "noop"
os.environ["CHAT_API_KEY"] = "stub-not-used"
os.environ["MILVUS_LITE_PATH"] = (Path(tempfile.mkdtemp()) / "hitl.db").as_posix()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers import registry  # noqa: E402
from app.providers.base import ChatResult, ToolCall  # noqa: E402


class StubChat:
    """决策阶段固定要求发起退款，生成阶段固定回话术。"""

    async def chat(self, *, role, messages, temperature=0.7, max_tokens=2048,
                   json_mode=False, tools=None):
        if role == "intent":
            content = '{"intent_name":"check_logistics","target_agent":"after_sales","confidence":0.95}'
        elif tools:
            return ChatResult(
                content="", model="stub", provider="stub",
                tool_calls=[ToolCall(id="call_1", name="request_refund",
                                     arguments={"order_id": "ORD-AIRPODS-PAID", "reason": "商品与描述不符"})],
            )
        else:
            content = "已为您提交退款申请，请在个人中心查看进度。"
        return ChatResult(content=content, model="stub", prompt_tokens=9, completion_tokens=9, provider="stub")


registry.override_chat(StubChat())


def pending_of(state: dict):
    ints = state.get("__interrupt__") or []
    return getattr(ints[0], "value", None) if ints else None


async def main() -> int:
    from app.core import checkpointer as cp
    from app.rag.ingestion import ingest

    saver = await cp.get_checkpointer()
    print(f"[0] checkpointer: {type(saver).__name__}")
    if "Postgres" not in type(saver).__name__:
        print("FAIL: 没有连上 PostgreSQL checkpointer，检查 DATABASE_URL")
        return 1

    stats = ingest(rebuild=True)
    print(f"[1] 知识库就绪: {stats['chunks']} 切片")

    from app.agents.after_sales.agent import AfterSalesAgent

    agent = AfterSalesAgent()

    # ── 场景 A：高危动作被拦下，等待人工确认 ────────────────────────────────
    state = await agent.run(conversation_id=9001, user_query="我要给 ORD-AIRPODS-PAID 退款",
                             user_id="user_001")
    pending = pending_of(state)
    print(f"[2] 是否暂停待审批: {bool(pending)}")
    print(f"    确认单: {pending}")
    assert pending and pending["actions"][0]["tool"] == "request_refund", "没有拦住高危工具"
    assert not state["tool_results"], "未确认就执行了工具"

    # ── 场景 B：模拟进程重启，凭 PG 里的 checkpoint 恢复 ────────────────────
    agent._compiled = None
    cp._saver = None
    state2 = await agent.resume(9001, {"approved": True, "note": "客服主管批准"})
    saver2 = await cp.get_checkpointer()
    print(f"[3] 重建实例+重连 checkpointer({type(saver2).__name__}) 后恢复: final={state2['final_response'][:24]!r}")
    refund = next((r for r in state2["tool_results"] if r["tool"] == "request_refund"), None)
    print(f"    退款执行: success={refund and refund['success']} data={refund and str(refund['data'])[:60]}")
    assert state2["approval"]["approved"] is True
    assert refund and refund["success"], "恢复后高危工具没执行"

    # ── 场景 C：人工拒绝 → 不执行，转人工 ──────────────────────────────────
    state3 = await agent.run(conversation_id=9002, user_query="给 ORD-AIRPODS-PAID 退款", user_id="user_001")
    assert pending_of(state3), "第二通会话没有进入审批"
    state4 = await agent.resume(9002, {"approved": False, "note": "金额过大，走人工"})
    print(f"[4] 拒绝后: handoff={state4['handoff']} reason={state4['handoff_reason']} 工具结果={state4['tool_results']}")
    assert state4["handoff"] and state4["handoff_reason"] == "approval_rejected"
    assert not state4["tool_results"], "拒绝后仍执行了工具"

    # ── 场景 D：低危工具不受影响，直通 ─────────────────────────────────────
    from app.core.hitl import parse_decision

    print(f"[5] 自然语言审批解析: 确认→{parse_decision('确认一下')}  拒绝→{parse_decision('先不要')}  模糊→{parse_decision('再说吧')}")
    assert parse_decision("确认")[0] is True
    assert parse_decision("先不要")[0] is False
    assert parse_decision("再说吧")[0] is False, "语义不明必须默认不执行"

    tables = await _list_checkpoint_tables()
    print(f"[6] PG 中的检查点表: {tables}")

    print("\n结果: PASS")
    return 0


async def _list_checkpoint_tables() -> list[str]:
    import psycopg
    from app.config import settings

    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        cur = await conn.execute(
            "select tablename from pg_tables where schemaname='public' and (tablename like 'checkpoint%') order by 1"
        )
        return [r[0] for r in await cur.fetchall()]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
