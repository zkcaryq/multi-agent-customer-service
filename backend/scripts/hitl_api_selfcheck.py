"""HTTP 层 HITL 自检：/api/v1/chat 暂停 → /api/v1/chat/confirm 恢复。

chat 用桩（不需要 API Key），checkpointer 用真实 PostgreSQL。
跑法：backend 目录下 `python scripts/hitl_api_selfcheck.py`
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
os.environ["MILVUS_LITE_PATH"] = (Path(tempfile.mkdtemp()) / "api_hitl.db").as_posix()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers import registry  # noqa: E402
from app.providers.base import ChatResult, ToolCall  # noqa: E402


class StubChat:
    async def chat(self, *, role, messages, temperature=0.7, max_tokens=2048,
                   json_mode=False, tools=None):
        if role == "intent":
            content = '{"intent_name":"check_logistics","target_agent":"after_sales","confidence":0.95}'
        elif tools:
            return ChatResult(content="", model="stub", provider="stub",
                              tool_calls=[ToolCall(id="c1", name="request_refund",
                                                   arguments={"order_id": "ORD-AIRPODS-PAID",
                                                              "reason": "商品与描述不符"})])
        else:
            content = "退款申请已提交，进度可在个人中心查询。"
        return ChatResult(content=content, model="stub", prompt_tokens=7, completion_tokens=7, provider="stub")


registry.override_chat(StubChat())


def main() -> int:
    from fastapi.testclient import TestClient

    from app.main import app
    from app.rag.ingestion import ingest

    stats = ingest(rebuild=True)
    print(f"[0] 知识库: {stats['chunks']} 切片")

    with TestClient(app) as client:
        conv = 7001
        r = client.post("/api/v1/chat", json={
            "conversation_id": conv, "user_id": "user_001", "message": "ORD-AIRPODS-PAID 我要退款",
        })
        assert r.status_code == 200, r.text
        first = r.json()
        print(f"[1] /chat → pending_confirmation={bool(first['pending_confirmation'])}")
        print(f"    actions={first['pending_confirmation'] and first['pending_confirmation']['actions']}")
        assert first["pending_confirmation"], "高危操作没有被拦下来审批"
        # 服务端可能另分配会话主键，审批必须按返回值来（否则找不到 pending 请求）
        conv = first["conversation_id"]

        r2 = client.post("/api/v1/chat/confirm", json={
            "conversation_id": conv, "agent": "after_sales", "approved": True, "note": "主管批准",
        })
        assert r2.status_code == 200, r2.text
        done = r2.json()
        print(f"[2] /chat/confirm → reply={done['reply'][:28]!r} handoff={done['handoff']}")
        print(f"    trace.approval={done['trace'].get('approval') if done['trace'] else None}")
        assert "退款申请已提交" in done["reply"]
        assert done["pending_confirmation"] is None

        # 拒绝路径
        r3 = client.post("/api/v1/chat", json={
            "conversation_id": conv + 1, "user_id": "user_001", "message": "ORD-AIRPODS-PAID 我要退款",
        })
        assert r3.json()["pending_confirmation"], "第二通会话没进审批"
        conv2 = r3.json()["conversation_id"]
        r4 = client.post("/api/v1/chat/confirm", json={
            "conversation_id": conv2, "agent": "after_sales", "approved": False, "note": "金额过大",
            "approver": "duty_manager",
        })
        rejected = r4.json()
        print(f"[3] 拒绝路径 → handoff={rejected['handoff']} reason={rejected['handoff_reason']!r} reply={rejected['reply'][:24]!r}")
        assert rejected["handoff"] and rejected["handoff_reason"] == "approval_rejected"

    print("\n结果: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
