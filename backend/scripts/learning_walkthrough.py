"""对照教程观察真实源码的离线实验，不连接业务库、不调用真实模型。

仓库根目录运行：
    uv run python backend/scripts/learning_walkthrough.py
    uv run python backend/scripts/learning_walkthrough.py approval

框架、融合/切片/记忆/执行器代码直接导入项目；只有模型与存储边界用测试替身。
approval 实验不经过 HTTP、SQL 审批表或 PostgreSQL saver，不能证明这些集成已通过。
所有替换仅作用于本脚本进程，不修改应用配置、数据库、索引或生产逻辑。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

# 允许从仓库根目录启动；路径根据文件位置计算，不依赖当前 PowerShell 目录。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def show(title: str, **values) -> None:
    """只输出实验数据，不打印 settings、连接地址或密钥。"""
    print(f"\n[{title}]")
    for name, value in values.items():
        print(f"  {name}: {value}")


async def fusion() -> None:
    from app.rag.retriever import HybridRetriever, RetrievalResult

    # 这里输入的是已归一化 BM25 分和已过阈值的向量分，直接观察融合公式。
    retriever = HybridRetriever(bm25_weight=0.3, vector_weight=0.7, top_k=3)
    sparse = [RetrievalResult("A", "退款政策", 1.0), RetrievalResult("B", "支付政策", 0.8)]
    dense = [RetrievalResult("B", "支付政策", 0.6), RetrievalResult("C", "发票规则", 0.9)]
    actual = [(item.chunk_id, item.score) for item in retriever._fusion(sparse, dense)]
    # A=0.3，B=0.24+0.42，C=0.63；不是按排名倒数计算 RRF。
    assert actual == [("B", 0.66), ("C", 0.63), ("A", 0.3)]
    show("fusion", result=actual, observation="两路同 chunk_id 相加，缺失路分数为 0")


async def memory() -> None:
    from app.core import memory as module

    mem = module.ConversationMemory(42)
    for index in range(8):
        mem.add_turn("user" if index % 2 == 0 else "assistant", f"消息{index + 1}")
    assert not mem.needs_compaction
    mem.add_turn("user", "第九条消息")
    assert mem.needs_compaction and len(mem.turns) == 9
    # 只替换远端摘要请求，summarize_async 的切分、标记与降级均执行真实源码。
    with patch.object(module, "_summarize_llm", new=AsyncMock(return_value="")):
        await mem.summarize_async()
    assert len(mem.turns) == 4
    assert "消息1" in mem.summary and "消息5" in mem.summary
    assert mem.turns[0].content == "消息6"
    show("memory", kept=[t.content for t in mem.turns], summary=mem.summary,
         observation="第九条消息触发，不是第九对问答；保留四条")


async def retry() -> None:
    from app.core import resilience
    from app.tools import executor
    from app.tools.base import ToolDefinition, ToolResult
    from app.tools.registry import ToolRegistry

    calls = 0

    async def flaky() -> ToolResult:
        nonlocal calls
        calls += 1
        if calls < 4:
            raise TimeoutError("教学模拟超时")
        return ToolResult(success=True, data={"attempts": calls})

    registry = ToolRegistry()
    registry.register(ToolDefinition("learning_retry", "教学重试", "仅内存计数",
                                     retry_count=3, timeout_ms=1000), flaky)
    breaker = resilience.CircuitBreaker()
    # 不真正等待七秒，但保留真实 retry_with_backoff 的循环和退避计算。
    with patch.object(executor, "tool_registry", registry), \
         patch.object(executor, "circuit_check", return_value=breaker), \
         patch.object(resilience.asyncio, "sleep", new_callable=AsyncMock) as sleep:
        result = await executor.invoke_tool("learning_retry", {})
        waits = [call.args[0] for call in sleep.call_args_list]
    assert result["success"] and calls == 4 and waits == [1.0, 2.0, 4.0]
    assert breaker.failure_count == 0
    show("retry", attempts=calls, backoff=waits, observation="额外重试三次，总共四次尝试")

    # 同时观察当前简化半开策略，避免背成“仅放行一个探针，失败立刻重开”。
    with patch.object(resilience.time, "monotonic", return_value=100.0):
        for _ in range(5):
            breaker.record_failure()
    with patch.object(resilience.time, "monotonic", return_value=131.0):
        assert not breaker.is_open()
        breaker.record_failure()
        assert breaker.state == "half_open" and not breaker.is_open()
    show("breaker boundary", state=breaker.state, failures=breaker.failure_count,
         observation="当前半开首次失败仍放行；不是标准独占探针实现")


async def approval() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agents import base_agent
    from app.core.memory import MemoryStore
    from app.providers.base import ChatResult, ToolCall
    from app.rag.retriever import RetrievalResult
    from app.tools import executor
    from app.tools.base import ToolDefinition, ToolParameterDef, ToolResult
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    writes: list[str] = []
    pending: dict[tuple[str, int], list] = {}
    registrations: list[int] = []

    async def fake_refund(order_id: str, user_id: str) -> ToolResult:
        # 用内存列表代替业务写入；故意不调用 after_sales.tools 的真实数据库 handler。
        writes.append(order_id)
        return ToolResult(True, {"order_id": order_id, "status": "pending_review"})

    async def fake_pending(agent, conv, user, actions) -> None:
        registrations.append(conv)
        pending.setdefault((agent, conv), actions)

    class ScriptedChat:
        async def chat(self, *, messages, tools=None, **kwargs):
            if tools and "已执行工具及结果" not in messages[-1]["content"]:
                return ChatResult(tool_calls=[ToolCall("call-1", "request_refund", {"order_id": "LEARN-1"})])
            # 固定回复只验证控制流，不评价真实模型会如何表述。
            return ChatResult(content="教学流程结束，请查看状态和工具执行记录。", model="learning-stub")

    class LearningAgent(base_agent.DomainAgent):
        AGENT_ID = 99
        AGENT_NAME = "learning"
        DECISION_PROMPT = "教学决策"

        def register_tools(self) -> None:
            registry.register(ToolDefinition(
                "request_refund", "教学退款", "仅用于验证审批顺序",
                parameters=[ToolParameterDef("order_id", "string", is_required=True)],
                context_params=["user_id"], retry_count=0,
            ), fake_refund, agent_id=self.AGENT_ID)

        def build_system_prompt(self, knowledge: str) -> str:
            return knowledge

    saver = InMemorySaver()
    with ExitStack() as stack:
        stack.enter_context(patch.object(base_agent, "tool_registry", registry))
        stack.enter_context(patch.object(executor, "tool_registry", registry))
        stack.enter_context(patch.object(base_agent, "get_checkpointer", new=AsyncMock(return_value=saver)))
        stack.enter_context(patch.object(base_agent, "ensure_pending", new=fake_pending))
        stack.enter_context(patch.object(base_agent, "record_audit", new=AsyncMock()))
        stack.enter_context(patch.object(base_agent, "memory_store", MemoryStore()))
        stack.enter_context(patch.object(base_agent, "get_chat", return_value=ScriptedChat()))
        agent = LearningAgent()
        docs = [RetrievalResult("policy", "教学退款规则", 1.0)]
        stack.enter_context(patch.object(agent.retriever, "retrieve", new=AsyncMock(return_value=docs)))
        stack.enter_context(patch.object(agent.reranker, "rerank", new=AsyncMock(return_value=docs)))

        paused = await agent.run(101, "教学退款", user_id="learning-user")
        assert paused.get("__interrupt__") and writes == []
        assert len(pending) == 1 and registrations == [101]
        show("approval pause", interrupt=True, writes=len(writes), pending_rows=len(pending))

        finished = await agent.resume(101, {"approved": True, "note": "教学批准"})
        assert writes == ["LEARN-1"]
        assert registrations == [101, 101] and len(pending) == 1
        assert finished["approval"]["approved"] and finished["tool_round"] == 1
        show("approval resume", registrations=len(registrations), pending_rows=len(pending),
             writes=len(writes), tool_round=finished["tool_round"])

        # 新会话走拒绝分支；同一会话重复审批的业务限制由真实审批表负责，本实验不伪装验证它。
        await agent.run(102, "教学拒绝", user_id="learning-user")
        rejected = await agent.resume(102, {"approved": False, "note": "教学拒绝"})
        assert writes == ["LEARN-1"] and rejected["handoff"]
        show("approval reject", total_writes=len(writes), handoff=rejected["handoff"],
             observation="拒绝未进入工具节点；未验证 PG 持久化、HTTP 鉴权或并发审批")


async def split() -> None:
    from app.rag.splitter import FixedSizeSplitter, MarkdownSplitter, SemanticSplitter, auto_split

    windows = [c.content for c in FixedSizeSplitter(5, 2).split("ABCDEFGHIJ")]
    assert windows == ["ABCDE", "DEFGH", "GHIJ"]
    markdown = MarkdownSplitter().split("# 退款\n七天\n## 地址\n发货前")
    assert len(markdown) == 2 and markdown[1].metadata["heading"] == "地址"
    # 一个超长段不会被段落聚合器再切开，证明 max_chunk_size 不是强上限。
    long = SemanticSplitter(max_chunk_size=10, min_chunk_size=2).split("X" * 20)
    assert len(long[0].content) == 20
    sheet = auto_split("## Sheet1\n商品\t价格\n手机\t100", "xlsx")
    assert "商品\t价格" in sheet[0].content
    show("split", windows=windows, markdown_chunks=len(markdown), paragraph_size=len(long[0].content),
         observation="字符窗口有重叠；段落聚合不是 embedding 语义切分，Excel 表头路由仍有边界")


async def metrics() -> None:
    from app.evaluation.metrics import accuracy, retrieval_case

    item = retrieval_case(["X", "A", "Y"], {"A", "B"}, k=3)
    assert item.hit and item.recall == 0.5 and item.reciprocal_rank == 0.5
    assert accuracy([True, False, None]) == 50.0
    show("metrics", hit=item.hit, recall=item.recall, precision=item.precision,
         reciprocal_rank=item.reciprocal_rank, eleven_of_thirteen=f"{11 / 13 * 100:.2f}%",
         observation="11/13=84.62%，不等于86%；None 不计入 accuracy 分母")


EXPERIMENTS = {fn.__name__: fn for fn in (fusion, memory, retry, approval, split, metrics)}


async def main(selected: str) -> None:
    # 在事件循环创建后阻止业务联网；不阻止 Windows 创建循环所需的内部 socket。
    with ExitStack() as stack:
        stack.enter_context(patch("dotenv.load_dotenv", return_value=False))
        stack.enter_context(patch.dict(os.environ, {
            "DATABASE_URL": "postgresql+asyncpg://learning:learning@127.0.0.1:1/learning",
            "REDIS_URL": "redis://127.0.0.1:1/0", "MILVUS_ENDPOINT": "http://127.0.0.1:1",
            "CHAT_API_KEY": "learning-not-a-real-key", "EMBEDDING_PROVIDER": "hash",
            "RERANK_PROVIDER": "noop", "MAX_TOOL_ROUNDS": "3",
        }))
        stack.enter_context(patch("redis.Redis.ping", side_effect=ConnectionError("教学模式使用内存记忆")))
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.getaddrinfo"):
            stack.enter_context(patch(target, side_effect=AssertionError("离线实验禁止业务联网")))
        chosen = EXPERIMENTS if selected == "all" else {selected: EXPERIMENTS[selected]}
        for name, experiment in chosen.items():
            await experiment()
            print(f"PASS {name}")
    print("\n全部所选实验通过。它们证明源码控制流，不代表真实模型效果或生产集成通过。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default="all", choices=["all", *EXPERIMENTS])
    asyncio.run(main(parser.parse_args().experiment))
