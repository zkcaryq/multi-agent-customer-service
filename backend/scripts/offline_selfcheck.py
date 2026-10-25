"""离线全链路自检：hash embedding + noop rerank + 桩 chat provider。

不依赖任何外部 API，用来验证 provider 抽象接线正确。
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["RERANK_PROVIDER"] = "noop"
os.environ["CHAT_PROVIDER"] = "openai_compat"
os.environ["CHAT_API_KEY"] = ""
os.environ["MILVUS_LITE_PATH"] = (Path(tempfile.mkdtemp()) / "verify.db").as_posix()
os.environ["LLM_MODEL_DEFAULT"] = "stub-model"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.providers import registry  # noqa: E402
from app.providers.base import ChatResult  # noqa: E402


class StubChat:
    """记录调用参数的假 chat，避免真的发网络请求。"""

    calls: list[dict] = []

    async def chat(self, *, role, messages, temperature=0.7, max_tokens=2048,
                   json_mode=False, tools=None):
        StubChat.calls.append({"role": role, "json_mode": json_mode, "tools": bool(tools)})
        if role == "intent":
            content = '{"intent_name":"check_logistics","target_agent":"after_sales","confidence":0.93}'
        else:
            content = f"[stub:{role}] 已处理"
        return ChatResult(content=content, model=settings.model_for(role),
                          prompt_tokens=10, completion_tokens=5, provider="stub")


registry.override_chat(StubChat())


async def main() -> int:
    print(registry.describe())
    print("\n[1] role->model 解析:", {r: settings.model_for(r) for r in ("intent", "summary", "after_sales")})

    from app.rag.embedder import embedder

    vecs = embedder.embed(["退货政策说明", "物流查询流程"])
    print(f"[2] hash embedding: {len(vecs)} 条, dim={len(vecs[0])}, 与配置一致={len(vecs[0]) == settings.embedding_dim}")

    from app.rag.ingestion import ingest

    stats = ingest(rebuild=True)
    print(f"[3] 入库: {stats['files']} 文件 / {stats['chunks']} 切片 / {stats['milvus_count']} 向量 / BM25 {stats['bm25_count']}")

    from app.rag.retriever import HybridRetriever

    hits = await HybridRetriever().retrieve("我要退货退款，怎么操作")
    print(f"[4] 混合检索命中 {len(hits)} 条，top1={hits[0].chunk_id if hits else '(无)'} score={hits[0].score if hits else '-'}")

    from app.rag.reranker import Reranker
    from app.rag.retriever import RetrievalResult

    rr = await Reranker().rerank("退货", [RetrievalResult("a", "退货流程", 0.9), RetrievalResult("b", "物流", 0.4)])
    print(f"[5] noop rerank 降级排序: {[r.chunk_id for r in rr]}")

    from app.agents.router import classify_intent

    intent = await classify_intent("我要退货退款")
    print(f"[6] 意图分类(LLM 桩): {intent.target_agent} conf={intent.confidence} by={intent.classifier}")

    from app.agents.after_sales.agent import after_sales_agent

    state = await after_sales_agent.run(conversation_id=1, user_query="我要退货退款，怎么操作")
    print(f"[7] Agent 输出: {state['final_response'][:40]!r} trace.model={state['trace'].get('model')} "
          f"tokens={state['trace'].get('prompt_tokens')}/{state['trace'].get('completion_tokens')}")
    print(f"[8] provider 调用记录: {StubChat.calls}")

    # 四个域都跑一遍，确认公共骨架（base_agent）没有把哪个域跑挂
    from app.agents.complaint.agent import complaint_agent
    from app.agents.during_sales.agent import during_sales_agent
    from app.agents.pre_sales.agent import pre_sales_agent

    per_domain = {}
    for idx, dom in enumerate((pre_sales_agent, during_sales_agent, after_sales_agent, complaint_agent), start=101):
        st = await dom.run(conversation_id=idx, user_query="测试问题", user_id="user_001")
        per_domain[dom.AGENT_NAME] = bool(st["final_response"]) and st["trace"].get("model") == settings.model_for(dom.AGENT_NAME)
    print(f"[9] 四域 Agent 全链路: {per_domain}")

    ok = (
        len(vecs[0]) == settings.embedding_dim
        and stats["milvus_count"] > 0
        and len(hits) > 0
        and intent.target_agent == "after_sales"
        and state["final_response"].startswith("[stub:after_sales]")
        and all(per_domain.values())
    )
    print(f"\n结果: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


raise SystemExit(asyncio.run(main()))
