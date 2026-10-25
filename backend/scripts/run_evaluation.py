"""离线评测：跑标注集，算意图/检索/工具选择指标，写入 evaluation_runs|results。

只读评测——工具阶段只取 decide_tools 的决策结果，不真正执行工具，
所以不会建退款单、不会建工单。

跑法：
    python scripts/run_evaluation.py                 # 全量
    python scripts/run_evaluation.py --limit 10
    python scripts/run_evaluation.py --no-persist

诚实性说明：当 EMBEDDING_PROVIDER=hash 或没有可用 CHAT_API_KEY 时，
相应指标会是 None 或明显偏低，脚本会在报告里点名当前 provider 配置，
不要把占位配置下的数字当结论。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SET = ROOT.parent / "data" / "eval" / "customer_service_cases.json"


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    return [dict(c, id=i + 1) for i, c in enumerate(cases)]  # case_index 连续，与库里唯一约束对齐


def relevant_chunk_ids(expected_source: str, expected_keyword: str | None) -> set[str]:
    from app.rag.bm25_store import bm25_store

    out = set()
    for doc in bm25_store._docs:
        if doc.source_document != expected_source:
            continue
        if expected_keyword and expected_keyword not in doc.content:
            continue
        out.add(doc.chunk_id)
    return out


async def evaluate(cases: list[dict], top_k: int) -> tuple[dict, list[dict]]:
    from app.agents.base_agent import AgentState
    from app.agents.router import agent_by_name, classify_intent
    from app.config import settings
    from app.core.hitl import should_handoff
    from app.evaluation.metrics import accuracy, mean, retrieval_case
    from app.rag.retriever import HybridRetriever

    retriever = HybridRetriever(top_k=top_k)
    llm_ready = bool(settings.chat_api_key and settings.chat_api_key != "stub-not-used")

    rows: list[dict] = []
    for case in cases:
        question = case["question"]
        intent = await classify_intent(question)
        intent_correct = intent.target_agent == case["expected_agent"]

        retrieved = await retriever.retrieve(question)
        ids = [r.chunk_id for r in retrieved]
        rel = relevant_chunk_ids(case["expected_source"], case.get("expected_keyword")) \
            if case.get("expected_source") else set()

        ret = retrieval_case(ids, rel, top_k) if rel else None

        # 工具决策：只跑 decide_tools，不执行工具，无副作用
        actual_tool = None
        tool_correct = None
        if llm_ready:
            agent = agent_by_name(intent.target_agent)
            state: AgentState = {
                "conversation_id": 0, "user_id": "eval", "user_query": question,
                "intent": intent.target_agent, "safety_passed": True, "safety_message": "",
                "retrieved_docs": [{"id": r.chunk_id, "content": r.content, "score": r.score,
                                    "source": r.source_document} for r in retrieved],
                "tool_calls": [], "tool_results": [], "final_response": "",
                "tool_round": 0, "tool_history": [],
                "collaboration_context": [],
                "handoff": False, "handoff_reason": "", "trace": {},
                "pending_confirmation": None, "approval": None,
            }
            try:
                state = await agent._decide_tools(state)
            except Exception as exc:
                print(f"  decide_tools case{case['id']} 失败: {exc}")
            actual_tool = state["tool_calls"][0]["name"] if state["tool_calls"] else None
            if case.get("expected_tool") or actual_tool:
                tool_correct = actual_tool == case.get("expected_tool")

        rows.append({
            "case_index": case["id"],
            "question": question,
            "expected_intent": case["expected_agent"],
            "actual_intent": intent.target_agent,
            "intent_correct": intent_correct,
            "intent_by": intent.classifier,
            "expected_tool": case.get("expected_tool"),
            "actual_tool": actual_tool,
            "tool_correct": tool_correct,
            "retrieval": ret,
            "retrieved_ids": ids[:top_k],
            "relevant_count": len(rel),
            "handoff_predicted": should_handoff(question, intent.target_agent),
            "handoff_expected": bool(case.get("should_handoff")),
        })

    def _flag(key):
        return [r[key] for r in rows]

    measured = [r["retrieval"] for r in rows if r["retrieval"]]
    summary = {
        "total_cases": len(rows),
        "intent_accuracy": accuracy(_flag("intent_correct")),
        "tool_call_accuracy": accuracy(_flag("tool_correct")),
        "retrieval_hit_rate": accuracy([m.hit for m in measured]),
        "retrieval_recall": mean([m.recall for m in measured]),
        "retrieval_precision": mean([m.precision for m in measured]),
        "retrieval_mrr": mean([m.reciprocal_rank for m in measured]),
        "retrieval_measured": len(measured),
        "handoff_accuracy": accuracy([
            r["handoff_predicted"] == r["handoff_expected"] for r in rows
            if r["handoff_expected"] or r["handoff_predicted"]
        ]),
        "llm_ready": llm_ready,
    }
    return summary, rows


async def persist(name: str, test_set: str, summary: dict, rows: list[dict], top_k: int) -> int:
    from app.models.database import EvaluationResult, EvaluationRun
    from app.models.engine import async_session_factory

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    async with async_session_factory() as session:
        run = EvaluationRun(
            name=name, test_set_name=test_set, total_cases=summary["total_cases"],
            completed_cases=summary["total_cases"], status="completed",
            created_at=now, completed_at=now,
            intent_accuracy=summary["intent_accuracy"],
            tool_call_accuracy=summary["tool_call_accuracy"],
            retrieval_hit_rate=summary["retrieval_hit_rate"],
            retrieval_recall=summary["retrieval_recall"],
            retrieval_precision=summary["retrieval_precision"],
            retrieval_mrr=summary["retrieval_mrr"],
        )
        session.add(run)
        await session.flush()

        for r in rows:
            ret = r["retrieval"]
            session.add(EvaluationResult(
                run_id=run.id, case_index=r["case_index"], question=r["question"],
                expected_intent=r["expected_intent"], actual_intent=r["actual_intent"],
                intent_correct=r["intent_correct"],
                expected_tool=r["expected_tool"], actual_tool=r["actual_tool"],
                tool_correct=r["tool_correct"],
                retrieval_hit=ret.hit if ret else None,
                retrieval_recall=ret.recall if ret else None,
                retrieval_precision=ret.precision if ret else None,
                retrieval_rr=ret.reciprocal_rank if ret else None,
                actual_answer=" | ".join(r["retrieved_ids"][:3]),
            ))
        await session.commit()
        return run.id


def report(summary: dict, rows: list[dict], top_k: int) -> None:
    from app.config import settings

    print(f"\n用例 {summary['total_cases']} 条 | top_k={top_k}")
    print(f"当前配置: chat={settings.chat_provider} key={'有' if summary['llm_ready'] else '无'} | "
          f"embedding={settings.embedding_provider} | rerank={settings.rerank_provider}")
    for key, label in (
        ("intent_accuracy", "意图准确率 %"),
        ("tool_call_accuracy", "工具选择准确率 %"),
        ("handoff_accuracy", "转人工判定准确率 %"),
        ("retrieval_hit_rate", f"检索命中率 %(参与 {summary['retrieval_measured']} 条)"),
        ("retrieval_recall", "检索召回率"),
        ("retrieval_precision", "检索精确率"),
        ("retrieval_mrr", "MRR"),
    ):
        print(f"  {label:28s} {summary[key] if summary[key] is not None else 'N/A'}")

    if settings.embedding_provider == "hash":
        print("  ⚠ embedding=hash 是占位实现，检索类指标只反映 BM25，不代表真实语义检索效果")

    wrong = [r for r in rows if not r["intent_correct"]]
    if wrong:
        print("\n意图分错的用例：")
        for r in wrong[:12]:
            print(f"  #{r['case_index']} 期望 {r['expected_intent']:12s} 实际 {r['actual_intent']:12s} "
                  f"({r['intent_by']}) {r['question'][:26]}")

    missed = [r for r in rows if r["retrieval"] and not r["retrieval"].hit]
    if missed:
        print("\n检索未命中的用例：")
        for r in missed[:12]:
            print(f"  #{r['case_index']} {r['question'][:34]}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_path", type=Path, default=DEFAULT_SET)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()

    from app.config import settings
    from app.models.engine import init_db
    from app.rag.bm25_store import bm25_store
    from app.rag.ingestion import BM25_INDEX_PATH, ingest
    from app.rag.vector_store import vector_store

    if not bm25_store.is_ready and Path(BM25_INDEX_PATH).exists():
        bm25_store.load(BM25_INDEX_PATH)

    if not bm25_store.is_ready or not vector_store.is_ready:
        print(f"[eval] 知识库索引缺失，先重建（embedding={settings.embedding_provider}）...")
        stats = ingest(rebuild=True)
        print(f"[eval] 已入库 {stats['chunks']} 切片 / Milvus {stats['milvus_count']} / BM25 {stats['bm25_count']}")

    cases = load_cases(args.set_path)
    if args.limit:
        cases = cases[: args.limit]

    summary, rows = await evaluate(cases, args.top_k)
    report(summary, rows, args.top_k)

    if not args.no_persist:
        await init_db()
        name = (f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} "
                f"chat={settings.chat_provider}/emb={settings.embedding_provider}/rerank={settings.rerank_provider}")
        run_id = await persist(name, args.set_path.name, summary, rows, args.top_k)
        print(f"\n已写入 evaluation_runs id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
