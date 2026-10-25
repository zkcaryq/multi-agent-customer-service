"""领域 Agent 的公共骨架。

retrieve → rerank → decide_tools → [await_approval] → call_tools → generate

售前/售中/售后/投诉四个 Agent 只在"用哪些工具、说什么话术"上不同，
图结构、审批拦截、工具执行、记忆压缩、run/resume 契约全部收敛在这里，
避免每加一个域就复制一遍 300 行。
"""

from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.config import settings
from app.core.approvals import ensure_pending
from app.core.audit import record_audit
from app.core.checkpointer import get_checkpointer, thread_id
from app.core.hitl import (
    high_risk_calls,
    parse_decision,
    pending_payload,
    should_handoff,
    user_requests_human,
)
from app.core.memory import memory_store
from app.providers import get_chat
from app.rag.reranker import Reranker
from app.rag.retriever import HybridRetriever
from app.rag.retriever import RetrievalResult
from app.tools.executor import invoke_tool
from app.tools.registry import tool_registry


class AgentState(TypedDict):
    conversation_id: int
    user_id: str
    user_query: str
    intent: str
    safety_passed: bool
    safety_message: str
    retrieved_docs: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_round: int
    tool_history: list[str]
    collaboration_context: list[dict[str, Any]]
    final_response: str
    handoff: bool
    handoff_reason: str
    trace: dict[str, Any]
    # Human-in-the-loop：待审批动作与审批结论
    pending_confirmation: dict[str, Any] | None
    approval: dict[str, Any] | None


class DomainAgent:
    """一个业务域的 LangGraph Agent。"""

    AGENT_ID: int = 0
    AGENT_NAME: str = ""
    DECISION_PROMPT: str = ""

    def __init__(self) -> None:
        self.register_tools()
        self.retriever = HybridRetriever(
            bm25_weight=settings.bm25_weight,
            vector_weight=settings.vector_weight,
            top_k=settings.retrieval_top_k,
        )
        self.reranker = Reranker()
        self._builder = self._build_graph()
        self._compiled: Any = None

    # ── 子类需要提供的两块 ──────────────────────────────────────────────────

    def register_tools(self) -> None:
        """把本域工具注册进全局 tool_registry（幂等，可重复调用）。"""
        raise NotImplementedError

    def build_system_prompt(self, knowledge: str) -> str:
        raise NotImplementedError

    # ── 图结构 ──────────────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("decide_tools", self._decide_tools)
        workflow.add_node("await_approval", self._await_approval)
        workflow.add_node("call_tools", self._call_tools)
        workflow.add_node("generate", self._generate)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        # 用户点名要人工时直接生成并转接，不再规划工具——
        # 否则"给我转人工"会先撞上高危操作的审批，人反而被拦在门外。
        workflow.add_conditional_edges(
            "rerank",
            self._route_after_retrieval,
            {"handoff": "generate", "decide": "decide_tools"},
        )
        workflow.add_conditional_edges(
            "decide_tools",
            self._should_call_tools,
            {"approve": "await_approval", "tools": "call_tools", "generate": "generate"},
        )
        workflow.add_conditional_edges(
            "await_approval",
            self._approval_passed,
            {"tools": "call_tools", "generate": "generate"},
        )
        workflow.add_conditional_edges(
            "call_tools",
            self._continue_after_tools,
            {"decide": "decide_tools", "generate": "generate"},
        )
        workflow.add_edge("generate", END)

        return workflow

    async def _get_graph(self) -> Any:
        """interrupt/resume 必须带 checkpointer 编译；checkpointer 初始化是异步的，
        所以编译推迟到首次调用，而不是 import 期的构造函数里。"""
        if self._compiled is None:
            self._compiled = self._builder.compile(checkpointer=await get_checkpointer())
        return self._compiled

    # ── Nodes ──────────────────────────────────────────────────────────────

    async def _retrieve(self, state: AgentState) -> AgentState:
        t0 = time.monotonic()
        try:
            results = await self.retriever.retrieve(state["user_query"])
            state["retrieved_docs"] = [
                {"id": r.chunk_id, "content": r.content, "score": r.score, "source": r.source_document}
                for r in results
            ]
            state["trace"]["retrieval_count"] = len(results)
        except Exception as exc:
            state["retrieved_docs"] = []
            state["trace"]["retrieval_error"] = str(exc)
        state["trace"]["retrieval_ms"] = int((time.monotonic() - t0) * 1000)
        return state

    async def _rerank(self, state: AgentState) -> AgentState:
        t0 = time.monotonic()
        if not state["retrieved_docs"]:
            state["trace"]["rerank_ms"] = 0
            return state
        try:
            raw = [
                RetrievalResult(
                    chunk_id=d["id"], content=d["content"], score=d["score"], source_document=d["source"]
                )
                for d in state["retrieved_docs"]
            ]
            reranked = await self.reranker.rerank(state["user_query"], raw)
            state["retrieved_docs"] = [
                {"id": r.chunk_id, "content": r.content, "score": r.score, "source": r.source_document}
                for r in reranked
            ]
        except Exception as exc:
            state["trace"]["rerank_error"] = str(exc)
        state["trace"]["rerank_ms"] = int((time.monotonic() - t0) * 1000)
        return state

    async def _decide_tools(self, state: AgentState) -> AgentState:
        tools_schema = tool_registry.to_openai_tools(agent_id=self.AGENT_ID)
        if not tools_schema:
            state["tool_calls"] = []
            return state

        knowledge = "\n".join(
            f"[{d['source']}] {d['content']}" for d in state["retrieved_docs"][:3]
        )
        previous = ""
        if state["tool_results"]:
            previous = (
                "\n\n已执行工具及结果：\n"
                + json.dumps(state["tool_results"], ensure_ascii=False)
                + "\n请根据结果判断下一步；信息足够时不要再调用工具，也不要重复相同调用。"
            )
        messages = [
            {
                "role": "system",
                "content": (
                    self.DECISION_PROMPT
                    + f"\n当前是第 {state['tool_round'] + 1} 轮工具规划，最多 {settings.max_tool_rounds} 轮。"
                ),
            },
            {
                "role": "user",
                "content": f"用户问题：{state['user_query']}\n\n已有知识库内容：\n{knowledge}{previous}",
            },
        ]

        t0 = time.monotonic()
        result = await get_chat().chat(
            role=self.AGENT_NAME,
            messages=messages,
            temperature=0.1,
            tools=tools_schema,
        )
        state["trace"]["decision_ms"] = int((time.monotonic() - t0) * 1000)
        executed = set(state["tool_history"])
        next_calls: list[dict[str, Any]] = []
        for tc in result.tool_calls:
            signature = self._tool_signature(tc.name, tc.arguments)
            if signature not in executed:
                next_calls.append({"name": tc.name, "arguments": tc.arguments})
        state["tool_calls"] = next_calls
        return state

    async def _await_approval(self, state: AgentState) -> AgentState:
        """暂停在人工审批上。

        第一次执行到 interrupt() 会抛 GraphInterrupt，把确认单交给调用方；
        resume 后本节点从头重跑，此时 interrupt() 直接返回人工结论，
        因此高危工具只在确认通过后被执行一次。
        """
        risk = high_risk_calls(state["tool_calls"])
        payload = pending_payload(risk, self.AGENT_NAME, state["conversation_id"])
        state["pending_confirmation"] = payload
        await ensure_pending(self.AGENT_NAME, state["conversation_id"],
                            state.get("user_id") or "", payload["actions"])

        raw = interrupt(payload)
        approved, note = parse_decision(raw)
        state["approval"] = {
            "approved": approved,
            "note": note,
            "tools": [c["name"] for c in risk],
        }
        await record_audit(
            action="approval_decided" if approved else "approval_rejected",
            entity_type="conversation",
            entity_id=state["conversation_id"],
            summary=f"{'批准' if approved else '拒绝'}高危操作: {', '.join(state['approval']['tools'])}",
            detail={"actions": payload["actions"], "note": note, "user_id": state.get("user_id")},
            actor=(raw.get("approver") if isinstance(raw, dict) else None) or "operator",
        )
        if not approved:
            state["handoff"] = True
            state["handoff_reason"] = "approval_rejected"
        return state

    async def _call_tools(self, state: AgentState) -> AgentState:
        """执行细节（校验/超时/重试/熔断）统一在 tools.executor。

        user_id / conversation_id 走运行时上下文，模型给的同号参数会被校验层拒绝，
        既防越权，也让工单等写入能挂到正确的会话上。
        """
        context = {
            "user_id": state.get("user_id") or "",
            "conversation_id": state["conversation_id"],
        }
        results = [await invoke_tool(tc["name"], tc["arguments"], context=context)
                   for tc in state["tool_calls"]]
        for call, result in zip(state["tool_calls"], results):
            result["arguments"] = call["arguments"]
        state["tool_results"].extend(results)
        state["tool_history"].extend(
            self._tool_signature(tc["name"], tc["arguments"]) for tc in state["tool_calls"]
        )
        state["tool_round"] += 1
        state["trace"]["tool_rounds"] = state["tool_round"]
        state["trace"]["tool_call_results"] = state["tool_results"]
        return state

    @staticmethod
    def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
        """同一轮流程不重复执行完全相同的工具，避免模型陷入调用循环。"""
        return f"{name}:{json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)}"

    async def _generate(self, state: AgentState) -> AgentState:
        t0 = time.monotonic()

        knowledge = "\n---\n".join(
            f"[{d['source']}] {d['content']}" for d in state["retrieved_docs"][:3]
        )
        system_prompt = self.build_system_prompt(knowledge or "暂无匹配知识")

        memory = memory_store.get_or_create(
            state["conversation_id"],
            max_turns=settings.max_conversation_turns,
            summary_trigger=settings.summary_trigger_turns,
        )
        memory.add_turn("user", state["user_query"])
        await memory.summarize_async()  # 超过阈值才真摘要，否则原样保留

        messages = memory.to_messages(system_prompt)

        if state.get("collaboration_context"):
            advice = "\n\n".join(
                f"[{item['agent']} Agent 建议]\n{item['content']}"
                for item in state["collaboration_context"]
            )
            messages.append({
                "role": "system",
                "content": (
                    "以下是其他领域 Agent 的协作建议。请结合主流程结果统一回答；"
                    "不要声称辅助 Agent 已执行工具或完成业务操作。\n\n" + advice
                ),
            })

        approval = state.get("approval") or {}
        if approval and not approval.get("approved"):
            messages.append({
                "role": "system",
                "content": (
                    f"人工审批未通过（原因：{approval.get('note') or '未说明'}）。"
                    "尚未执行任何退款/工单操作，请向用户说明已转人工跟进，不要声称已办理。"
                ),
            })

        if state["tool_results"]:
            tool_context = "\n\n## 工具查询结果\n"
            for tr in state["tool_results"]:
                data_str = json.dumps(tr.get("data", {}), ensure_ascii=False, indent=2)
                tool_context += f"\n### {tr['tool']}\n{data_str}\n"
            messages.append({"role": "system", "content": tool_context})

        result = await get_chat().chat(
            role=self.AGENT_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )

        reply = result.content
        memory.add_turn("assistant", reply)

        self._detect_handoff(state)

        if hasattr(memory_store, "persist"):
            memory_store.persist(memory)

        state["final_response"] = reply
        state["trace"]["model"] = result.model
        state["trace"]["prompt_tokens"] = result.prompt_tokens
        state["trace"]["completion_tokens"] = result.completion_tokens
        state["trace"]["generation_ms"] = int((time.monotonic() - t0) * 1000)
        return state

    def _detect_handoff(self, state: AgentState) -> None:
        """转人工判定：用户主动要求、工具全失败、知识盲区。"""
        if user_requests_human(state["user_query"]):
            state["handoff"] = True
            state["handoff_reason"] = "user_request"
        elif should_handoff(state["user_query"], self.AGENT_NAME):
            state["handoff"] = True
            state["handoff_reason"] = "complaint_domain"
        elif state["tool_results"] and all(not tr.get("success") for tr in state["tool_results"]):
            state["handoff"] = True
            state["handoff_reason"] = "tool_failure_all"
        elif not state["retrieved_docs"] and not state["tool_results"] and not state["tool_calls"]:
            state["handoff"] = True
            state["handoff_reason"] = "knowledge_gap"

    # ── Conditional edges ───────────────────────────────────────────────────

    @staticmethod
    def _route_after_retrieval(state: AgentState) -> str:
        return "handoff" if user_requests_human(state["user_query"]) else "decide"

    def _should_call_tools(self, state: AgentState) -> str:
        if high_risk_calls(state["tool_calls"]):
            return "approve"
        return "tools" if state["tool_calls"] else "generate"

    def _approval_passed(self, state: AgentState) -> str:
        return "tools" if (state.get("approval") or {}).get("approved") else "generate"

    def _continue_after_tools(self, state: AgentState) -> str:
        """工具结果回到模型继续规划，达到轮次上限后强制收敛到回答。"""
        return "decide" if state["tool_round"] < settings.max_tool_rounds else "generate"

    # ── Public API ──────────────────────────────────────────────────────────

    def _config(self, conversation_id: int) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id(self.AGENT_NAME, conversation_id)}}

    async def run(
        self,
        conversation_id: int,
        user_query: str,
        user_id: str = "",
        collaboration_context: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        initial: AgentState = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "user_query": user_query,
            "intent": self.AGENT_NAME,
            "safety_passed": True,
            "safety_message": "",
            "retrieved_docs": [],
            "tool_calls": [],
            "tool_results": [],
            "tool_round": 0,
            "tool_history": [],
            "collaboration_context": collaboration_context or [],
            "final_response": "",
            "handoff": False,
            "handoff_reason": "",
            "trace": {},
            "pending_confirmation": None,
            "approval": None,
        }
        graph = await self._get_graph()
        return await graph.ainvoke(initial, self._config(conversation_id))

    async def consult(self, user_query: str) -> dict[str, Any]:
        """以只读顾问身份提供领域建议，不调用业务工具，也不产生副作用。"""
        results = await self.retriever.retrieve(user_query)
        if results:
            results = await self.reranker.rerank(user_query, results)
        knowledge = "\n---\n".join(
            f"[{item.source_document}] {item.content}" for item in results[:3]
        )
        prompt = self.build_system_prompt(knowledge or "暂无匹配知识")
        response = await get_chat().chat(
            role=self.AGENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": prompt + "\n你现在是协作顾问，只提供本领域判断，不调用工具、不声称已办理业务。",
                },
                {"role": "user", "content": user_query},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return {
            "agent": self.AGENT_NAME,
            "content": response.content,
            "sources": [item.source_document for item in results[:3]],
        }

    async def resume(self, conversation_id: int, decision: Any) -> AgentState:
        """带着人工结论从暂停点继续。进程重启后同样可用——状态在 PostgreSQL 里。"""
        graph = await self._get_graph()
        return await graph.ainvoke(Command(resume=decision), self._config(conversation_id))
