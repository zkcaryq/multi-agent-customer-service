"""领域 Agent 的公共骨架 —— 整个项目最核心的一个文件。

【它解决什么问题】
售前 / 售中 / 售后 / 投诉四个业务域，流程其实一模一样：
    检索知识 → 重排 → 让模型决定调哪个工具 → （高危则暂停等人工）→ 执行工具 → 组织回答
不同的是「用哪些工具」「说什么话术」。如果每个域各写一遍，等于复制 4 份 400 行，
改一处审批逻辑要改 4 遍。所以把流程收敛成本类，子类只提供三样东西：
    AGENT_ID / AGENT_NAME / DECISION_PROMPT + register_tools() + build_system_prompt()

【它在整条链路里的位置】
    api/chat.py            ← HTTP 入口，负责落库、脱敏、协调多 Agent
      → agents/coordinator.py  ← 决定主 Agent 和辅助 Agent（只读咨询）
        → 本类的 run()          ← 真正跑 LangGraph 状态机
          → providers.get_chat() ← 模型抽象（OpenAI 兼容 / LangChain 适配）
          → rag.retriever / reranker ← 混合检索与重排
          → tools.executor.invoke_tool ← 工具校验、超时、重试、熔断
          → core.approvals / audit / checkpointer ← 审批队列、审计、状态持久化

【输入 / 输出】
输入：conversation_id（会话主键）、user_query（用户这句话）、user_id（外部用户号）、
      collaboration_context（其他领域 Agent 的只读建议，可选）
输出：AgentState（一个普通 dict），调用方从里面取 final_response / handoff /
      pending_confirmation / trace 四样东西组装 HTTP 响应。

【为什么状态是 TypedDict 而不是 dataclass / Pydantic】
LangGraph 的 checkpointer 会把 state 序列化成 JSON 存进 PostgreSQL。
TypedDict 本身就是 dict 的子类语义，序列化零成本、字段可增可减；
Pydantic 模型需要额外的 dump 逻辑，且版本升级时反序列化容易炸。
代价是没有运行时校验 —— 靠 `AgentState` 的类型标注 + 单测兜底。

【为什么图里要拆 6 个节点而不是一个大函数】
1. 每个节点是一个「可恢复的断点」：进程崩了，checkpointer 记的是节点边界；
2. 审批必须停在独立节点 await_approval，interrupt() 才能干净地挂起/恢复；
3. trace 的耗时统计天然按节点切分，监控页能直接看到检索/决策/生成各花多久。
"""

from __future__ import annotations

import json          # 工具结果拼进提示词、工具签名去重都要 JSON
import time          # 每个节点自己掐表，写进 trace
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph   # 状态图：节点 + 边 + 终止标记
from langgraph.types import Command, interrupt  # interrupt=挂起；Command(resume=...)=恢复

from app.config import settings               # 全局配置（模型岗位、轮次上限、检索参数…）
from app.core.approvals import ensure_pending  # 把待审批动作登记进审批表（后台可见）
from app.core.audit import record_audit        # 审批结论落审计日志
from app.core.checkpointer import get_checkpointer, thread_id  # PG 检查点 + 线程号约定
from app.core.hitl import (                    # HITL 判定与语义解析，全部是纯函数
    high_risk_calls,      # 从待执行工具里挑出高危的（退款/改地址/建投诉单）
    parse_decision,       # 把人工回复归一化成 approved/note（支持自然语言）
    pending_payload,      # 组装给前端的确认单（动作、参数、提示语）
    should_handoff,       # 是否该转人工（投诉域恒为真）
    user_requests_human,  # 用户是否明确要求人工
)
from app.core.memory import memory_store       # 会话记忆（Redis 持久化 + 摘要压缩）
from app.providers import get_chat             # 模型供应商抽象，业务层不认识具体厂商
from app.rag.reranker import Reranker          # 重排（供应商不可用时自动降级）
from app.rag.retriever import HybridRetriever  # BM25 + 向量混合检索
from app.rag.retriever import RetrievalResult  # 重排器要求的输入结构
from app.tools.executor import invoke_tool     # 统一工具执行器（校验/熔断/超时/重试）
from app.tools.registry import tool_registry   # 全局工具注册表（按 agent_id 过滤）


class AgentState(TypedDict):
    """一张会话在状态机里流动时携带的全部数据。

    注意：LangGraph 的每个节点都返回「整个 state」，而不是局部变量。
    这样任何一步都能从检查点恢复，代价是 state 要尽量小（别塞大对象）。
    这里的字段分四组：身份、检索、工具、结果/轨迹。
    """

    # ── 身份与输入 ──────────────────────────────────────────────────────
    conversation_id: int
    user_id: str          # 外部用户号（如 user_001），工具层用它做归属校验
    user_query: str       # 本轮用户原话（已脱敏；不会带历史，历史在 memory 里）
    intent: str           # 路由得到的意图名，仅用于日志/追踪
    safety_passed: bool   # 内容安全检测结果（拦截时不会进入本图）
    safety_message: str

    # ── 检索 ────────────────────────────────────────────────────────────
    retrieved_docs: list[dict[str, Any]]  # [{id, content, score, source}]，最多 top_k 条

    # ── 工具 ────────────────────────────────────────────────────────────
    tool_calls: list[dict[str, Any]]      # 本轮模型决定要调的工具 [{name, arguments}]
    tool_results: list[dict[str, Any]]    # 已执行结果 [{tool, success, data, error, arguments, ...}]
    tool_round: int                       # 已执行了几轮（多轮规划的上限由此控制）
    tool_history: list[str]               # 执行过的「工具+参数」签名，用来防重复调用

    # ── 协作与输出 ──────────────────────────────────────────────────────
    collaboration_context: list[dict[str, Any]]  # 辅助 Agent 的只读建议
    final_response: str
    handoff: bool                         # 是否需要转人工
    handoff_reason: str                   # user_request / complaint_domain / tool_failure_all / knowledge_gap / approval_rejected
    trace: dict[str, Any]                 # 各节点耗时、token、工具结果，最终落到 message_traces

    # ── Human-in-the-loop ───────────────────────────────────────────────
    pending_confirmation: dict[str, Any] | None  # 给前端的确认单（暂停时非空）
    approval: dict[str, Any] | None              # 人工结论 {approved, note, tools}


class DomainAgent:
    """一个业务域的 LangGraph Agent。子类只需覆盖常量与两个方法。"""

    # 子类必须覆盖：数值主键用于按 Agent 过滤工具；名字用于 thread_id 和日志
    AGENT_ID: int = 0
    AGENT_NAME: str = ""
    # 决策阶段的系统提示：告诉模型"哪些场景该调工具"，而不是让它自由发挥
    DECISION_PROMPT: str = ""

    def __init__(self) -> None:
        # 注册本域工具到全局表（幂等：同一进程里重复构造 Agent 不会重复注册）
        self.register_tools()
        # 检索器按配置注入权重，换配置不需要改代码
        self.retriever = HybridRetriever(
            bm25_weight=settings.bm25_weight,
            vector_weight=settings.vector_weight,
            top_k=settings.retrieval_top_k,
        )
        # 重排器是懒加载的（第一次调用才建连接），这里只是实例化
        self.reranker = Reranker()
        # 先搭图结构（纯内存操作，很快）
        self._builder = self._build_graph()
        # 编译结果缓存：编译需要 checkpointer，而 checkpointer 是异步初始化的，
        # 所以推迟到 _get_graph()，这里先占位
        self._compiled: Any = None

    # ── 子类需要提供的两块 ──────────────────────────────────────────────────

    def register_tools(self) -> None:
        """把本域工具注册进全局 tool_registry（幂等，可重复调用）。"""
        # 不提供默认实现是故意的：忘记注册工具，Agent 会在决策阶段"无工具可用"，
        # 表现为永远只做问答不办事 —— 这个 bug 很难从日志看出来，所以强制子类实现。
        raise NotImplementedError

    def build_system_prompt(self, knowledge: str) -> str:
        """生成回答阶段的系统提示：角色话术 + 本次检索到的知识。
        知识为空时传「暂无匹配知识」，模型会据此更保守地表达（而不是胡编）。
        """
        raise NotImplementedError

    # ── 图结构 ──────────────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """把流程拆成节点和边。

        拓扑（括号内是「可能跳过」的条件分支）：

            retrieve → rerank ─┬─→ generate            （用户明确要人工，跳过一切工具）
                               └─→ decide_tools ─┬─→ generate       （模型认为不需要工具）
                                                 ├─→ call_tools     （普通工具）
                                                 └─→ await_approval （高危工具）
                                                        │
                                     await_approval ─┬─→ call_tools（批准）
                                                     └─→ generate  （拒绝，直接说明+转人工）
            call_tools ─→ decide_tools（还有轮次额度，让模型看结果继续规划）
                       └→ generate    （到上限，强制收口）
        """
        workflow = StateGraph(AgentState)

        # 节点名即日志里的步骤名，监控页按这些名字展示耗时
        workflow.add_node("retrieve", self._retrieve)          # 混合检索知识库
        workflow.add_node("rerank", self._rerank)              # 重排（可降级）
        workflow.add_node("decide_tools", self._decide_tools)  # 模型决定调什么工具
        workflow.add_node("await_approval", self._await_approval)  # 高危：挂起等人工
        workflow.add_node("call_tools", self._call_tools)      # 执行工具
        workflow.add_node("generate", self._generate)          # 组织最终回答

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")  # 检索必然接重排（重排内部自行处理空结果）
        # 用户点名要人工时直接生成并转接，不再规划工具——
        # 否则"给我转人工"会先撞上高危操作的审批，人反而被拦在门外。
        workflow.add_conditional_edges(
            "rerank",
            self._route_after_retrieval,   # 纯函数：只看 user_query，不消耗 token
            {"handoff": "generate", "decide": "decide_tools"},
        )
        # 决策后三种去向：高危→审批；有普通工具→执行；无工具→直接回答
        workflow.add_conditional_edges(
            "decide_tools",
            self._should_call_tools,
            {"approve": "await_approval", "tools": "call_tools", "generate": "generate"},
        )
        # 审批只有两种结局：批准→执行；拒绝→回答里说明未执行并转人工
        workflow.add_conditional_edges(
            "await_approval",
            self._approval_passed,
            {"tools": "call_tools", "generate": "generate"},
        )
        # 工具执行完回到决策：这是「多轮工具规划」的关键回边，
        # 由 _continue_after_tools 按 max_tool_rounds 截断，防止模型无限调用
        workflow.add_conditional_edges(
            "call_tools",
            self._continue_after_tools,
            {"decide": "decide_tools", "generate": "generate"},
        )
        workflow.add_edge("generate", END)

        return workflow

    async def _get_graph(self) -> Any:
        """interrupt/resume 必须带 checkpointer 编译；checkpointer 初始化是异步的，
        所以编译推迟到首次调用，而不是 import 期的构造函数里。

        为什么一定要带 checkpointer：interrupt() 挂起后，进程里没有任何地方保存
        「这条会话停在 await_approval」这个事实；只有 checkpointer 会把 state 写进
        PostgreSQL。重启后 resume 才能从同一个节点接着跑。
        """
        if self._compiled is None:
            self._compiled = self._builder.compile(checkpointer=await get_checkpointer())
        return self._compiled

    # ── Nodes ──────────────────────────────────────────────────────────────
    #
    # 约定：每个节点都「原地改 state 并返回 state」。LangGraph 会把返回值作为
    # 下一个节点的输入，并（在配置了 checkpointer 时）把它写进检查点。

    async def _retrieve(self, state: AgentState) -> AgentState:
        """节点 1：混合检索（BM25 关键词 + 向量语义，加权融合）。

        失败不抛异常：检索是"增强项"，挂了也要能靠工具和模型常识作答，
        但错误会写进 trace.retrieval_error，监控页一眼能看到。
        """
        t0 = time.monotonic()
        try:
            results = await self.retriever.retrieve(state["user_query"])
            # 只把下游需要的字段塞进 state，不要把整个 RetrievalResult 对象放进去
            # （checkpointer 要 JSON 序列化，自定义类会失败）
            state["retrieved_docs"] = [
                {"id": r.chunk_id, "content": r.content, "score": r.score, "source": r.source_document}
                for r in results
            ]
            state["trace"]["retrieval_count"] = len(results)
        except Exception as exc:  # noqa: BLE001 —— 检索失败必须降级，不能打断整条会话
            state["retrieved_docs"] = []
            state["trace"]["retrieval_error"] = str(exc)
        state["trace"]["retrieval_ms"] = int((time.monotonic() - t0) * 1000)
        return state

    async def _rerank(self, state: AgentState) -> AgentState:
        """节点 2：重排。把候选按与问题的相关性重新打分，截到 top_n。

        两层降级：
        1) 没有候选 → 直接跳过，耗时记 0；
        2) 重排服务不可用 → reranker 内部退回融合分数排序，错误记进 trace。
        """
        t0 = time.monotonic()
        if not state["retrieved_docs"]:
            state["trace"]["rerank_ms"] = 0
            return state
        try:
            # state 里是 dict，重排器要 RetrievalResult，这里做一次显式转换
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
        except Exception as exc:  # noqa: BLE001
            state["trace"]["rerank_error"] = str(exc)
        state["trace"]["rerank_ms"] = int((time.monotonic() - t0) * 1000)
        return state

    async def _decide_tools(self, state: AgentState) -> AgentState:
        """节点 3：让模型决定这一轮要调哪些工具（Function Calling 的"决策"）。

        - 只喂前 3 条知识：再多的上下文收益递减，还会挤占工具结果的位置；
        - 已经执行过的工具会连同结果一起回喂，并明确提示"信息够了就别再调"；
        - 用「工具+参数」签名去重：模型偶尔会重复提交同一调用，这里直接过滤掉。
        """
        # 只暴露本 Agent 的工具（tool_registry 按 AGENT_ID 过滤），
        # 避免售前 Agent 调售后工具这种越界
        tools_schema = tool_registry.to_openai_tools(agent_id=self.AGENT_ID)
        if not tools_schema:
            # 没注册工具就当纯问答，直接进 generate
            state["tool_calls"] = []
            return state

        knowledge = "\n".join(
            f"[{d['source']}] {d['content']}" for d in state["retrieved_docs"][:3]
        )
        previous = ""
        if state["tool_results"]:
            # 多轮规划的关键：把上一轮结果原样给模型看，它才知道"再查一次"还是"够了"
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
                    # 把轮次上限告诉模型，让它有收敛意识；真正的截断在 _continue_after_tools
                    + f"\n当前是第 {state['tool_round'] + 1} 轮工具规划，最多 {settings.max_tool_rounds} 轮。"
                ),
            },
            {
                "role": "user",
                # 用户原话 + 知识 + 历史工具结果，三段拼在一起，模型一次看全
                "content": f"用户问题：{state['user_query']}\n\n已有知识库内容：\n{knowledge}{previous}",
            },
        ]

        t0 = time.monotonic()
        result = await get_chat().chat(
            role=self.AGENT_NAME,     # 岗位名 → 绑定该域配置的模型（model_for(role)）
            messages=messages,
            temperature=0.1,          # 决策要稳定，不要创意
            tools=tools_schema,       # 只有传了 tools，模型才可能返回 tool_calls
        )
        state["trace"]["decision_ms"] = int((time.monotonic() - t0) * 1000)

        executed = set(state["tool_history"])
        next_calls: list[dict[str, Any]] = []
        for tc in result.tool_calls:
            signature = self._tool_signature(tc.name, tc.arguments)
            if signature not in executed:      # 去重：同参数同工具第二次直接丢弃
                next_calls.append({"name": tc.name, "arguments": tc.arguments})
        state["tool_calls"] = next_calls
        return state

    async def _await_approval(self, state: AgentState) -> AgentState:
        """节点 4：暂停在人工审批上（HITL 的核心）。

        第一次执行到 interrupt() 会抛 GraphInterrupt，把确认单交给调用方；
        resume 后本节点从头重跑，此时 interrupt() 直接返回人工结论，
        因此高危工具只在确认通过后被执行一次。

        为什么必须整节点重跑：LangGraph 的恢复语义是「重放当前节点」，
        节点内 interrupt() 之前的代码会再执行一遍 —— 所以 interrupt 前的
        ensure_pending 必须幂等（已有 pending 就跳过），否则每次恢复都插一条审批单。
        """
        risk = high_risk_calls(state["tool_calls"])   # 只把高危动作拿去审批，普通工具不受影响
        payload = pending_payload(risk, self.AGENT_NAME, state["conversation_id"])
        state["pending_confirmation"] = payload
        # 登记进审批表：后台"人工审批"页读的是这张表，不是 LangGraph 的状态
        await ensure_pending(self.AGENT_NAME, state["conversation_id"],
                            state.get("user_id") or "", payload["actions"])

        # 这里是真正的挂起点：第一次进来会在这里中断，把 payload 抛给调用层
        raw = interrupt(payload)
        # 恢复后拿到人工结论；兼容 {"approved": true} 和"同意/拒绝"这类自然语言
        approved, note = parse_decision(raw)
        state["approval"] = {
            "approved": approved,
            "note": note,
            "tools": [c["name"] for c in risk],
        }
        # 无论批准还是拒绝都留痕：审计页要能回答"谁在什么时候放行了什么"
        await record_audit(
            action="approval_decided" if approved else "approval_rejected",
            entity_type="conversation",
            entity_id=state["conversation_id"],
            summary=f"{'批准' if approved else '拒绝'}高危操作: {', '.join(state['approval']['tools'])}",
            detail={"actions": payload["actions"], "note": note, "user_id": state.get("user_id")},
            # 后台审批走令牌里的用户名；用户侧确认没有 approver 字段，回落成 operator
            actor=(raw.get("approver") if isinstance(raw, dict) else None) or "operator",
        )
        if not approved:
            # 拒绝不是"报错"，而是一个正常分支：标记转人工，交给 generate 说明情况
            state["handoff"] = True
            state["handoff_reason"] = "approval_rejected"
        return state

    async def _call_tools(self, state: AgentState) -> AgentState:
        """节点 5：执行本轮的 1..N 个工具。

        执行细节（校验/超时/重试/熔断）统一在 tools.executor。

        user_id / conversation_id 走运行时上下文，模型给的同号参数会被校验层拒绝，
        既防越权，也让工单等写入能挂到正确的会话上。
        """
        # 可信上下文：来自服务端会话，绝不来自模型参数
        context = {
            "user_id": state.get("user_id") or "",
            "conversation_id": state["conversation_id"],
        }
        # 顺序执行而非并发：工具可能写同一张表（如退款改订单状态），
        # 并发会引入不必要的锁竞争；单轮工具数通常 ≤3，串行成本可接受
        results = [await invoke_tool(tc["name"], tc["arguments"], context=context)
                   for tc in state["tool_calls"]]
        for call, result in zip(state["tool_calls"], results):
            # 把模型给的参数一起记进结果，前端/评测/面试复盘时能看清"传了什么"
            result["arguments"] = call["arguments"]
        state["tool_results"].extend(results)
        # 记签名，供下一轮决策去重；同时作为"已经做过什么"的证据
        state["tool_history"].extend(
            self._tool_signature(tc["name"], tc["arguments"]) for tc in state["tool_calls"]
        )
        state["tool_round"] += 1
        state["trace"]["tool_rounds"] = state["tool_round"]
        # 工具结果同时进 trace：监控页和离线评测都读这里
        state["trace"]["tool_call_results"] = state["tool_results"]
        return state

    @staticmethod
    def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
        """同一轮流程不重复执行完全相同的工具，避免模型陷入调用循环。

        用「工具名 + 排序后的参数 JSON」做签名：参数顺序不同但内容相同，
        也应视为同一次调用（sort_keys 保证这一点）。
        """
        return f"{name}:{json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)}"

    async def _generate(self, state: AgentState) -> AgentState:
        """节点 6：组装最终回答。

        消息拼装顺序是有讲究的（越靠后越"权威"）：
            1) system：角色话术 + 本次知识库内容
            2) 会话记忆：摘要 + 最近若干轮（Redis 里恢复，跨进程/跨重启）
            3) 协作建议：其他领域 Agent 的只读意见（若有）
            4) 审批被拒提示：明确禁止声称"已办理"
            5) 工具结果：本轮真实查到的数据（最新、最具体）
        """
        t0 = time.monotonic()

        # 回答阶段多给一条知识（3 条），因为这里不调工具，纯靠上下文
        knowledge = "\n---\n".join(
            f"[{d['source']}] {d['content']}" for d in state["retrieved_docs"][:3]
        )
        system_prompt = self.build_system_prompt(knowledge or "暂无匹配知识")

        # 取出（或新建）该会话的记忆对象；上限/摘要阈值来自配置
        memory = memory_store.get_or_create(
            state["conversation_id"],
            max_turns=settings.max_conversation_turns,
            summary_trigger=settings.summary_trigger_turns,
        )
        memory.add_turn("user", state["user_query"])
        # 超过阈值才真正调模型做摘要；没超过就原样保留（避免每轮都花一次 token）
        await memory.summarize_async()

        messages = memory.to_messages(system_prompt)

        if state.get("collaboration_context"):
            # 辅助 Agent 只提供建议，明确禁止模型声称"对方已执行操作"
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
            # 审批被拒时最危险的幻觉是"已为您退款"，这里用强约束压住
            messages.append({
                "role": "system",
                "content": (
                    f"人工审批未通过（原因：{approval.get('note') or '未说明'}）。"
                    "尚未执行任何退款/工单操作，请向用户说明已转人工跟进，不要声称已办理。"
                ),
            })

        if state["tool_results"]:
            # 工具结果作为最具体的证据放在最后，模型会优先采信
            tool_context = "\n\n## 工具查询结果\n"
            for tr in state["tool_results"]:
                data_str = json.dumps(tr.get("data", {}), ensure_ascii=False, indent=2)
                tool_context += f"\n### {tr['tool']}\n{data_str}\n"
            messages.append({"role": "system", "content": tool_context})

        result = await get_chat().chat(
            role=self.AGENT_NAME,
            messages=messages,
            temperature=0.7,      # 回答阶段允许一点表达自由，但不高（客服要稳）
            max_tokens=2048,
        )

        reply = result.content
        memory.add_turn("assistant", reply)   # 回答也进记忆，下一轮才有多轮上下文

        self._detect_handoff(state)           # 统一在这一步判定是否转人工

        if hasattr(memory_store, "persist"):
            # Redis 实现会把记忆写回（带 TTL）；内存实现没有 persist，就只在进程内
            memory_store.persist(memory)

        state["final_response"] = reply
        # 模型信息进 trace：面试/排障时能回答"这条回答用的哪个模型、花了多少 token"
        state["trace"]["model"] = result.model
        state["trace"]["prompt_tokens"] = result.prompt_tokens
        state["trace"]["completion_tokens"] = result.completion_tokens
        state["trace"]["generation_ms"] = int((time.monotonic() - t0) * 1000)
        return state

    def _detect_handoff(self, state: AgentState) -> None:
        """转人工判定：用户主动要求、工具全失败、知识盲区。

        判定顺序即优先级：用户的显式要求最高，其次领域规则（投诉恒转），
        再是"工具全军覆没"，最后才是"既没知识也没动作"。
        """
        if user_requests_human(state["user_query"]):
            state["handoff"] = True
            state["handoff_reason"] = "user_request"
        elif should_handoff(state["user_query"], self.AGENT_NAME):
            # 投诉域即使没说要人工，也必须有真人跟进
            state["handoff"] = True
            state["handoff_reason"] = "complaint_domain"
        elif state["tool_results"] and all(not tr.get("success") for tr in state["tool_results"]):
            # 调了工具但全失败：说明后端系统性故障，别让模型硬编答案
            state["handoff"] = True
            state["handoff_reason"] = "tool_failure_all"
        elif not state["retrieved_docs"] and not state["tool_results"] and not state["tool_calls"]:
            # 无知识、无工具、无动作 —— 典型的"知识盲区"，别让模型自由发挥
            state["handoff"] = True
            state["handoff_reason"] = "knowledge_gap"

    # ── Conditional edges ───────────────────────────────────────────────────
    # 这几个函数必须是「纯函数」：LangGraph 可能在任意时刻调用它们来决定下一步，
    # 里面不能有 IO，也不能改 state。

    @staticmethod
    def _route_after_retrieval(state: AgentState) -> str:
        """返回 "handoff" 表示直接去 generate；"decide" 表示照常规划工具。"""
        return "handoff" if user_requests_human(state["user_query"]) else "decide"

    def _should_call_tools(self, state: AgentState) -> str:
        """决策后分流：高危 → 审批；有普通工具 → 执行；无工具 → 回答。"""
        if high_risk_calls(state["tool_calls"]):
            return "approve"
        return "tools" if state["tool_calls"] else "generate"

    def _approval_passed(self, state: AgentState) -> str:
        """审批后分流：批准 → 执行；拒绝 → 直接回答（state 里已标记转人工）。"""
        return "tools" if (state.get("approval") or {}).get("approved") else "generate"

    def _continue_after_tools(self, state: AgentState) -> str:
        """工具结果回到模型继续规划，达到轮次上限后强制收敛到回答。

        没有这条限制，模型可能无限"再查一次"，把 token 烧光还回不了话。
        """
        return "decide" if state["tool_round"] < settings.max_tool_rounds else "generate"

    # ── Public API ──────────────────────────────────────────────────────────

    def _config(self, conversation_id: int) -> dict[str, Any]:
        """LangGraph 的线程配置。

        thread_id 必须是「Agent 名 + 会话号」：同一个会话里，售后和投诉是两个
        独立的状态机，共用 thread_id 会互相覆盖检查点。
        """
        return {"configurable": {"thread_id": thread_id(self.AGENT_NAME, conversation_id)}}

    async def run(
        self,
        conversation_id: int,
        user_query: str,
        user_id: str = "",
        collaboration_context: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        """跑完一轮完整流程（可能停在审批）。

        返回的 state 里若有 pending_confirmation，说明流程中断在 await_approval，
        调用方需要把确认单返回给前端，等人工决策后再调 resume()。
        """
        # 初始 state 必须给全字段：TypedDict 不做默认值填充，
        # 少给字段会在第一个节点就 KeyError
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
        # ainvoke 返回的是最终 state；若中途 interrupt，返回值里会多一个 __interrupt__ 键，
        # api/chat.py 靠它判断"这条是待审批"而不是正常回答
        return await graph.ainvoke(initial, self._config(conversation_id))

    async def consult(self, user_query: str) -> dict[str, Any]:
        """以只读顾问身份提供领域建议，不调用业务工具，也不产生副作用。

        多 Agent 协同时，辅助 Agent 走这条路径：只检索 + 生成建议。
        好处是"帮忙看一眼"不会触发退款/建单，避免一次用户提问产生两笔业务操作。
        """
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
                    # 再强调一次"只建议不执行"，防止模型在文本里假装办了事
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
        """带着人工结论从暂停点继续。进程重启后同样可用——状态在 PostgreSQL 里。

        decision 可以是 {"approved": true, "note": "..."}，也可以是"同意"/"拒绝"。
        """
        graph = await self._get_graph()
        # Command(resume=...) 让 LangGraph 把当前节点重放一遍，interrupt() 这次直接返回 decision
        return await graph.ainvoke(Command(resume=decision), self._config(conversation_id))
