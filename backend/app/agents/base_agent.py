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
TypedDict 给普通字典提供静态字段约定，不负责运行时校验或填默认值。
LangGraph 支持多种状态 schema；本项目选择 TypedDict 是因为节点本来就读写 dict。
检查点使用框架的 serializer（默认可使用 msgpack 等编码），不能笼统称为 JSON 存储。
把数据库对象转成普通字段能降低耦合；可序列化类型仍取决于所用 serializer。

【为什么图里要拆 6 个节点而不是一个大函数】
1. 每个节点是一个「可恢复的断点」：进程崩了，checkpointer 记的是节点边界；
2. 本项目把审批独立成 await_approval，便于控制重放前后的副作用；框架并不强制独立节点；
3. trace 的耗时统计天然按节点切分，监控页能直接看到检索/决策/生成各花多久。
"""

# 让全部类型标注用 PEP 585/604 新语法（list[dict[str, Any]] | None）且运行时不求值，
# AgentState 的字段因此可以引用尚未定义的名字
from __future__ import annotations

import json          # 工具结果拼进提示词、工具签名去重都要 JSON
import time          # 每个节点自己掐表，写进 trace
# TypedDict 只给静态字段约定：不校验、不填默认值，漏字段要到第一个节点下标取值时才炸
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

    注意：本项目习惯返回整个 state；LangGraph 也支持只返回发生变化的字段。
    框架按状态 channel/reducer 合并更新后存检查点，恢复并不要求返回整个字典。
    这里的字段分四组：身份、检索、工具、结果/轨迹。
    """

    # ── 身份与输入 ──────────────────────────────────────────────────────
    # 会话主键：既参与 thread_id，也是审批表和审计日志的 entity_id，所以必须是 int（对齐外键）
    conversation_id: int
    user_id: str          # 外部用户号（如 user_001），工具层用它做归属校验
    user_query: str       # 本轮用户原话（已脱敏；不会带历史，历史在 memory 里）
    intent: str           # 路由得到的意图名，仅用于日志/追踪
    safety_passed: bool   # 内容安全检测结果（拦截时不会进入本图）
    # 拦截话术：本类既不读也不写，run() 填了空串后一路原样带着，留给调用方组装响应
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
    # 空串有两种含义：正常生成前停在审批（generate 没跑），或模型返回空文本；调用方靠 __interrupt__ 区分
    final_response: str
    handoff: bool                         # 是否需要转人工
    handoff_reason: str                   # user_request / complaint_domain / tool_failure_all / knowledge_gap / approval_rejected
    trace: dict[str, Any]                 # 各节点耗时、token、工具结果，最终落到 message_traces

    # ── Human-in-the-loop ───────────────────────────────────────────────
    # 给前端的确认单（暂停时非空）；注意中断那次运行里它仍是 None —— 节点在 interrupt() 处抛出，
    # 本次赋值不会合并进 state，确认单只能从返回值的 __interrupt__[0].value 取
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
        # 构造只存参数不建连接；第 4 个参数 similarity_threshold 故意没传，
        # 于是走配置默认值（SIMILARITY_THRESHOLD），这里看不到它就等于少一处可调
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
        # 把 TypedDict 类交给 StateGraph：每个键成为一个 channel，节点返回值按键合并
        workflow = StateGraph(AgentState)

        # 节点名即日志里的步骤名，监控页按这些名字展示耗时
        workflow.add_node("retrieve", self._retrieve)          # 混合检索知识库
        workflow.add_node("rerank", self._rerank)              # 重排（可降级）
        workflow.add_node("decide_tools", self._decide_tools)  # 模型决定调什么工具
        workflow.add_node("await_approval", self._await_approval)  # 高危：挂起等人工
        workflow.add_node("call_tools", self._call_tools)      # 执行工具
        workflow.add_node("generate", self._generate)          # 组织最终回答

        # 入口固定 retrieve：意图/领域路由在 coordinator 做完才进图，图内不再做意图分流
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
        # 全图唯一的无条件终点边：四条条件边都把 generate 当出口，所以任何收口路径都会经过消息拼装与记忆落盘
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
            # await get_checkpointer() 是让出点：同 Agent 的首次并发请求可能都看到 None、各编译一次，
            # 后写覆盖前写（结果等价，只是白付一次编译）
            # 缓存一次成型：若 PG 初始化失败降级成内存 saver，之后也不会重试 PG，只能重启进程
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
            # （避免把 ORM 会话、连接等不可移植对象带入检查点；具体编码由 serializer 决定）
            state["retrieved_docs"] = [
                {"id": r.chunk_id, "content": r.content, "score": r.score, "source": r.source_document}
                for r in results
            ]
            # 口径是"两路融合并截到 top_k 之后的候选数"（默认 5），不是最终进提示词的条数（那里硬截 [:3]）
            state["trace"]["retrieval_count"] = len(results)
        except Exception as exc:  # noqa: BLE001 —— 检索失败必须降级，不能打断整条会话
            # 显式置空而不是留着上轮残值：_rerank 靠"空候选"跳过，_detect_handoff 靠"无知识"判知识盲区
            state["retrieved_docs"] = []
            # 只存 str(exc)：异常对象带连接/游标，既进不了检查点也不能直接给前端
            state["trace"]["retrieval_error"] = str(exc)
        # 计时写在 except 之外：降级路径也要留耗时，监控页才能区分"检索慢"和"检索挂了"
        state["trace"]["retrieval_ms"] = int((time.monotonic() - t0) * 1000)
        return state

    async def _rerank(self, state: AgentState) -> AgentState:
        """节点 2：重排。把候选按与问题的相关性重新打分，截到 top_n。

        两层降级：
        1) 没有候选 → 直接跳过，耗时记 0；
        2) ProviderError → reranker 内部记录 warning 并降级；只有继续向外抛的错误才记进 trace。
        """
        t0 = time.monotonic()
        # 空候选直接收口：省一次必定空转的供应商调用；写 0 而不是留空键，前端不必做存在性判断
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
            # 走到这里说明连 reranker 内部的降级都没兜住：retrieved_docs 保持检索节点的融合分数顺序
            # （下面的重排赋值没执行，等于"没重排"而不是"排坏了"）
            state["trace"]["rerank_error"] = str(exc)
        # 同样在 except 之外计时：降级也要留下耗时，慢和挂是两种告警
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

        # [:3] 与 _generate 的 [:3] 是两处独立截断（改一处不会同步）；[来源] 前缀让模型能在理由里指认出处
        knowledge = "\n".join(
            f"[{d['source']}] {d['content']}" for d in state["retrieved_docs"][:3]
        )
        # 用空串而不是 None 起头：它直接拼进 f-string，None 会在提示词里显示成字面量 "None"
        previous = ""
        if state["tool_results"]:
            # 多轮规划的关键：把上一轮结果原样给模型看，它才知道"再查一次"还是"够了"
            previous = (
                "\n\n已执行工具及结果：\n"
                + json.dumps(state["tool_results"], ensure_ascii=False)
                + "\n请根据结果判断下一步；信息足够时不要再调用工具，也不要重复相同调用。"
            )
        # 两条消息：system 放决策规则+轮次，user 放事实材料；不带跨轮对话，历史摘要只在 generate 注入
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
        # 这里覆盖上轮 decision_ms，记录最后一轮决策耗时；不是多轮决策耗时总和。
        state["trace"]["decision_ms"] = int((time.monotonic() - t0) * 1000)

        # 只包含此前工具节点已记录的签名；本轮 next_calls 内部没有再次去重，也不是持久化业务幂等键。
        executed = set(state["tool_history"])
        next_calls: list[dict[str, Any]] = []
        for tc in result.tool_calls:
            signature = self._tool_signature(tc.name, tc.arguments)
            # 过滤以前轮次的同参调用；失败调用也已进 history，下一轮不能以相同参数重试它。
            if signature not in executed:      # 去重：同参数同工具第二次直接丢弃
                next_calls.append({"name": tc.name, "arguments": tc.arguments})
        # 模型工具名在这里尚未按 AGENT_ID 再授权；schema 可见性与执行权限校验是两回事。
        state["tool_calls"] = next_calls
        return state

    async def _await_approval(self, state: AgentState) -> AgentState:
        """节点 4：暂停在人工审批上（HITL 的核心）。

        第一次执行到 interrupt() 会抛 GraphInterrupt，把确认单交给调用方；
        resume 后本节点从头重跑，此时 interrupt() 直接返回人工结论，
        因此正常路径只在批准后进入工具节点；这不是故障重放下的 exactly-once 保证。

        为什么必须整节点重跑：LangGraph 的恢复语义是「重放当前节点」，
        节点内 interrupt() 之前的代码会再执行一遍 —— 所以 interrupt 前的
        ensure_pending 必须幂等（已有 pending 就跳过），否则每次恢复都插一条审批单。
        """
        risk = high_risk_calls(state["tool_calls"])   # 确认卡片只列高危动作；同批普通工具也等待该审批分支结束
        payload = pending_payload(risk, self.AGENT_NAME, state["conversation_id"])
        state["pending_confirmation"] = payload
        # 登记进审批表：后台"人工审批"页读的是这张表，不是 LangGraph 的状态
        await ensure_pending(self.AGENT_NAME, state["conversation_id"],
                            state.get("user_id") or "", payload["actions"])

        # 这里是真正的挂起点：第一次进来会在这里中断，把 payload 抛给调用层
        # 恢复从节点开头重放，再次执行 ensure_pending；不要把它理解为 Python 栈原地冻结后续跑。
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
        防止模型改写这两个字段，并让写入关联会话；HTTP 用户身份认证仍需单独补齐。
        """
        # 上下文不取自模型 arguments；但 user_id 最初来自未鉴权的 ChatRequest，尚非认证身份。
        context = {
            "user_id": state.get("user_id") or "",
            "conversation_id": state["conversation_id"],
        }
        # 顺序执行而非并发：工具可能写同一张表（如退款改订单状态），
        # 并发会引入不必要的锁竞争；单轮工具数通常 ≤3，串行成本可接受
        results = [await invoke_tool(tc["name"], tc["arguments"], context=context)
                   for tc in state["tool_calls"]]
        for call, result in zip(state["tool_calls"], results):
            # 把模型给的参数一起记进结果，前端与评测复盘时能看清"这条调用传了什么"
            result["arguments"] = call["arguments"]
        state["tool_results"].extend(results)
        # 记签名，供下一轮决策去重；同时作为"已经做过什么"的证据
        state["tool_history"].extend(
            self._tool_signature(tc["name"], tc["arguments"]) for tc in state["tool_calls"]
        )
        # 一轮可含多个工具；max_tool_rounds 限制规划批次数，不等于最多调用三个工具。
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

        消息按以下顺序拼装；靠后放置有助于提示，但不能构成安全保证：
            1) system：角色话术 + 本次知识库内容
            2) 会话记忆：摘要 + 最近若干轮（Redis 里恢复，跨进程/跨重启）
            3) 协作建议：其他领域 Agent 的只读意见（若有）
            4) 审批被拒提示：明确禁止声称"已办理"
            5) 工具结果：本轮真实查到的数据（最新、最具体）
        """
        t0 = time.monotonic()

        # 回答和工具决策都只取前 3 条；即使重排返回 5 条，也不会全部进入提示词。
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
        # 一条 user 消息算一个 Turn，后面 assistant 再算一个；配置里的 8/10 不是 8/10 对问答。
        memory.add_turn("user", state["user_query"])
        # 超过阈值才真正调模型做摘要；没超过就原样保留（避免每轮都花一次 token）
        # 记忆只在 generate 注入；classify_intent 和 _decide_tools 当前不读取跨轮摘要，省略订单号可能失败。
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
            # 把业务结果放在最后作为事实材料；当前只拼 data，success/error 没有传给生成模型。
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

        # 检测发生在生成之后，只修改状态标志；通用 handoff 不会在这里自动创建 HumanHandoff 记录。
        self._detect_handoff(state)           # 统一在这一步判定是否转人工

        if hasattr(memory_store, "persist"):
            # Redis 实现会把记忆写回（带 TTL）；内存实现没有 persist，就只在进程内
            memory_store.persist(memory)

        state["final_response"] = reply
        # 模型信息进 trace：排障时能回答"这条回答用的哪个模型、花了多少 token"
        state["trace"]["model"] = result.model
        # 仅最终生成调用的 usage；不含意图、咨询、工具决策、摘要和 embedding 的用量。
        state["trace"]["prompt_tokens"] = result.prompt_tokens
        state["trace"]["completion_tokens"] = result.completion_tokens
        # 计时从本节点开头开始，含取记忆、摘要及持久化等工作，不等于供应商纯推理耗时。
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
            # 全失败也可能是权限拒绝/参数错误，不一定是系统故障；这里统一标记转人工。
            state["handoff"] = True
            state["handoff_reason"] = "tool_failure_all"
        elif not state["retrieved_docs"] and not state["tool_results"] and not state["tool_calls"]:
            # 无知识、无工具、无动作 —— 典型的"知识盲区"，别让模型自由发挥
            state["handoff"] = True
            state["handoff_reason"] = "knowledge_gap"

    # ── Conditional edges ───────────────────────────────────────────────────
    # 本项目把边函数写成无副作用判断，便于复现。框架支持异步路由；
    # “不做 IO”是这里的设计约定，并不是 LangGraph 的强制限制。

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

        中断结果通过 __interrupt__ 携带确认单；节点尚未返回，不能依赖 pending_confirmation。
        API 的 _respond 读取 Interrupt.value，前端确认后再调用 resume()。
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
        # run 每轮重置工具结果/history/approval；跨轮自然语言记忆另存在 Redis，不能混为图的执行状态。
        return await graph.ainvoke(initial, self._config(conversation_id))

    async def consult(self, user_query: str) -> dict[str, Any]:
        """以只读顾问身份提供领域建议，不调用业务工具，也不产生副作用。

        多 Agent 协同时，辅助 Agent 走这条路径：只检索 + 生成建议。
        好处是"帮忙看一眼"不会触发退款/建单，避免一次用户提问产生两笔业务操作。
        """
        # consult 使用同一全库检索器，没有按领域 metadata 过滤；领域差别主要是提示词。
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
        """带着人工结论从暂停点继续。使用 PostgreSQL saver 且相同 thread_id 时可跨进程恢复；内存降级不具备此能力。

        decision 可以是 {"approved": true, "note": "..."}，也可以是"同意"/"拒绝"。
        """
        graph = await self._get_graph()
        # Command(resume=...) 让 LangGraph 把当前节点重放一遍，interrupt() 这次直接返回 decision
        # 先前审批表已提交但这里恢复失败，会产生“审批已处理、图未完成”的窗口，需要人工补偿。
        return await graph.ainvoke(Command(resume=decision), self._config(conversation_id))
