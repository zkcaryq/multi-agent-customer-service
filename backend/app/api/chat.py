"""用户侧聊天接口：/chat、/chat/confirm、/feedback，以及它们背后的落库逻辑。

【这一层的职责】
只做 HTTP 契约与"会话生命周期"，不碰业务判断：
1. 脱敏 + 内容安全（命中即拦截，不进入 Agent）；
2. 调 coordinator 分诊（第 07 章）并**在跑图之前确定会话主键**；
3. 跑图 / 恢复审批（第 06、09 章）；
4. 把结果落库（messages + message_traces）并组装响应。

【为什么会话主键必须在这里定】
同一个 ID 同时是：数据库外键、Redis 记忆 key、LangGraph thread_id、审批表会话号。
先跑图再建会话，会导致第一轮的状态落在临时 ID 上，第二轮换 ID 上下文就断了。

【错误边界】
- 会话存储不可用 → 503（不能拿客户端临时 ID 继续跑，否则审批会出现孤儿状态）
- 模型/供应商不可用 → 503（细节进日志，给用户一句话）
- 其它异常 → 500（同样不泄漏内部异常文本）
"""

import json

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.agents.coordinator import coordinator
from app.agents.router import agent_by_name
from app.core.approvals import decide
from app.core.log import get_logger, set_trace
from app.core.security import Severity, SafetyAction, content_safety, mask_sensitive
from app.models.database import (
    Conversation,
    Feedback,
    Message,
    MessageTrace,
    TraceFunctionCall,
    User,
    Agent,
)
from app.models.engine import async_session_factory
from app.providers.base import ProviderError
from app.schemas.chat import ChatRequest, ChatResponse, ConfirmRequest, FeedbackRequest
from app.config import settings

log = get_logger(__name__)


def _utcnow() -> datetime:
    """本库的 datetime 列是 naive TIMESTAMP，统一写 naive UTC。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)

router = APIRouter(tags=["chat"])


async def _get_or_create_user(user_id: str) -> int:
    """Get existing user or create one. Returns user DB id."""
    async with async_session_factory() as session:
        from sqlalchemy import select

        # 执行用户查找 SQL；external_id 来自请求字段，查到用户不等于完成了登录认证。
        result = await session.execute(
            select(User).where(User.external_id == user_id).limit(1)
        )
        user = result.scalar_one_or_none()
        # 已有用户复用数据库主键，供 Conversation.user_id 外键使用。
        if user:
            return user.id

        user = User(external_id=user_id, nickname=user_id, phone="")
        # 把待新增对象加入事务；并发首次请求仍可能竞争 external_id 唯一约束。
        session.add(user)
        await session.commit()
        return user.id


async def _get_or_create_conversation(
    user_db_id: int, intent_agent: str, requested_id: int = 0,
) -> int:
    """复用当前用户已有会话；不存在时创建，保证记忆、消息和 checkpoint 同 ID。

    `requested_id` 来自客户端（0 表示"新会话"）。这里做两件事：
    ① 范围校验（>0 且 <= INT32 上限）—— 前端曾用 Date.now() 的 13 位毫秒时间戳
       当会话号，进审批表时整型溢出；校验放在入口，错误立即可见；
    ② 归属校验 —— 只有属于当前用户的会话才复用，防止拿到别人的会话号续聊。
    """
    async with async_session_factory() as session:
        from sqlalchemy import select

        agent_id = (await session.execute(
            select(Agent.id).where(Agent.name == intent_agent).limit(1)
        )).scalar_one_or_none()

        if 0 < requested_id <= 2_147_483_647:
            existing = (await session.execute(
                select(Conversation).where(
                    Conversation.id == requested_id,
                    # 这是按“请求声称的用户”检查会话归属；当前无用户 JWT，不能视为完整防冒用机制。
                    Conversation.user_id == user_db_id,
                )
            )).scalar_one_or_none()
            if existing is not None:
                # 复用会话时更新当前主域；Redis 仍按会话共享，图检查点按 agent:conversation 分开。
                existing.agent_id = agent_id
                existing.last_message_at = _utcnow()
                await session.commit()
                return existing.id

        conv = Conversation(
            user_id=user_db_id,
            agent_id=agent_id,
            title=None,
            source="api",
            # 本项目的 datetime 列都是 TIMESTAMP WITHOUT TIME ZONE（naive UTC），
            # 传 tz-aware 会让 asyncpg 报 "can't subtract offset-naive and offset-aware"
            last_message_at=_utcnow(),
        )
        session.add(conv)
        await session.commit()
        # 提交后从数据库读取自增主键/默认字段，确保返回真实 ID。
        await session.refresh(conv)
        return conv.id


async def _save_message(conversation_id: int, role: str, content: str) -> int:
    """Save a message and return its id."""
    async with async_session_factory() as session:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        session.add(msg)
        # Update last_message_at on conversation
        from sqlalchemy import update
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(last_message_at=_utcnow())
        )
        await session.commit()
        await session.refresh(msg)
        return msg.id


async def _save_trace(message_id: int, state: dict) -> None:
    """把 state["trace"] 摊平存进 message_traces + trace_function_calls。

    为什么在这里做映射而不是让 Agent 直接写库：
    Agent 只负责"产生 trace"，落库格式（列名、拆分表）属于接口层职责；
    这样离线脚本可直接读取 trace；但 Agent 内的审批和工具仍可能写库，不能笼统称为无副作用。
    """
    trace = state.get("trace", {})
    if not trace:
        return

    async with async_session_factory() as session:
        mt = MessageTrace(
            message_id=message_id,
            intent_agent=(trace.get("coordination") or {}).get("selected_agent"),
            intent_confidence=(trace.get("coordination") or {}).get("confidence"),
            retrieval_query=state.get("user_query"),
            generation_model=trace.get("model"),
            generation_prompt_tokens=trace.get("prompt_tokens"),
            generation_completion_tokens=trace.get("completion_tokens"),
            generation_duration_ms=trace.get("generation_ms"),
            retrieval_recalled_count=trace.get("retrieval_count"),
            retrieval_top_k=settings.retrieval_top_k,
            retrieval_duration_ms=trace.get("retrieval_ms"),
            rerank_model=settings.rerank_model,
            rerank_duration_ms=trace.get("rerank_ms"),
        )
        session.add(mt)
        await session.commit()
        await session.refresh(mt)

        # Tool call details
        for tc in trace.get("tool_call_results", []):
            session.add(TraceFunctionCall(
                trace_id=mt.id,
                tool_name=tc.get("tool", "unknown"),
                parameters=mask_sensitive(json.dumps(tc.get("arguments", {}), ensure_ascii=False)),
                is_success=tc.get("success", False),
                # 只截断结果摘要，未对这段和全部 audit detail 做脱敏；不能把 parameters 的脱敏扩大为全部日志。
                result_summary=str(tc.get("data", ""))[:256] if tc.get("data") else tc.get("error", ""),
                duration_ms=tc.get("duration_ms"),
            ))
        await session.commit()


async def _persist_exchange(
    conversation_id: int, user_msg: str, assistant_reply: str, state: dict,
) -> int | None:
    """写入一轮问答。DB 故障不影响答复，只降级为不落库。"""
    try:
        # 用户消息、机器人消息、trace 分别提交，整轮问答不是一个原子事务。
        await _save_message(conversation_id, "user", user_msg)
        bot_msg_id = await _save_message(conversation_id, "bot", assistant_reply)
        await _save_trace(bot_msg_id, state)
        return bot_msg_id
    except Exception as e:
        log.warning("exchange persist failed conv=%s: %s", conversation_id, e)
        # 这个降级只覆盖对话结束后的消息/trace 写入；起始建会话失败仍返回 503。
        return None


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """一轮对话：脱敏 → 安全 → 分诊 → 建/复用会话 → 跑图 → 落库 → 组装响应。

    中途在高危工具处会返回 pending_confirmation（图停在审批节点），
    此时 reply 是确认提示语，客户端需要调 /chat/confirm 继续。
    """
    # Step 1: Mask sensitive info + content safety check
    # 脱敏文本送模型；后面的 _respond 仍传 req.message 原文落 messages，不是全链路不保存原文。
    clean_msg = mask_sensitive(req.message)
    safety = content_safety.check(clean_msg)
    if not safety.passed:
        return ChatResponse(
            conversation_id=req.conversation_id,
            reply=content_safety.block_message(safety.severity, safety.action),
            intent="blocked",
        )

    # Step 2: Intent classification → route to agent
    # 发生在 Agent 执行 try 块外；分类返回结构异常并不一定转换成下面定义的 ProviderError 503。
    plan = await coordinator.plan(clean_msg)
    intent = plan.intent

    # 会话 ID 同时作为数据库主键、Redis memory key 和 LangGraph thread_id。
    # 先确定真实会话，再启动 Agent，避免每轮对话切换 ID 导致上下文断裂。
    try:
        # 当前用户身份来自客户端 user_id，没有认证依赖；投产前需要从可信令牌派生这个值。
        user_db_id = await _get_or_create_user(req.user_id)
        conversation_id = await _get_or_create_conversation(
            user_db_id, intent.target_agent, req.conversation_id,
        )
    except Exception as exc:
        # PostgreSQL 同时承载会话、审批和 LangGraph checkpoint。这里不能拿客户端
        # 临时 ID 继续运行，否则高危操作到审批节点时必然出现孤儿状态或整数溢出。
        log.exception("conversation persistence unavailable [CONVERSATION_STORE_ERROR]")
        raise HTTPException(
            status_code=503,
            detail="会话存储暂时不可用，请稍后重试",
        ) from exc
    # 会话号定了就把 trace 换成它，之后这一轮的日志都能按 conv 检索
    set_trace(f"conv-{conversation_id}")

    # Step 3: Run agent (LangGraph workflow)
    try:
        state = await coordinator.run(
            plan,
            conversation_id=conversation_id,
            user_query=clean_msg,
            user_id=req.user_id,
        )
    except ProviderError as exc:
        # 配置/凭据问题：细节留在日志，给用户的是一句能听懂的话——异常原文里可能带模型名和端点
        log.error("provider unavailable [MODEL_UNAVAILABLE]: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="智能客服暂时不可用，请稍后再试或转人工",
        ) from exc
    except Exception as exc:
        log.exception("agent run failed [AGENT_ERROR]")
        raise HTTPException(
            status_code=500,
            detail="处理这条消息时出错，请重试",
        ) from exc

    return await _respond(
        state,
        user_id=req.user_id,
        user_msg=req.message,
        agent_name=intent.target_agent,
        intent_name=intent.intent_name,
        conversation_id=conversation_id,
    )


@router.post("/chat/confirm", response_model=ChatResponse)
async def chat_confirm(req: ConfirmRequest) -> ChatResponse:
    """用户侧/前端审批入口。"""
    # 公共确认路由当前无 require_staff/所有权检查；后台审批鉴权不自动覆盖此入口。
    return await apply_approval(req)


async def apply_approval(req: ConfirmRequest) -> ChatResponse:
    """人工审批回调：从暂停点恢复执行。会话列表页/后台坐席共用这一条逻辑。

    当 saver 确实是 PostgreSQL 且 thread_id 一致时可跨进程续跑；内存降级不支持。
    结论先落到 approval_requests（含超时判定），再驱动图恢复——超时会让
    真正生效的结论从"批准"变成"自动拒绝"。
    """
    # 未知名字在 router 中默认落售前；此处没有使用枚举校验确认域是否合法。
    agent = agent_by_name(req.agent)
    resume = getattr(agent, "resume", None)
    if resume is None:
        raise HTTPException(status_code=400, detail=f"{req.agent} 不支持人工审批")

    # 先提交审批投影，再恢复图；二者之间进程故障会留下需要补偿的已决定记录。
    verdict = await decide(req.agent, req.conversation_id, req.approved, req.note, req.approver)
    if not verdict.get("ok"):
        raise HTTPException(status_code=409, detail=verdict.get("error", "审批请求不可处理"))

    try:
        # 实际从检查点恢复的是原待执行工具参数，不重新用批准请求生成业务参数。
        state = await resume(
            req.conversation_id,
            {"approved": verdict["approved"], "note": verdict["note"], "approver": verdict["approver"]},
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("approval resume failed [RESUME_ERROR]")
        raise HTTPException(
            status_code=500,
            detail="审批已记录，但恢复执行失败，请联系人工",
        ) from exc

    decision_text = (
        f"人工审批：{'批准' if verdict['approved'] else '拒绝'}（{verdict['approver']}）{verdict['note']}".strip()
    )
    return await _respond(
        state,
        # 审批轮次挂在原会话用户下（user_id 来自 checkpoint 里的 state），不新造用户
        user_id=state.get("user_id") or "anonymous",
        user_msg=decision_text,
        agent_name=req.agent,
        intent_name="after_sales",
        conversation_id=req.conversation_id,
    )


async def _respond(
    state: dict,
    *,
    user_id: str,
    user_msg: str,
    agent_name: str,
    intent_name: str,
    conversation_id: int,
) -> ChatResponse:
    """把 LangGraph 的返回统一成接口响应；interrupt（待审批）单独标注。"""
    # 节点在 interrupt() 处抛异常，它自己写的 pending_confirmation 不会落到 state，
    # 确认单只能从 Interrupt 对象的 value 里取。
    # 读框架中断返回值，区别“等待审批”与 final_response 为空的普通状态。
    interrupts = state.get("__interrupt__") or []
    # 当前只处理第一个中断，适合本图单审批节点；不是并行多中断通用聚合器。
    pending = getattr(interrupts[0], "value", None) if interrupts else None
    reply = state["final_response"] or (pending or {}).get("prompt", "")

    message_id = None
    if reply:
        # 仅有 reply 才落消息；审批提示也会被持久化为一次回复。
        message_id = await _persist_exchange(conversation_id, user_msg, reply, state)

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=reply,
        intent=intent_name,
        handoff=state.get("handoff", False),
        handoff_reason=state.get("handoff_reason", ""),
        # 响应带部分运行轨迹；状态中有的 decision/coordination 字段不一定全部映射到数据库列。
        trace=state.get("trace"),
        pending_confirmation=pending,
    )


@router.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    """Persist user feedback to database."""
    try:
        async with async_session_factory() as session:
            fb = Feedback(
                message_id=req.message_id,
                rating=req.rating,
                reason=req.reason,
            )
            session.add(fb)
            await session.commit()
    except Exception as e:
        log.warning("feedback persist failed conv=%s: %s", req.conversation_id, e)

    return {"status": "ok", "message_id": req.message_id, "rating": req.rating}


@router.get("/tools")
async def list_tools() -> dict:
    from app.agents import load_all_agents
    from app.tools.registry import tool_registry

    load_all_agents()
    tools: list[dict] = []
    seen: set[str] = set()
    for spec in tool_registry.list_all():
        if spec.definition.name not in seen:
            seen.add(spec.definition.name)
            tools.append({
                "name": spec.definition.name,
                "display_name": spec.definition.display_name,
                "description": spec.definition.description,
                "version": spec.definition.version,
                "agent_id": spec.agent_id,
            })
    return {"agent": "all", "tools": tools}
