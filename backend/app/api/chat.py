"""Chat REST API endpoint — with PostgreSQL persistence."""

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

        result = await session.execute(
            select(User).where(User.external_id == user_id).limit(1)
        )
        user = result.scalar_one_or_none()
        if user:
            return user.id

        user = User(external_id=user_id, nickname=user_id, phone="")
        session.add(user)
        await session.commit()
        return user.id


async def _get_or_create_conversation(
    user_db_id: int, intent_agent: str, requested_id: int = 0,
) -> int:
    """复用当前用户已有会话；不存在时创建，保证记忆、消息和 checkpoint 同 ID。"""
    async with async_session_factory() as session:
        from sqlalchemy import select

        agent_id = (await session.execute(
            select(Agent.id).where(Agent.name == intent_agent).limit(1)
        )).scalar_one_or_none()

        if 0 < requested_id <= 2_147_483_647:
            existing = (await session.execute(
                select(Conversation).where(
                    Conversation.id == requested_id,
                    Conversation.user_id == user_db_id,
                )
            )).scalar_one_or_none()
            if existing is not None:
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
    """Persist trace data to message_traces."""
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
                result_summary=str(tc.get("data", ""))[:256] if tc.get("data") else tc.get("error", ""),
                duration_ms=tc.get("duration_ms"),
            ))
        await session.commit()


async def _persist_exchange(
    conversation_id: int, user_msg: str, assistant_reply: str, state: dict,
) -> int | None:
    """写入一轮问答。DB 故障不影响答复，只降级为不落库。"""
    try:
        await _save_message(conversation_id, "user", user_msg)
        bot_msg_id = await _save_message(conversation_id, "bot", assistant_reply)
        await _save_trace(bot_msg_id, state)
        return bot_msg_id
    except Exception as e:
        log.warning("exchange persist failed conv=%s: %s", conversation_id, e)
        return None


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Step 1: Mask sensitive info + content safety check
    clean_msg = mask_sensitive(req.message)
    safety = content_safety.check(clean_msg)
    if not safety.passed:
        return ChatResponse(
            conversation_id=req.conversation_id,
            reply=content_safety.block_message(safety.severity, safety.action),
            intent="blocked",
        )

    # Step 2: Intent classification → route to agent
    plan = await coordinator.plan(clean_msg)
    intent = plan.intent

    # 会话 ID 同时作为数据库主键、Redis memory key 和 LangGraph thread_id。
    # 先确定真实会话，再启动 Agent，避免每轮对话切换 ID 导致上下文断裂。
    try:
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
    return await apply_approval(req)


async def apply_approval(req: ConfirmRequest) -> ChatResponse:
    """人工审批回调：从暂停点恢复执行。会话列表页/后台坐席共用这一条逻辑。

    状态存在 PostgreSQL 的 checkpoint 里，所以重启服务、换一个实例都能续跑。
    结论先落到 approval_requests（含超时判定），再驱动图恢复——超时会让
    真正生效的结论从"批准"变成"自动拒绝"。
    """
    agent = agent_by_name(req.agent)
    resume = getattr(agent, "resume", None)
    if resume is None:
        raise HTTPException(status_code=400, detail=f"{req.agent} 不支持人工审批")

    verdict = await decide(req.agent, req.conversation_id, req.approved, req.note, req.approver)
    if not verdict.get("ok"):
        raise HTTPException(status_code=409, detail=verdict.get("error", "审批请求不可处理"))

    try:
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
    interrupts = state.get("__interrupt__") or []
    pending = getattr(interrupts[0], "value", None) if interrupts else None
    reply = state["final_response"] or (pending or {}).get("prompt", "")

    message_id = None
    if reply:
        message_id = await _persist_exchange(conversation_id, user_msg, reply, state)

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=reply,
        intent=intent_name,
        handoff=state.get("handoff", False),
        handoff_reason=state.get("handoff_reason", ""),
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
