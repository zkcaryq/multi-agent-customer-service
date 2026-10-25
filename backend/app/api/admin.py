"""Admin panel REST API — conversations, feedback, knowledge, agents, tools."""

from __future__ import annotations

import asyncio
import os
import shutil
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select, func, desc

from app.core.auth import (
    DECIDER_ROLES,
    ROLE_ADMIN,
    Staff,
    require_roles,
    require_staff,
)

from app.models.database import (
    Agent,
    AgentRetrievalParam,
    AuditLog,
    Conversation,
    Feedback,
    KnowledgeChunk,
    KnowledgeDocument,
    Message,
    MessageTrace,
    TraceFunctionCall,
    ApprovalRequest,
    Tool,
    User,
)
from app.models.engine import async_session_factory

# 后台所有接口默认要求已登录的 staff；写操作再按角色细分
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_staff)],
)


def _utcnow() -> datetime:
    """本库的 datetime 列是 naive TIMESTAMP，统一写/比 naive UTC。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# Upload destination — project root / data/
UPLOAD_DIR = str(
    __import__("pathlib").Path(__file__).parent.parent.parent.parent / "data"
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class AgentUpdate(BaseModel):
    model: str | None = None
    system_prompt: str | None = None
    top_k: int | None = None
    similarity_threshold: float | None = None
    bm25_weight: float | None = None


# ── Dashboard stats ───────────────────────────────────────────────────────────

@router.get("/dashboard")
async def dashboard_stats():
    async with async_session_factory() as session:
        now = _utcnow()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        total_conv = (await session.execute(select(func.count(Conversation.id)))).scalar() or 0
        today_conv = (await session.execute(
            select(func.count(Conversation.id)).where(Conversation.created_at >= today)
        )).scalar() or 0
        active_conv = (await session.execute(
            select(func.count(Conversation.id)).where(Conversation.status == "active")
        )).scalar() or 0
        handoff_conv = (await session.execute(
            select(func.count(Conversation.id)).where(Conversation.status == "handoff")
        )).scalar() or 0
        total_msgs = (await session.execute(select(func.count(Message.id)))).scalar() or 0
        helpful = (await session.execute(
            select(func.count(Feedback.id)).where(Feedback.rating == "helpful")
        )).scalar() or 0
        unhelpful = (await session.execute(
            select(func.count(Feedback.id)).where(Feedback.rating == "unhelpful")
        )).scalar() or 0
        total_fb = helpful + unhelpful
        satisfaction = round(helpful / total_fb * 100, 1) if total_fb > 0 else None

        # Agent distribution
        agent_dist = (await session.execute(
            select(Conversation.agent_id, func.count(Conversation.id))
            .where(Conversation.agent_id.isnot(None))
            .group_by(Conversation.agent_id)
        )).all()
        agent_rows = (await session.execute(
            select(Agent.id, Agent.display_name)
        )).all()
        agent_map = {agent_id: display_name for agent_id, display_name in agent_rows}
        agents = [
            {"name": agent_map.get(aid, f"Agent {aid}"), "count": cnt}
            for aid, cnt in agent_dist
        ]

        return {
            "total_conversations": total_conv,
            "today_conversations": today_conv,
            "active_conversations": active_conv,
            "total_messages": total_msgs,
            "handoff_count": handoff_conv,
            "satisfaction": satisfaction,
            "total_feedback": total_fb,
            "agent_distribution": agents,
        }


# ── Conversations ─────────────────────────────────────────────────────────────

@router.get("/conversations")
async def list_conversations(page: int = 1, page_size: int = 20, status: str = ""):
    async with async_session_factory() as session:
        q = select(Conversation).order_by(desc(Conversation.last_message_at))
        if status:
            q = q.where(Conversation.status == status)
        offset = (page - 1) * page_size
        result = await session.execute(q.limit(page_size).offset(offset))
        convs = result.scalars().all()

        agent_rows = (await session.execute(
            select(Agent.id, Agent.display_name)
        )).all()
        agent_names = {agent_id: display_name for agent_id, display_name in agent_rows}

        rows: list[dict] = []
        for c in convs:
            msg_count = (await session.execute(
                select(func.count(Message.id)).where(Message.conversation_id == c.id)
            )).scalar() or 0
            rows.append({
                "id": c.id,
                "user_id": c.user_id,
                "agent_name": agent_names.get(c.agent_id or 1, "售前客服"),
                "status": c.status,
                "title": c.title or f"会话 #{c.id}",
                "message_count": msg_count,
                "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            })
        return {"rows": rows, "page": page, "page_size": page_size}


@router.get("/conversations/{conv_id}/messages")
async def conversation_messages(conv_id: int):
    async with async_session_factory() as session:
        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.created_at)
        )
        msgs = result.scalars().all()
        return {
            "conversation_id": conv_id,
            "messages": [
                {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
                for m in msgs
            ],
        }


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.get("/feedback")
async def list_feedback(page: int = 1, page_size: int = 20, rating: str = ""):
    async with async_session_factory() as session:
        q = select(Feedback).order_by(desc(Feedback.created_at))
        if rating:
            q = q.where(Feedback.rating == rating)
        offset = (page - 1) * page_size
        result = await session.execute(q.limit(page_size).offset(offset))
        items = result.scalars().all()

        return {
            "rows": [
                {
                    "id": f.id,
                    "message_id": f.message_id,
                    "rating": f.rating,
                    "reason": f.reason,
                    "review_status": f.review_status,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in items
            ],
            "page": page, "page_size": page_size,
        }


@router.get("/feedback/stats")
async def feedback_stats():
    async with async_session_factory() as session:
        total = (await session.execute(select(func.count(Feedback.id)))).scalar() or 0
        helpful = (await session.execute(
            select(func.count(Feedback.id)).where(Feedback.rating == "helpful")
        )).scalar() or 0
        unhelpful = (await session.execute(
            select(func.count(Feedback.id)).where(Feedback.rating == "unhelpful")
        )).scalar() or 0
        pending = (await session.execute(
            select(func.count(Feedback.id)).where(Feedback.review_status == "pending")
        )).scalar() or 0

        rate = round(helpful / total * 100, 1) if total > 0 else None

        return {
            "total": total,
            "helpful": helpful,
            "unhelpful": unhelpful,
            "satisfaction_rate": rate,
            "pending_review": pending,
        }


# ── Knowledge ─────────────────────────────────────────────────────────────────

@router.get("/knowledge")
async def list_knowledge(page: int = 1, page_size: int = 20):
    async with async_session_factory() as session:
        q = select(KnowledgeDocument).order_by(desc(KnowledgeDocument.created_at))
        offset = (page - 1) * page_size
        result = await session.execute(q.limit(page_size).offset(offset))
        docs = result.scalars().all()

        rows: list[dict] = []
        for d in docs:
            chunk_count = (await session.execute(
                select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.document_id == d.id)
            )).scalar() or 0
            total_retrievals = (await session.execute(
                select(func.coalesce(func.sum(KnowledgeChunk.retrieval_count), 0))
                .where(KnowledgeChunk.document_id == d.id)
            )).scalar() or 0
            rows.append({
                "id": d.id,
                "title": d.title,
                "file_type": d.file_type,
                "split_method": d.split_method,
                "chunk_count": chunk_count,
                "total_retrievals": total_retrievals,
                "status": d.status,
                "version": d.version,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            })
        return {"rows": rows, "page": page, "page_size": page_size}


@router.get("/knowledge/stats")
async def knowledge_stats():
    async with async_session_factory() as session:
        total_docs = (await session.execute(
            select(func.count(KnowledgeDocument.id))
        )).scalar() or 0
        total_chunks = (await session.execute(
            select(func.count(KnowledgeChunk.id))
        )).scalar() or 0
        total_retrievals = (await session.execute(
            select(func.coalesce(func.sum(KnowledgeChunk.retrieval_count), 0))
        )).scalar() or 0
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_retrievals": total_retrievals,
        }


# ── Agents ────────────────────────────────────────────────────────────────────

@router.get("/agents")
async def list_agents():
    from app.agents.after_sales.prompt import AFTER_SALES_SYSTEM_PROMPT
    from app.agents.complaint.prompt import COMPLAINT_SYSTEM_PROMPT
    from app.agents.during_sales.prompt import DURING_SALES_SYSTEM_PROMPT
    from app.agents.pre_sales.prompt import PRE_SALES_SYSTEM_PROMPT
    from app.config import settings

    prompts = {
        "pre_sales": PRE_SALES_SYSTEM_PROMPT,
        "during_sales": DURING_SALES_SYSTEM_PROMPT,
        "after_sales": AFTER_SALES_SYSTEM_PROMPT,
        "complaint": COMPLAINT_SYSTEM_PROMPT,
    }
    async with async_session_factory() as session:
        result = await session.execute(select(Agent).order_by(Agent.id))
        agents = result.scalars().all()

        rows: list[dict] = []
        for a in agents:
            params_result = await session.execute(
                select(AgentRetrievalParam).where(AgentRetrievalParam.agent_id == a.id)
            )
            params = {p.param_key: p.param_value for p in params_result.scalars().all()}
            rows.append({
                "id": a.id,
                "name": a.name,
                "display_name": a.display_name,
                "model": settings.model_for(a.name),
                "system_prompt": prompts.get(a.name, ""),
                "status": a.status,
                "retrieval_params": {
                    "top_k": str(settings.retrieval_top_k),
                    "similarity_threshold": str(settings.similarity_threshold),
                    "bm25_weight": str(settings.bm25_weight),
                    **{k: v for k, v in params.items() if k not in {
                        "top_k", "similarity_threshold", "bm25_weight"
                    }},
                },
                "config_source": "environment_and_code",
            })
        return {"agents": rows}


@router.put("/agents/{agent_id}")
async def update_agent(
    agent_id: int,
    body: AgentUpdate,
    staff: Staff = Depends(require_roles(ROLE_ADMIN)),
):
    raise HTTPException(
        409,
        "运行配置由 .env 与 Agent Prompt 代码管理；请修改配置并重启后端",
    )


# ── Tools ─────────────────────────────────────────────────────────────────────

@router.get("/tools")
async def list_admin_tools():
    from app.agents import load_all_agents
    from app.tools.registry import tool_registry

    load_all_agents()
    specs = sorted(tool_registry.list_all(), key=lambda item: item.definition.name)
    return {
        "tools": [
            {
                "id": index,
                "name": spec.definition.name,
                "display_name": spec.definition.display_name,
                "description": spec.definition.description,
                "http_method": "FUNCTION",
                "endpoint_url": "",
                "version": spec.definition.version,
                "status": "active",
                "timeout_ms": spec.definition.timeout_ms,
                "retry_count": spec.definition.retry_count,
                "agent_id": spec.agent_id,
            }
            for index, spec in enumerate(specs, start=1)
        ]
    }


# ── Knowledge upload ──────────────────────────────────────────────────────────

@router.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    """Upload a document, save it, and run the ingestion pipeline."""
    if not file.filename:
        raise HTTPException(400, "No file selected")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".md", ".faq", ".txt", ".pdf", ".docx", ".xlsx"):
        raise HTTPException(400, f"Unsupported format: {ext}. Supports .md, .faq, .txt, .pdf, .docx, .xlsx")

    # Detect split method
    SPLIT_MAP = {
        "md": "structure", "faq": "qa_boundary", "txt": "fixed_size",
        "pdf": "semantic", "docx": "semantic", "xlsx": "row_table",
    }
    split_method = SPLIT_MAP.get(ext.lstrip("."), "fixed_size")
    file_type = ext.lstrip(".")

    # Save file to data/ directory
    save_dir = os.path.join(UPLOAD_DIR, file_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run ingestion (includes DB persistence via _persist_to_db)
    try:
        from app.rag.ingestion import ingest_file
        # ingest_file 包含同步解析/模型调用，并在内部用 asyncio.run 写 PostgreSQL。
        # 放到工作线程，既不阻塞 FastAPI 事件循环，也避免嵌套事件循环导致 DB 落库失败。
        stats = await asyncio.to_thread(ingest_file, save_path)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")

    return {
        "status": "ok",
        "filename": file.filename,
        "document_id": stats.get("document_id"),
        "chunks": stats.get("chunks", 0),
        "split_method": split_method,
        "file_type": file_type,
        "elapsed_ms": stats.get("elapsed_ms", 0),
    }


# ── 审批队列 / 审计留痕 / 工单 / 离线评测（运营可见闭环）────────────────────

class ApprovalDecision(BaseModel):
    approved: bool
    note: str = ""


class TicketResolve(BaseModel):
    note: str = ""
    status: str = "resolved"


def _num(value):
    """Numeric 列取出来是 Decimal，直接给前端会变成字符串。"""
    return None if value is None else float(value)


@router.get("/approvals")
async def list_approvals(status: str | None = None, limit: int = 50):
    """待审批队列。读取时顺带把超时未处理的置为 expired（无需额外定时器）。"""
    from app.core.approvals import list_requests

    return {"items": await list_requests(status, min(max(limit, 1), 200))}


@router.post("/approvals/{request_id}/decide")
async def decide_approval(
    request_id: int,
    body: ApprovalDecision,
    staff: Staff = Depends(require_roles(*DECIDER_ROLES)),
):
    """坐席在后台审批：等价于用户侧 /chat/confirm，共用同一套超时与恢复逻辑。"""
    from sqlalchemy import select as _select

    from app.api.chat import apply_approval
    from app.models.database import ApprovalRequest
    from app.schemas.chat import ConfirmRequest

    async with async_session_factory() as session:
        row = (await session.execute(
            _select(ApprovalRequest).where(ApprovalRequest.id == request_id)
        )).scalar_one_or_none()

    if row is None:
        raise HTTPException(404, f"审批请求 {request_id} 不存在")

    return await apply_approval(ConfirmRequest(
        conversation_id=row.conversation_id,
        agent=row.agent_name,
        approved=body.approved,
        note=body.note,
        # 审批人取自令牌，不接受前端自报，否则审计日志可被伪造
        approver=staff.username,
    ))


@router.get("/audit-logs")
async def list_audit_logs(action: str | None = None, limit: int = 50):
    """敏感动作留痕：审批结论、退款受理、投诉建单都在这一张表。"""
    from sqlalchemy import select as _select

    from app.models.database import AuditLog

    async with async_session_factory() as session:
        stmt = _select(AuditLog).order_by(AuditLog.id.desc()).limit(min(max(limit, 1), 200))
        if action:
            stmt = stmt.where(AuditLog.action == action)
        rows = (await session.execute(stmt)).scalars().all()

    return {"items": [{
        "id": r.id,
        "action": r.action,
        "entity_type": r.entity_type,
        "entity_id": r.entity_id,
        "summary": r.summary,
        "detail": r.detail,
        "admin_user_id": r.admin_user_id,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in rows]}


@router.get("/monitor/summary")
async def monitor_summary(hours: int = 24):
    """近一段时间的链路健康摘要，直接基于 Trace/工具调用/审批表聚合。"""
    hours = min(max(hours, 1), 168)
    since = _utcnow() - timedelta(hours=hours)
    async with async_session_factory() as session:
        trace_filter = MessageTrace.created_at >= since
        total_traces = (await session.execute(
            select(func.count(MessageTrace.id)).where(trace_filter)
        )).scalar() or 0
        avg_generation = (await session.execute(
            select(func.avg(MessageTrace.generation_duration_ms)).where(trace_filter)
        )).scalar()
        avg_retrieval = (await session.execute(
            select(func.avg(MessageTrace.retrieval_duration_ms)).where(trace_filter)
        )).scalar()
        token_row = (await session.execute(
            select(
                func.sum(MessageTrace.generation_prompt_tokens),
                func.sum(MessageTrace.generation_completion_tokens),
            ).where(trace_filter)
        )).one()

        tool_base = (
            select(func.count(TraceFunctionCall.id))
            .join(MessageTrace, MessageTrace.id == TraceFunctionCall.trace_id)
            .where(trace_filter)
        )
        tool_calls = (await session.execute(tool_base)).scalar() or 0
        tool_failures = (await session.execute(
            tool_base.where(TraceFunctionCall.is_success.is_(False))
        )).scalar() or 0
        pending_approvals = (await session.execute(
            select(func.count(ApprovalRequest.id)).where(ApprovalRequest.status == "pending")
        )).scalar() or 0

    return {
        "window_hours": hours,
        "trace_count": total_traces,
        "avg_generation_ms": round(_num(avg_generation) or 0, 1),
        "avg_retrieval_ms": round(_num(avg_retrieval) or 0, 1),
        "prompt_tokens": int(token_row[0] or 0),
        "completion_tokens": int(token_row[1] or 0),
        "tool_calls": tool_calls,
        "tool_failures": tool_failures,
        "tool_success_rate": round((tool_calls - tool_failures) / tool_calls * 100, 1) if tool_calls else None,
        "pending_approvals": pending_approvals,
    }


@router.get("/monitor/traces")
async def monitor_traces(limit: int = 50):
    """最近的消息级链路及工具调用明细，供管理端回放排障。"""
    limit = min(max(limit, 1), 200)
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(MessageTrace, Message, Conversation)
            .join(Message, Message.id == MessageTrace.message_id)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .order_by(MessageTrace.id.desc())
            .limit(limit)
        )).all()

        items = []
        for trace, message, conversation in rows:
            calls = (await session.execute(
                select(TraceFunctionCall)
                .where(TraceFunctionCall.trace_id == trace.id)
                .order_by(TraceFunctionCall.id)
            )).scalars().all()
            items.append({
                "trace_id": trace.id,
                "conversation_id": conversation.id,
                "message_id": message.id,
                "agent": trace.intent_agent,
                "intent_confidence": _num(trace.intent_confidence),
                "model": trace.generation_model,
                "retrieval_count": trace.retrieval_recalled_count,
                "retrieval_ms": trace.retrieval_duration_ms,
                "rerank_ms": trace.rerank_duration_ms,
                "generation_ms": trace.generation_duration_ms,
                "prompt_tokens": trace.generation_prompt_tokens,
                "completion_tokens": trace.generation_completion_tokens,
                "created_at": trace.created_at.isoformat() if trace.created_at else None,
                "tool_calls": [{
                    "name": call.tool_name,
                    "success": call.is_success,
                    "duration_ms": call.duration_ms,
                    "parameters": call.parameters,
                    "result": call.result_summary,
                } for call in calls],
            })

    return {"items": items}


@router.get("/tickets")
async def list_tickets(status: str | None = None, limit: int = 50):
    """人工工单（human_handoffs）。投诉 Agent 建单、Agent 判定的转人工都落在这里。"""
    from sqlalchemy import select as _select

    from app.models.database import HumanHandoff

    async with async_session_factory() as session:
        stmt = (
            _select(HumanHandoff, Conversation, User)
            .join(Conversation, Conversation.id == HumanHandoff.conversation_id)
            .join(User, User.id == Conversation.user_id)
            .order_by(HumanHandoff.id.desc())
            .limit(min(max(limit, 1), 200))
        )
        if status:
            stmt = stmt.where(HumanHandoff.status == status)
        rows = (await session.execute(stmt)).all()

    return {"items": [{
        "id": h.id,
        "ticket_id": f"TS-{h.id:06d}",
        "category": h.reason,
        "summary": h.summary,
        "from_agent": h.intent_agent,
        "status": h.status,
        "customer": user.external_id,
        "conversation_id": h.conversation_id,
        "created_at": h.created_at.isoformat() if h.created_at else None,
        "resolved_at": h.resolved_at.isoformat() if h.resolved_at else None,
        "resolution_note": h.resolution_note,
    } for h, _conv, user in rows]}


@router.post("/tickets/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: int,
    body: TicketResolve,
    staff: Staff = Depends(require_roles(*DECIDER_ROLES)),
):
    """坐席处理完工单后回填结论。"""
    from sqlalchemy import select as _select

    from app.models.database import HumanHandoff

    async with async_session_factory() as session:
        ticket = (await session.execute(
            _select(HumanHandoff).where(HumanHandoff.id == ticket_id)
        )).scalar_one_or_none()
        if ticket is None:
            raise HTTPException(404, f"工单 {ticket_id} 不存在")

        ticket.status = body.status
        ticket.resolution_note = body.note
        ticket.resolved_at = _utcnow()
        session.add(AuditLog(
            action="ticket_resolved",
            entity_type="human_handoff",
            entity_id=ticket.id,
            summary=f"工单处理完成：{body.status}",
            detail=f"{staff.username}: {body.note}",
        ))
        await session.commit()

    return {"status": "ok", "ticket_id": ticket_id, "ticket_status": body.status}


@router.get("/evaluations")
async def list_evaluations(limit: int = 20):
    """离线评测批次列表：每次 run 的指标都来自 evaluation_runs。"""
    from sqlalchemy import select as _select

    from app.models.database import EvaluationRun

    async with async_session_factory() as session:
        rows = (await session.execute(
            _select(EvaluationRun).order_by(EvaluationRun.id.desc()).limit(min(max(limit, 1), 100))
        )).scalars().all()

    return {"items": [{
        "id": r.id,
        "name": r.name,
        "test_set": r.test_set_name,
        "total_cases": r.total_cases,
        "completed_cases": r.completed_cases,
        "intent_accuracy": _num(r.intent_accuracy),
        "tool_call_accuracy": _num(r.tool_call_accuracy),
        "retrieval_hit_rate": _num(r.retrieval_hit_rate),
        "retrieval_recall": _num(r.retrieval_recall),
        "retrieval_precision": _num(r.retrieval_precision),
        "retrieval_mrr": _num(r.retrieval_mrr),
        "avg_total_duration_ms": r.avg_total_duration_ms,
        "status": r.status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in rows]}


@router.get("/evaluations/{run_id}")
async def evaluation_detail(run_id: int, only_wrong: bool = False, limit: int = 100):
    """单批评测的逐条结果，only_wrong=true 时只看分错的用例（badcase 复盘）。"""
    from sqlalchemy import select as _select

    from app.models.database import EvaluationResult, EvaluationRun

    async with async_session_factory() as session:
        run = (await session.execute(
            _select(EvaluationRun).where(EvaluationRun.id == run_id)
        )).scalar_one_or_none()
        if run is None:
            raise HTTPException(404, f"评测批次 {run_id} 不存在")

        stmt = (
            _select(EvaluationResult)
            .where(EvaluationResult.run_id == run_id)
            .order_by(EvaluationResult.case_index)
            .limit(min(max(limit, 1), 500))
        )
        if only_wrong:
            stmt = stmt.where(EvaluationResult.intent_correct.is_(False))
        cases = (await session.execute(stmt)).scalars().all()

    return {
        "run": {"id": run.id, "name": run.name, "test_set": run.test_set_name,
                "intent_accuracy": _num(run.intent_accuracy),
                "retrieval_hit_rate": _num(run.retrieval_hit_rate),
                "retrieval_mrr": _num(run.retrieval_mrr)},
        "cases": [{
            "case_index": c.case_index,
            "question": c.question,
            "expected_intent": c.expected_intent,
            "actual_intent": c.actual_intent,
            "intent_correct": c.intent_correct,
            "expected_tool": c.expected_tool,
            "actual_tool": c.actual_tool,
            "tool_correct": c.tool_correct,
            "retrieval_hit": c.retrieval_hit,
            "retrieved": c.actual_answer,
        } for c in cases],
    }
