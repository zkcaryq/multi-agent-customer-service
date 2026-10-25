"""审批队列读写。

LangGraph 的暂停态藏在 checkpoints 表里、不可查询，所以这里维护一张可查投影
approval_requests：Agent 在 interrupt 前登记，后台/前端在决定时回写，
超时由这里统一判定（读列表时顺带清扫，不依赖定时器）。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import select, text

from app.config import settings
from app.core.timeutil import utcnow_naive
from app.models.database import ApprovalRequest
from app.models.engine import async_session_factory


_now = utcnow_naive


async def ensure_pending(agent_name: str, conversation_id: int, user_id: str,
                         actions: list[dict[str, Any]]) -> None:
    """登记一条待审批；已存在就不动。

    节点在 resume 后会整段重跑，若无脑覆盖会把已批准的结论倒回 pending。
    """
    async with async_session_factory() as session:
        exists = (await session.execute(
            select(ApprovalRequest.id).where(
                ApprovalRequest.agent_name == agent_name,
                ApprovalRequest.conversation_id == conversation_id,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return
        session.add(ApprovalRequest(
            agent_name=agent_name,
            conversation_id=conversation_id,
            user_id=user_id,
            actions=json.dumps(actions, ensure_ascii=False),
            status="pending",
        ))
        await session.commit()


def _to_dict(row: ApprovalRequest) -> dict[str, Any]:
    return {
        "id": row.id,
        "agent": row.agent_name,
        "conversation_id": row.conversation_id,
        "user_id": row.user_id,
        "actions": json.loads(row.actions or "[]"),
        "status": row.status,
        "requested_at": row.requested_at.isoformat() if row.requested_at else None,
        "decided_at": row.decided_at.isoformat() if row.decided_at else None,
        "decided_by": row.decided_by,
        "decision_note": row.decision_note,
        "timeout_s": settings.approval_timeout_s,
    }


async def decide(agent_name: str, conversation_id: int, approved: bool,
                 note: str = "", approver: str = "operator") -> dict[str, Any]:
    """回写审批结论。超时则强制改为拒绝，并告诉调用方真正生效的结论。

    超时判定整条放在 SQL 里用数据库自己的 now() 比较：requested_at 是 PG 写的，
    而应用机与 PG 主机之间存在时钟漂移（实测约 1 分钟），用 Python 的钟去比会
    算出负数，导致"永不超时"。
    """
    expire_sql = text(
        "update approval_requests set status='expired', decided_at=now(), "
        " decided_by='system', decision_note='超时未审批，系统自动拒绝' "
        "where agent_name=:agent and conversation_id=:conv and status='pending' "
        "  and requested_at < now() - make_interval(secs => :secs)"
    )
    async with async_session_factory() as session:
        await session.execute(expire_sql, {
            "agent": agent_name, "conv": conversation_id, "secs": settings.approval_timeout_s,
        })
        # 超时是既成事实，必须先落库；否则后续分支提前 return 会让这条 UPDATE 回滚，
        # 请求永远停在 pending，下次还能被"批准"。
        await session.commit()

        row = (await session.execute(
            select(ApprovalRequest).where(
                ApprovalRequest.agent_name == agent_name,
                ApprovalRequest.conversation_id == conversation_id,
            )
        )).scalar_one_or_none()

        if row is None:
            return {"ok": False, "error": "没有待审批请求（可能已处理或会话号不对）"}
        if row.status == "expired":
            return {"ok": True, "approved": False, "note": row.decision_note or "超时未审批，系统自动拒绝",
                    "approver": row.decided_by or "system", "expired": True, "status": "expired",
                    "request": _to_dict(row)}
        if row.status != "pending":
            return {"ok": False, "error": f"该请求已处理（status={row.status}）"}

        row.status = "approved" if approved else "rejected"
        row.decided_at = _now()
        row.decided_by = approver
        row.decision_note = (note or "")[:512]
        await session.commit()
        return {"ok": True, "approved": approved, "note": note, "approver": approver,
                "expired": False, "status": row.status, "request": _to_dict(row)}


async def list_requests(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """列表前先清扫超时行，保证 pending 里不会留着已经过期的。"""
    await sweep_expired()
    async with async_session_factory() as session:
        stmt = select(ApprovalRequest).order_by(ApprovalRequest.id.desc()).limit(limit)
        if status:
            stmt = stmt.where(ApprovalRequest.status == status)
        rows = (await session.execute(stmt)).scalars().all()
    return [_to_dict(r) for r in rows]


async def sweep_expired() -> int:
    """把超时未处理的请求置为 expired（同样用数据库时钟判定）。

    没有引入 APScheduler/Celery：读队列时顺带清扫，效果等价且少一个组件。
    """
    async with async_session_factory() as session:
        result = await session.execute(text(
            "update approval_requests set status='expired', decided_at=now(), "
            " decided_by='system', decision_note='超时未审批，系统自动拒绝' "
            "where status='pending' and requested_at < now() - make_interval(secs => :secs)"
        ), {"secs": settings.approval_timeout_s})
        await session.commit()
        return result.rowcount or 0
