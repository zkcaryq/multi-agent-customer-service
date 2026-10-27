"""审批队列读写。

LangGraph 暂停态可通过图状态 API 查询，但不便直接作为后台审批列表，所以这里维护一张可查投影
approval_requests：Agent 在 interrupt 前登记，后台/前端在决定时回写，
超时由这里统一判定（读列表时顺带清扫，不依赖定时器）。

【这一层的职责边界】
- 不负责挂起/恢复（那是 LangGraph 的 interrupt/Command）；
- 只负责"人看得见、批得动、超时会自己失效"这三件事。
所以它是一张普通业务表，跟图框架解耦：后台页面、自检脚本都能直接查。
"""

from __future__ import annotations

# json 存 actions 列；select 走 ORM、text 走裸 SQL——超时判定只能写成裸 SQL 才能用上数据库的 now()
import json
from datetime import datetime   # 本模块不直接构造 datetime，时间一律来自 _now 或数据库 now()
from typing import Any

from sqlalchemy import select, text

from app.config import settings
from app.core.timeutil import utcnow_naive
from app.models.database import ApprovalRequest
from app.models.engine import async_session_factory


_now = utcnow_naive   # 全项目唯一的 naive UTC 来源（见 core/timeutil.py）


async def ensure_pending(agent_name: str, conversation_id: int, user_id: str,
                         actions: list[dict[str, Any]]) -> None:
    """登记一条待审批；已存在就不动。

    节点在 resume 后会整段重跑，若无脑覆盖会把已批准的结论倒回 pending。

    "存在即跳过"是这里唯一重要的事：
    interrupt 前写、resume 时节点重放又执行一次 → 必须幂等。
    判断条件是 (agent_name, conversation_id)，与表上的唯一约束一致；
    没有按 status 过滤，是为了让"已处理过"的会话不会被重新打开成 pending
    （代价是同一会话第二次高危操作会拿到 409，这是已知限制）。
    """
    # 独立数据库会话；退出时未提交事务会回滚，但不包含 LangGraph 检查点或外部业务系统事务。
    async with async_session_factory() as session:
        # 先查后插不是并发原子 upsert；唯一约束阻止重复行，但竞争失败仍会抛 IntegrityError。
        exists = (await session.execute(
            # 只取 id 不取整行：这里只要"有没有"，不必把 actions 的大 JSON 读进内存
            select(ApprovalRequest.id).where(
                # 这两列就是唯一约束 uq_approval_thread，也是 LangGraph thread_id 的两段
                ApprovalRequest.agent_name == agent_name,
                ApprovalRequest.conversation_id == conversation_id,
            )
        )).scalar_one_or_none()   # 唯一约束保证至多一行；真出现多行会抛错而不是当成"已存在"
        # 不区分 pending/approved/rejected，因此同域同会话第二次新高危操作也会被已有记录挡住。
        if exists is not None:
            return
        session.add(ApprovalRequest(
            # agent_name 由调用方传 self.AGENT_NAME，审批接口要靠它找回同一个 agent 的图
            agent_name=agent_name,
            # 会话号同时是 checkpoint thread 的一部分，三者对不上就恢复不到挂起点
            conversation_id=conversation_id,
            # 落的是客户端自报 user_id，只作展示与追溯，不参与审批权限判断
            user_id=user_id,
            # actions 用 JSON 文本存：结构随工具而异（退款是 order_id/reason，
            # 改地址是 order_id/new_address），建列反而不灵活
            actions=json.dumps(actions, ensure_ascii=False),
            # 数据库 requested_at 默认值记录申请时间；前端 payload 的 requested_at 不是超时判据。
            status="pending",
        ))
        # 必须现在就提交：下一句就是 interrupt()，它会抛 GraphInterrupt 结束本次执行，
        # 不提交的话后台看不到待审批单，会话也永远停在"图里挂了、表里没有"
        await session.commit()


def _to_dict(row: ApprovalRequest) -> dict[str, Any]:
    """ORM 行 → 接口响应。时间统一转 ISO 字符串，前端不用处理 datetime 类型。"""
    return {
        "id": row.id,
        # 字段名与 ConfirmRequest 对齐（agent/conversation_id），前端可把这份直接回填成请求体
        "agent": row.agent_name,
        "conversation_id": row.conversation_id,
        "user_id": row.user_id,
        # 恢复当时待审批参数快照；这里只解码，不核实订单金额或当前业务状态。
        # actions 列可能被手工写入或工具结构变更影响，坏 JSON 会在这里抛 ValueError 并让整页列表 500
        "actions": json.loads(row.actions or "[]"),
        "status": row.status,   # pending/approved/rejected/expired 四态，前端按它决定按钮是否可点
        # naive UTC 直接 isoformat，不带 +00:00 后缀；前端按本地时间解析会整体偏 8 小时
        "requested_at": row.requested_at.isoformat() if row.requested_at else None,
        "decided_at": row.decided_at.isoformat() if row.decided_at else None,
        "decided_by": row.decided_by,   # expired 行是 'system'，其余取决于入口（后台取令牌名，公共入口可自报）
        "decision_note": row.decision_note,
        # 让前端能展示"还剩多久过期"，不用它自己算配置
        "timeout_s": settings.approval_timeout_s,
    }


async def decide(agent_name: str, conversation_id: int, approved: bool,
                 note: str = "", approver: str = "operator") -> dict[str, Any]:
    """回写审批结论。超时则强制改为拒绝，并告诉调用方真正生效的结论。

    超时判定整条放在 SQL 里用数据库自己的 now() 比较：requested_at 是 PG 写的，
    而应用机与 PG 主机之间存在时钟漂移（实测约 1 分钟），用 Python 的钟去比会
    算出负数，导致"永不超时"。
    """
    # ① 先按数据库时钟把超时的置为 expired。
    #    用 make_interval(secs => :secs) 而不是拼字符串，避免注入且类型明确。
    expire_sql = text(
        "update approval_requests set status='expired', decided_at=now(), "
        " decided_by='system', decision_note='超时未审批，系统自动拒绝' "
        "where agent_name=:agent and conversation_id=:conv and status='pending' "
        "  and requested_at < now() - make_interval(secs => :secs)"
    )
    # 独立数据库会话；退出时未提交事务会回滚，但不包含 LangGraph 检查点或外部业务系统事务。
    async with async_session_factory() as session:
        # 三个都是绑定参数（不是拼字符串）：secs 取 APPROVAL_TIMEOUT_S，默认 900 秒
        await session.execute(expire_sql, {
            "agent": agent_name, "conv": conversation_id, "secs": settings.approval_timeout_s,
        })
        # 影响 0 行是正常情况（还没超时或本来就不是 pending），这里不检查 rowcount
        # 超时是既成事实，必须先落库；否则后续分支提前 return 会让这条 UPDATE 回滚，
        # 请求永远停在 pending，下次还能被"批准"。
        await session.commit()

        # ② 读回最终状态，按状态分三种出口
        # UPDATE 已提交后再读，所以读到的一定是"过期清扫之后"的状态
        row = (await session.execute(
            # 这里要整行：status/decided_* 还要回给调用方，与 ensure_pending 只取 id 不同
            select(ApprovalRequest).where(
                # 幂等键两列，与唯一约束一致；没有按 status 过滤，所以"已处理"也能查到
                ApprovalRequest.agent_name == agent_name,
                ApprovalRequest.conversation_id == conversation_id,
            )
        )).scalar_one_or_none()

        if row is None:
            # 出口①：这条键从来没登记过——多半是 agent 名或会话号不对，调用方转 409，不会恢复图
            return {"ok": False, "error": "没有待审批请求（可能已处理或会话号不对）"}
        # expired 分支仍允许返回 ok=True 的拒绝结论；重复确认过期请求不具有普通“已处理”分支的 409 保护。
        if row.status == "expired":
            # 超时优先于本次操作：哪怕现在点了"批准"，也按拒绝执行
            return {"ok": True, "approved": False, "note": row.decision_note or "超时未审批，系统自动拒绝",
                    "approver": row.decided_by or "system", "expired": True, "status": "expired",
                    "request": _to_dict(row)}
        if row.status != "pending":
            # 已批准/已拒绝：重复点击不应该第二次执行工具，返回冲突让调用方转 409
            return {"ok": False, "error": f"该请求已处理（status={row.status}）"}

        # ③ 正常回写
        # 当前是读后写，没有 SELECT FOR UPDATE/CAS；两个审批请求可能同时读到 pending。
        row.status = "approved" if approved else "rejected"
        # 正常决定时间取应用 UTC；过期 SQL 用数据库 now()，两种来源并存。
        row.decided_at = _now()
        row.decided_by = approver
        row.decision_note = (note or "")[:512]   # 与列宽一致，超长截断而不是让 DB 报错
        await session.commit()
        return {"ok": True, "approved": approved, "note": note, "approver": approver,
                "expired": False, "status": row.status, "request": _to_dict(row)}


async def list_requests(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """列表前先清扫超时行，保证 pending 里不会留着已经过期的。

    读时清扫 = 无需定时任务。后台每次刷新列表都会把过期的 pending 变成 expired，
    代价是"没人看列表时不会自动过期"——但真正的超时判定在 decide() 里，
    这里只是让列表好看。
    """
    # 仅清扫审批投影表；没有调用 agent.resume，不会主动向离线用户推送拒绝回复。
    await sweep_expired()
    # 独立数据库会话；退出时未提交事务会回滚，但不包含 LangGraph 检查点或外部业务系统事务。
    async with async_session_factory() as session:
        stmt = select(ApprovalRequest).order_by(ApprovalRequest.id.desc()).limit(limit)
        # 先设置 limit 再 where 是 SQLAlchemy 构造语句，最终 WHERE 仍在数据库 LIMIT 前执行。
        if status:
            stmt = stmt.where(ApprovalRequest.status == status)
        rows = (await session.execute(stmt)).scalars().all()
    return [_to_dict(r) for r in rows]


async def sweep_expired() -> int:
    """把超时未处理的请求置为 expired（同样用数据库时钟判定）。

    没有引入 APScheduler/Celery：读队列时顺带清扫，减少一个组件；与准点定时清扫、自动恢复图并不等价。
    返回被清扫的行数，便于自检断言。
    """
    # 独立数据库会话；退出时未提交事务会回滚，但不包含 LangGraph 检查点或外部业务系统事务。
    async with async_session_factory() as session:
        result = await session.execute(text(
            "update approval_requests set status='expired', decided_at=now(), "
            " decided_by='system', decision_note='超时未审批，系统自动拒绝' "
            "where status='pending' and requested_at < now() - make_interval(secs => :secs)"
        ), {"secs": settings.approval_timeout_s})
        await session.commit()
        # 行数表示本次更新了多少审批记录，不是有多少业务工具已停止或图已结束。
        return result.rowcount or 0
