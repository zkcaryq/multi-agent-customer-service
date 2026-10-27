"""审计留痕。

人工审批结论、退款受理等敏感动作写 audit_log，供事后追溯。
写失败不影响业务流程，只打警告。
"""

from __future__ import annotations

import json  # detail 存成 JSON 文本列，不是数据库原生 JSONB 类型
from typing import Any

from sqlalchemy import select  # 只用它按用户名反查一次 admin_users.id

from app.core.log import get_logger  # 失败时唯一的出口：一条 warning，没有告警对接
from app.models.database import AdminUser, AuditLog
from app.models.engine import async_session_factory  # 每次调用开一个新会话，与业务事务完全隔离


async def record_audit(
    *,  # 全部关键字传参：字段多且含义相近（entity_type/entity_id），位置传参很容易写反
    action: str,
    entity_type: str,
    entity_id: int | None = None,  # 允许 None：全局性动作（如登录）没有对应行主键
    summary: str = "",
    detail: dict[str, Any] | None = None,
    actor: str | None = None,  # 传的是用户名而不是 id，认证与否的可信度由调用方决定
) -> None:
    try:
        async with async_session_factory() as session:
            # 没有 actor 或查不到后台用户名时留空；这不构成经过认证的用户身份记录。
            admin_id = None
            # 后台入口将 JWT 用户名传入；公共确认入口当前允许客户端自报 approver，需要区分可信度。
            if actor:
                admin_id = (await session.execute(
                    select(AdminUser.id).where(AdminUser.username == actor).limit(1)
                )).scalar_one_or_none()  # username 上有唯一约束，limit(1) 是保险；真出现多行时 scalar_one_or_none 抛 MultipleResultsFound，被外层 except 吞成"审计丢失"

            # 普通业务表留痕，没有只追加存储、哈希防篡改或独立合规审计系统。
            session.add(AuditLog(
                admin_user_id=admin_id,
                action=action,  # 调用方自由字符串（admin_login / refund_accepted 等），没有枚举校验，写错也能存进去
                entity_type=entity_type,
                entity_id=entity_id,
                # ip_address 列存在但这里从不填，事后无法定位来源 IP；action/entity_type 也只靠约定对齐
                # 限制列宽；detail 不做脱敏，可能包含地址或业务文本，不应认为全量日志都已脱敏。
                summary=summary[:512],  # 与 String(512) 对齐；超长部分静默截断不可恢复，宁可少写也别指望它做全文
                detail=json.dumps(detail, ensure_ascii=False) if detail else None,  # ensure_ascii=False 省体积且可读；空 dict 存 NULL 而不是 "null"，查"有没有明细"更直观。detail 含 datetime 等非 JSON 类型时 dumps 抛 TypeError，整条审计都写不进
            ))
            # 审计独立提交，与退款/改地址事务不原子；业务成功而审计失败是允许发生的降级。
            await session.commit()
    except Exception as exc:
        # 兜住建会话/查询/序列化/提交任一环节的失败：业务不回滚、调用方拿不到任何异常。
        # 代价是审计可能有洞；且没有 exc_info，只剩一行 message，堆栈信息丢失。
        get_logger(__name__).warning("audit write failed (business unaffected): %s", exc)
