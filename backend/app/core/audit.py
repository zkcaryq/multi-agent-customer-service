"""审计留痕。

人工审批结论、退款受理等敏感动作写 audit_log，供事后追溯。
写失败不影响业务流程，只打警告。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from app.core.log import get_logger
from app.models.database import AdminUser, AuditLog
from app.models.engine import async_session_factory


async def record_audit(
    *,
    action: str,
    entity_type: str,
    entity_id: int | None = None,
    summary: str = "",
    detail: dict[str, Any] | None = None,
    actor: str | None = None,
) -> None:
    try:
        async with async_session_factory() as session:
            admin_id = None
            if actor:
                admin_id = (await session.execute(
                    select(AdminUser.id).where(AdminUser.username == actor).limit(1)
                )).scalar_one_or_none()

            session.add(AuditLog(
                admin_user_id=admin_id,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                summary=summary[:512],
                detail=json.dumps(detail, ensure_ascii=False) if detail else None,
            ))
            await session.commit()
    except Exception as exc:
        get_logger(__name__).warning("audit write failed (business unaffected): %s", exc)
