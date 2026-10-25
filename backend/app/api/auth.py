"""后台登录接口（唯一不需要令牌的入口）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.audit import record_audit
from app.core.auth import Staff, authenticate, create_token, require_staff
from app.schemas.auth import LoginRequest, StaffOut, TokenResponse

router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    user = await authenticate(body.username, body.password)
    if user is None:
        # 不区分"用户不存在"与"密码错"，也不回显是哪个字段错了，避免账号枚举
        raise HTTPException(401, "用户名或密码错误")

    token, expires_in = create_token(user)
    await record_audit(
        action="admin_login",
        entity_type="admin_user",
        entity_id=user.id,
        summary=f"{user.username} 登录后台",
        actor=user.username,
    )
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=StaffOut(id=user.id, username=user.username, role=user.role),
    )


@router.get("/auth/me", response_model=StaffOut)
async def me(staff: Staff = Depends(require_staff)) -> StaffOut:
    return StaffOut(id=staff.id, username=staff.username, role=staff.role)
