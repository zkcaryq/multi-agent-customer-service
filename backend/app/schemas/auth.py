"""后台登录相关的请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StaffOut(BaseModel):
    id: int
    username: str
    role: str


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: StaffOut
