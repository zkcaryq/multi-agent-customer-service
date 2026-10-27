"""对话接口的请求/响应模型（含会话号取值范围校验）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # PostgreSQL conversations.id / approval_requests.conversation_id 是 INTEGER。
    # 0 表示新会话；禁止把 Date.now() 之类的毫秒时间戳当数据库主键。
    conversation_id: int = Field(default=0, ge=0, le=2_147_483_647, description="会话 ID，0 表示新会话")
    # 演示客户端自报标识，尚未从认证令牌派生；工具参数可信不等于客户端身份可信。
    user_id: str = Field(default="user_001", description="用户标识")
    message: str = Field(..., min_length=1, max_length=4000, description="用户消息")


class ChatResponse(BaseModel):
    conversation_id: int
    message_id: int | None = None
    reply: str
    intent: str = ""
    handoff: bool = False
    handoff_reason: str = ""
    trace: dict | None = None
    # 非空表示流程停在人工审批上，需要调 /chat/confirm 续跑
    pending_confirmation: dict | None = None


class ConfirmRequest(BaseModel):
    conversation_id: int = Field(..., ge=1, le=2_147_483_647, description="会话 ID")
    agent: str = Field(default="after_sales", description="发起审批的 Agent")
    approved: bool = Field(..., description="人工是否批准")
    note: str = Field(default="", max_length=500, description="审批备注")
    # 公共确认入口可自报此字段；后台入口会用 JWT 用户名覆盖，两者的信任边界不同。
    approver: str = Field(default="operator", max_length=64, description="审批人（审计留痕）")


class FeedbackRequest(BaseModel):
    message_id: int
    rating: str  # "helpful" | "not_helpful"
    reason: str | None = None


class HealthResponse(BaseModel):
    status: str
    agent: str
    version: str
