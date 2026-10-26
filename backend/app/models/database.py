"""ORM models — matches schema.sql MVP core tables."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Model registry ────────────────────────────────────────────────────────────

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(128))
    endpoint_url: Mapped[str] = mapped_column(String(512), nullable=False)
    api_key_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    context_window: Mapped[Optional[int]] = mapped_column(Integer)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    pricing_tier: Mapped[Optional[str]] = mapped_column(String(16))
    cost_per_1k_input: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    cost_per_1k_output: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# ── Content safety ────────────────────────────────────────────────────────────

class ContentSafetyRule(Base):
    __tablename__ = "content_safety_rules"

    id: Mapped[int] = mapped_column(primary_key=True)
    rule_type: Mapped[str] = mapped_column(String(32), nullable=False)
    pattern: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_hit_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# ── Admin ──────────────────────────────────────────────────────────────────────

class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="operator")
    status: Mapped[str] = mapped_column(String(16), default="active")
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# ── Agents ─────────────────────────────────────────────────────────────────────

class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(128))
    model: Mapped[str] = mapped_column(String(32), nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    retrieval_params: Mapped[list[AgentRetrievalParam]] = relationship(
        back_populates="agent", lazy="selectin"
    )


class AgentRetrievalParam(Base):
    __tablename__ = "agent_retrieval_params"
    __table_args__ = (UniqueConstraint("agent_id", "param_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    param_key: Mapped[str] = mapped_column(String(64), nullable=False)
    param_value: Mapped[str] = mapped_column(String(128), nullable=False)

    agent: Mapped[Agent] = relationship(back_populates="retrieval_params")


# ── Intent definitions ────────────────────────────────────────────────────────

class IntentDefinition(Base):
    __tablename__ = "intent_definitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    intent_name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(128))
    target_agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    example_utterances: Mapped[Optional[str]] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    target_agent: Mapped[Agent] = relationship(lazy="selectin")


class SlotDefinition(Base):
    __tablename__ = "slot_definitions"
    __table_args__ = (UniqueConstraint("intent_name", "slot_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    intent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    slot_name: Mapped[str] = mapped_column(String(64), nullable=False)
    slot_type: Mapped[str] = mapped_column(String(32), nullable=False)
    question_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    validation_rule: Mapped[Optional[str]] = mapped_column(String(256))
    is_required: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── Tools ──────────────────────────────────────────────────────────────────────

class Tool(Base):
    __tablename__ = "tools"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(128))
    description: Mapped[Optional[str]] = mapped_column(Text)
    endpoint_url: Mapped[str] = mapped_column(String(512), nullable=False)
    http_method: Mapped[str] = mapped_column(String(8), default="POST")
    timeout_ms: Mapped[int] = mapped_column(Integer, default=5000)
    retry_count: Mapped[int] = mapped_column(Integer, default=2)
    auth_type: Mapped[str] = mapped_column(String(32), default="none")
    auth_credential_ref: Mapped[Optional[str]] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(16), default="1.0")
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    parameters: Mapped[list[ToolParameter]] = relationship(
        back_populates="tool", lazy="selectin"
    )


class ToolParameter(Base):
    __tablename__ = "tool_parameters"

    id: Mapped[int] = mapped_column(primary_key=True)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tools.id"), nullable=False)
    param_name: Mapped[str] = mapped_column(String(64), nullable=False)
    param_type: Mapped[str] = mapped_column(String(32), nullable=False)
    param_desc: Mapped[Optional[str]] = mapped_column(String(256))
    is_required: Mapped[bool] = mapped_column(Boolean, default=False)
    default_value: Mapped[Optional[str]] = mapped_column(String(128))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    tool: Mapped[Tool] = relationship(back_populates="parameters")


class AgentTool(Base):
    __tablename__ = "agent_tools"
    __table_args__ = (UniqueConstraint("agent_id", "tool_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tools.id"), nullable=False)
    tool_version: Mapped[Optional[str]] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── Users ──────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    nickname: Mapped[Optional[str]] = mapped_column(String(128))
    phone: Mapped[str] = mapped_column(String(256), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class UserAttribute(Base):
    __tablename__ = "user_attributes"
    __table_args__ = (UniqueConstraint("user_id", "attr_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    attr_key: Mapped[str] = mapped_column(String(64), nullable=False)
    attr_value: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class UserTag(Base):
    __tablename__ = "user_tags"
    __table_args__ = (UniqueConstraint("user_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    tag: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── Conversations & messages ───────────────────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    agent_id: Mapped[Optional[int]] = mapped_column(ForeignKey("agents.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active")
    title: Mapped[Optional[str]] = mapped_column(String(256))
    source: Mapped[str] = mapped_column(String(32), default="widget")
    last_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", lazy="selectin"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(8), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    trace: Mapped[Optional[MessageTrace]] = relationship(back_populates="message", uselist=False)


class MessageTrace(Base):
    __tablename__ = "message_traces"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), unique=True, nullable=False)
    intent_agent: Mapped[Optional[str]] = mapped_column(String(64))
    intent_confidence: Mapped[Optional[float]] = mapped_column(Numeric(4, 3))
    intent_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    retrieval_query: Mapped[Optional[str]] = mapped_column(Text)
    retrieval_recalled_count: Mapped[Optional[int]] = mapped_column(Integer)
    retrieval_top_k: Mapped[Optional[int]] = mapped_column(Integer)
    retrieval_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    rerank_model: Mapped[Optional[str]] = mapped_column(String(64))
    rerank_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    generation_model: Mapped[Optional[str]] = mapped_column(String(32))
    generation_prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    generation_completion_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    generation_ttft_ms: Mapped[Optional[int]] = mapped_column(Integer)
    generation_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    message: Mapped[Message] = relationship(back_populates="trace")


class TraceFunctionCall(Base):
    __tablename__ = "trace_function_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    trace_id: Mapped[int] = mapped_column(ForeignKey("message_traces.id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_version: Mapped[Optional[str]] = mapped_column(String(16))
    parameters: Mapped[Optional[str]] = mapped_column(Text)
    is_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    result_summary: Mapped[Optional[str]] = mapped_column(Text)
    attempt_index: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── Feedback ───────────────────────────────────────────────────────────────────

class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=False)
    rating: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    reviewed_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("admin_users.id"), nullable=True
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    review_status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# ── Knowledge base ─────────────────────────────────────────────────────────────

class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    split_method: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(512))
    original_filename: Mapped[Optional[str]] = mapped_column(String(256))
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)  # BIGINT in schema
    uploaded_by: Mapped[Optional[int]] = mapped_column(ForeignKey("admin_users.id"), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("knowledge_documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    last_retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChunkMetadata(Base):
    __tablename__ = "chunk_metadata"
    __table_args__ = (UniqueConstraint("chunk_id", "meta_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("knowledge_chunks.id"), nullable=False)
    meta_key: Mapped[str] = mapped_column(String(64), nullable=False)
    meta_value: Mapped[Optional[str]] = mapped_column(String(512))


class ChunkTag(Base):
    __tablename__ = "chunk_tags"
    __table_args__ = (UniqueConstraint("chunk_id", "tag"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("knowledge_chunks.id"), nullable=False)
    tag: Mapped[str] = mapped_column(String(128), nullable=False)


# ── Handoff ────────────────────────────────────────────────────────────────────

class HumanHandoff(Base):
    __tablename__ = "human_handoffs"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    intent_agent: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    claimed_by: Mapped[Optional[int]] = mapped_column(ForeignKey("admin_users.id"), nullable=True)
    resolution_note: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class HandoffSlot(Base):
    __tablename__ = "handoff_slots"
    __table_args__ = (UniqueConstraint("handoff_id", "slot_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    handoff_id: Mapped[int] = mapped_column(ForeignKey("human_handoffs.id"), nullable=False)
    slot_name: Mapped[str] = mapped_column(String(64), nullable=False)
    slot_value: Mapped[Optional[str]] = mapped_column(String(256))


class HandoffAgentAttempt(Base):
    __tablename__ = "handoff_agent_attempts"
    __table_args__ = (UniqueConstraint("handoff_id", "attempt_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    handoff_id: Mapped[int] = mapped_column(ForeignKey("human_handoffs.id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False)
    result: Mapped[str] = mapped_column(String(128), nullable=False)
    error_detail: Mapped[Optional[str]] = mapped_column(Text)


# ── 业务系统：订单 / 退款 ──────────────────────────────────────────────────────

class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_no: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="unpaid")
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    payment_method: Mapped[Optional[str]] = mapped_column(String(32))
    shipping_address: Mapped[Optional[str]] = mapped_column(String(256))

    # 物流信息随订单存放，轨迹按时间顺序存 JSON 字符串
    carrier: Mapped[Optional[str]] = mapped_column(String(64))
    tracking_no: Mapped[Optional[str]] = mapped_column(String(64))
    logistics_status: Mapped[Optional[str]] = mapped_column(String(32))
    estimated_delivery: Mapped[Optional[str]] = mapped_column(String(32))
    logistics_events: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    paid_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    shipped_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped[User] = relationship(lazy="selectin")
    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order", lazy="selectin"
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False, index=True)
    sku_name: Mapped[str] = mapped_column(String(128), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    price: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False, default=0)

    order: Mapped[Order] = relationship(back_populates="items")


class Refund(Base):
    __tablename__ = "refunds"
    # 一个订单同时只允许一张未终态退款单：审批恢复/进程重启后重复提交天然幂等
    __table_args__ = (
        Index(
            "uq_refunds_open_per_order",
            "order_id",
            unique=True,
            postgresql_where=text("status NOT IN ('refunded', 'rejected', 'cancelled')"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    refund_no: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending_review")
    handled_by: Mapped[Optional[str]] = mapped_column(String(64))
    approval_note: Mapped[Optional[str]] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    order: Mapped[Order] = relationship(lazy="selectin")


# ── 离线评测（对应 schema.sql 的 evaluation_runs / evaluation_results）─────────

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("idx_er_agent_id", "agent_id"),
        Index("idx_er_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    agent_id: Mapped[Optional[int]] = mapped_column(ForeignKey("agents.id"), nullable=True)
    test_set_name: Mapped[Optional[str]] = mapped_column(String(256))
    total_cases: Mapped[Optional[int]] = mapped_column(Integer)
    completed_cases: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    intent_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    retrieval_hit_rate: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    retrieval_recall: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    retrieval_precision: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    retrieval_mrr: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    tool_call_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    generation_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    hallucination_rate: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    avg_ttft_ms: Mapped[Optional[int]] = mapped_column(Integer)
    avg_total_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_by: Mapped[Optional[int]] = mapped_column(ForeignKey("admin_users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    results: Mapped[list[EvaluationResult]] = relationship(
        back_populates="run", lazy="selectin", cascade="all, delete-orphan"
    )


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (
        UniqueConstraint("run_id", "case_index"),
        Index("idx_eres_run_id", "run_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False)
    case_index: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)

    expected_intent: Mapped[Optional[str]] = mapped_column(String(64))
    actual_intent: Mapped[Optional[str]] = mapped_column(String(64))
    intent_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    expected_tool: Mapped[Optional[str]] = mapped_column(String(64))
    actual_tool: Mapped[Optional[str]] = mapped_column(String(64))
    tool_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    retrieval_hit: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    retrieval_recall: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    retrieval_precision: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    retrieval_rr: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    generation_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    hallucination: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    actual_answer: Mapped[Optional[str]] = mapped_column(Text)
    expected_answer: Mapped[Optional[str]] = mapped_column(Text)
    ttft_ms: Mapped[Optional[int]] = mapped_column(Integer)
    total_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    run: Mapped[EvaluationRun] = relationship(back_populates="results")


# ── 人工审批队列（LangGraph interrupt 的可查询投影）──────────────────────────

class ApprovalRequest(Base):
    """一次高危操作的审批请求。

    LangGraph 的暂停状态存在 checkpoints 表里，不可直接查询；这张表是它的
    可查询投影，由 Agent 节点在 interrupt 前写入、由审批接口写入结论。
    节点在 resume 时会整段重跑，所以写入必须"存在即跳过"，否则会把
    已经批准的结论倒回 pending。
    """

    __tablename__ = "approval_requests"
    __table_args__ = (UniqueConstraint("agent_name", "conversation_id", name="uq_approval_thread"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128))
    actions: Mapped[str] = mapped_column(Text, nullable=False)  # JSON：要执行的工具与参数
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    decided_by: Mapped[Optional[str]] = mapped_column(String(64))
    decision_note: Mapped[Optional[str]] = mapped_column(String(512))

    @property
    def thread_id(self) -> str:
        return f"{self.agent_name}:{self.conversation_id}"


# ── 审计：人工审批、退款等敏感动作留痕（对应 schema.sql 的 audit_log）──────────

class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    admin_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("admin_users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(String(512))
    detail: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
