"""不依赖外部服务的纯逻辑单元测试：口令/令牌、审批语义、工具入参校验。

这三块都是"错了不会立刻报错、但线上会出事"的地方，所以给最小闭环。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.auth import (
    AuthError,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.core.hitl import parse_decision, should_handoff, user_requests_human
from app.agents.router import collaboration_candidates
from app.tools.base import ToolDefinition, ToolParameterDef
from app.tools.registry import ToolSpec, validate_arguments
from app.schemas.chat import ChatRequest


def test_chat_request_uses_zero_for_a_new_conversation():
    request = ChatRequest(user_id="user_001", message="hello")
    assert request.conversation_id == 0


def test_chat_request_rejects_timestamp_as_integer_primary_key():
    with pytest.raises(ValidationError):
        ChatRequest(
            conversation_id=1_792_944_052_677,
            user_id="user_001",
            message="hello",
        )


# ── 口令与令牌 ────────────────────────────────────────────────────────────────

def test_password_roundtrip():
    stored = hash_password("test-password-123")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("test-password-123", stored)
    assert not verify_password("wrong", stored)


def test_same_password_hashes_differently_because_of_salt():
    a, b = hash_password("same"), hash_password("same")
    assert a != b
    assert verify_password("same", a) and verify_password("same", b)


def test_verify_password_rejects_malformed_hash():
    # 库里被写坏/被换算法的哈希一律判失败，不能抛异常把登录接口打挂
    assert not verify_password("x", "not-a-hash")
    assert not verify_password("x", "bcrypt$1$aa$bb")


def test_token_roundtrip_carries_identity_and_role():
    user = SimpleNamespace(id=7, username="ops_lead", role="supervisor")
    token, expires_in = create_token(user)
    claims = decode_token(token)
    assert claims["sub"] == "ops_lead"
    assert claims["uid"] == 7
    assert claims["role"] == "supervisor"
    assert expires_in > 0


def test_tampered_token_is_rejected():
    user = SimpleNamespace(id=7, username="ops_lead", role="supervisor")
    token, _ = create_token(user)
    with pytest.raises(AuthError):
        decode_token(token[:-4] + "aaaa")


def test_forged_role_token_is_rejected():
    # 只改 payload 里的 role 也想进后台——签名对不上就必须拒
    import base64
    import json

    user = SimpleNamespace(id=7, username="op", role="operator")
    token, _ = create_token(user)
    header, payload, _sig = token.split(".")

    def b64d(x: str) -> bytes:
        return base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))

    claims = json.loads(b64d(payload))
    claims["role"] = "admin"
    forged_payload = base64.urlsafe_b64encode(
        json.dumps(claims, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    with pytest.raises(AuthError):
        decode_token(f"{header}.{forged_payload}.{_sig}")


# ── 审批语义 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    ({"approved": True, "note": "同意"}, True),
    ({"approved": False, "note": "金额过大"}, False),
    ("确认执行", True),
    ("同意", True),
    ("驳回", False),
    ("拒绝", False),
    ("", False),                 # 空输入不许放行高危操作
    ("嗯", False),               # 语义不明一律不执行
    ("先别动", False),
])
def test_parse_decision_never_approves_ambiguous_input(value, expected):
    approved, _note = parse_decision(value)
    assert approved is expected


def test_parse_decision_keeps_the_note_for_audit():
    _approved, note = parse_decision({"approved": True, "note": "主管已核对金额"})
    assert note == "主管已核对金额"


def test_reject_word_wins_over_approve_word():
    # 混合表述（既有"同意"又有"不通过"）落在拒绝侧：_from_words 先判拒绝词
    assert parse_decision("同意退款不通过")[0] is False
    assert parse_decision("可以，但这次不同意")[0] is False


@pytest.mark.parametrize("text, expected", [
    ("confirmed", True),   # 含字母 n 的英文批准语不能被 "n" 误判成拒绝
    ("approve", True),
    ("yes", True),
    ("n", False),
    ("no", False),
])
def test_english_decision_matches_whole_words(text, expected):
    assert parse_decision(text)[0] is expected


# ── 转人工判定 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["转人工", "我要找人工客服", "叫个真人来", "给我找客服"])
def test_user_requests_human(text):
    assert user_requests_human(text)


def test_plain_question_does_not_trigger_handoff():
    assert not user_requests_human("这个手机电池多大")
    assert not user_requests_human("")


def test_complaint_domain_always_hands_off():
    # 投诉域本身要求人工跟进，不必等用户开口
    assert should_handoff("你们卖的可能是假货", "complaint")
    assert not should_handoff("这个手机电池多大", "pre_sales")


# ── Multi-Agent 协作路由 ─────────────────────────────────────────────────────

def test_single_domain_query_keeps_one_agent():
    assert collaboration_candidates("这个手机多少钱", "pre_sales") == ["pre_sales"]


def test_compound_query_adds_a_read_only_consultant():
    agents = collaboration_candidates("我要投诉假货，还要查询退款什么时候到账", "complaint")
    assert agents == ["complaint", "after_sales"]


def test_primary_agent_always_stays_first():
    agents = collaboration_candidates("查物流并开发票", "during_sales")
    assert agents[0] == "during_sales"
    assert "after_sales" in agents


# ── 工具入参校验 ──────────────────────────────────────────────────────────────

def _spec(**kwargs) -> ToolSpec:
    definition = ToolDefinition(
        name="request_refund",
        display_name="发起退款",
        description="按订单号发起退款",
        parameters=[
            ToolParameterDef("order_id", "string", "订单号", is_required=True),
            ToolParameterDef("amount", "number", "退款金额"),
            ToolParameterDef("urgent", "boolean", "是否加急"),
        ],
        context_params=["user_id", "conversation_id"],
        **kwargs,
    )
    return ToolSpec(definition=definition, handler=lambda **_kw: None, agent_id=3)


CONTEXT = {"user_id": "user_001", "conversation_id": 42}


def test_valid_arguments_are_normalized_and_context_injected():
    args, errors = validate_arguments(
        _spec(), {"order_id": "ORD-1", "amount": 199.0, "urgent": "true"}, context=CONTEXT,
    )
    assert errors == []
    assert args == {
        "order_id": "ORD-1", "amount": 199.0, "urgent": True,
        "user_id": "user_001", "conversation_id": 42,
    }


def test_unknown_parameter_is_rejected():
    _args, errors = validate_arguments(
        _spec(), {"order_id": "ORD-1", "coupon_code": "FREE"}, context=CONTEXT,
    )
    assert any("未声明" in e for e in errors)


def test_required_parameter_missing_is_rejected():
    _args, errors = validate_arguments(_spec(), {"amount": 10}, context=CONTEXT)
    assert any("order_id" in e for e in errors)


def test_empty_string_counts_as_missing_required():
    # 模型常给出 order_id="" ，不能当合法参数透传到下游
    _args, errors = validate_arguments(_spec(), {"order_id": ""}, context=CONTEXT)
    assert any("order_id" in e for e in errors)


def test_type_coercion_failure_is_reported_not_crashed():
    _args, errors = validate_arguments(
        _spec(), {"order_id": "ORD-1", "amount": "很多钱"}, context=CONTEXT,
    )
    assert any("amount" in e and "类型错误" in e for e in errors)


def test_model_cannot_forge_trusted_context_params():
    # 越权的关键拦截点：模型若自己塞了 user_id，直接判错而不是覆盖后放行
    args, errors = validate_arguments(
        _spec(), {"order_id": "ORD-1", "user_id": "victim_999"}, context=CONTEXT,
    )
    assert any("系统注入" in e or "不能提供" in e for e in errors)
    assert args.get("user_id") != "victim_999"


def test_missing_context_is_an_error_not_a_silent_empty():
    _args, errors = validate_arguments(_spec(), {"order_id": "ORD-1"}, context={})
    assert any("上下文" in e for e in errors)


def test_function_schema_hides_context_params_from_the_model():
    """可信上下文不进 function schema：模型看不到也就编不出、伪造不了。"""
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(_spec().definition, _spec().handler, agent_id=3)
    schema = registry.to_openai_tools(3)[0]["function"]["parameters"]
    assert "user_id" not in schema["properties"]
    assert "conversation_id" not in schema["properties"]
    assert schema["required"] == ["order_id"]


# ── provider 装配 ────────────────────────────────────────────────────────────

def test_chat_override_ignores_configured_provider():
    """自检注入桩必须与 CHAT_PROVIDER 的取值无关。

    曾经脚本往 _cache["chat:openai_compat"] 里塞桩，配置一换成 langchain_openai
    就静默失效，自检直接打到真实模型——这条测试盯的就是这个回归。
    """
    from app.providers import registry

    stub = object()
    try:
        registry.override_chat(stub)  # type: ignore[arg-type]
        assert registry.get_chat() is stub
    finally:
        registry.override_chat(None)


def test_chat_override_reset_falls_back_to_config():
    from app.config import settings
    from app.providers import registry

    registry.override_chat(None)
    provider = registry.get_chat()
    assert type(provider).__name__ != "object"
    assert settings.chat_provider in registry._CHAT


# ── 意图路由守卫 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected", [
    # 明确要退款，即使提到商品不符也不该去投诉域（否则会建投诉单，退款流程跑偏）
    ("这单我要退款，商品和描述不符", "after_sales"),
    ("我要退货退款", "after_sales"),
    # 有明确投诉诉求时保持投诉域
    ("我要投诉你们，要求赔偿", "complaint"),
    ("东西是假货，我要退款并投诉", "complaint"),
])
def test_refund_request_is_not_misrouted_to_complaint(query, expected):
    from app.agents.router import _guard_routing

    assert _guard_routing(query, "complaint") == expected


def test_guard_never_changes_non_complaint_targets():
    from app.agents.router import _guard_routing

    for target in ("pre_sales", "during_sales", "after_sales"):
        assert _guard_routing("我要退款", target) == target
