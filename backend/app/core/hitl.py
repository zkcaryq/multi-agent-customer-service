"""高危动作的人工审批策略。

哪些工具必须人工确认、怎么判定人回复是"同意"还是"拒绝"，集中在这里，
不散落到各 Agent 的图逻辑里。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# 工具名 -> 展示给审批人的动作。新增高危操作只改这张表。
HIGH_RISK_TOOLS: dict[str, str] = {
    "request_refund": "发起退款",
    "modify_address": "修改收货地址",
    "create_service_ticket": "创建投诉工单",
}

_APPROVE_WORDS = ("确认", "同意", "批准", "通过", "可以", "好的", "执行",
                  "yes", "y", "ok", "confirm", "confirmed", "approve")
_REJECT_WORDS = ("拒绝", "取消", "不同意", "不通过", "驳回", "不要", "no", "n", "reject", "deny")

# 英文审批语按整词切分用
_WORD_SPLIT = re.compile(r"[^a-z0-9]+")

# 用户明确开口要人工
HANDOFF_KEYWORDS = ("转人工", "人工客服", "找客服", "找人工", "人工处理", "真人")


def user_requests_human(question: str) -> bool:
    return any(kw in (question or "") for kw in HANDOFF_KEYWORDS)


def should_handoff(question: str, intent_agent: str = "") -> bool:
    """是否需要转人工的唯一判定口径。

    Agent 的实际决策和离线评测都走这里，避免两边各写一套规则对不上。
    投诉域本身就要求人工跟进（工单已建或在建），不必等用户开口。
    """
    return user_requests_human(question) or intent_agent == "complaint"


def is_high_risk(tool_name: str) -> bool:
    return tool_name in HIGH_RISK_TOOLS


def high_risk_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tc for tc in tool_calls if is_high_risk(tc.get("name", ""))]


def pending_payload(calls: list[dict[str, Any]], agent_name: str, conversation_id: int) -> dict[str, Any]:
    """交给前端/审批人的确认单。参数一并给出，审批人看得见要动哪个订单。"""
    return {
        "agent": agent_name,
        "conversation_id": conversation_id,
        "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actions": [
            {"tool": c["name"], "label": HIGH_RISK_TOOLS[c["name"]], "arguments": c.get("arguments", {})}
            for c in calls
        ],
        "prompt": "以下操作有风险，需人工确认后才执行。请回复\"确认\"或\"拒绝\"。",
    }


def parse_decision(resume_value: Any) -> tuple[bool, str]:
    """审批结果归一化。

    既接受结构化 {"approved": true/false, "note": ...}，也接受人工直接敲的自然语言。
    """
    if isinstance(resume_value, dict):
        approved = resume_value.get("approved")
        note = str(resume_value.get("note", "") or "")
        if isinstance(approved, bool):
            return approved, note
        approved, text = _from_words(str(approved) or note)
        return approved, note
    return _from_words(str(resume_value or ""))


def _from_words(text: str) -> tuple[bool, str]:
    lowered = text.strip().lower()
    # 英文词按整词匹配：否则 "n" 会命中任何含字母 n 的句子（"confirmed" 被当成拒绝）
    tokens = {t for t in _WORD_SPLIT.split(lowered) if t}

    def hit(words: tuple[str, ...]) -> bool:
        for word in words:
            if word.isascii():
                if word in tokens:
                    return True
            elif word in lowered:
                return True
        return False

    if hit(_REJECT_WORDS):
        return False, text
    if hit(_APPROVE_WORDS):
        return True, text
    # 语义不明时一律不执行高危操作，转人工
    return False, text
