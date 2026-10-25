"""高危动作的人工审批策略。

集中回答三个问题，避免散落在各 Agent 的图逻辑里：
1. **哪些工具必须人工确认？**（`HIGH_RISK_TOOLS`）
2. **人回复的"同意/拒绝"怎么归一化？**（`parse_decision`）
3. **什么情况该转人工？**（`should_handoff`）

【为什么是纯函数模块】
这些判定同时被四处使用：Agent 的图节点、审批接口、后台展示、离线评测。
一旦某处各写一套，就会出现"Agent 认为该审批、评测认为不该"的分歧。
所以这里只放无状态的纯函数与常量表。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# 工具名 -> 展示给审批人的动作。新增高危操作只改这张表。
# 为什么用"白名单"而不是"黑名单"：新增工具默认是安全的（只读），
# 要动钱/动地址/建单必须显式登记，避免"忘了标高危"导致绕过审批。
HIGH_RISK_TOOLS: dict[str, str] = {
    "request_refund": "发起退款",
    "modify_address": "修改收货地址",
    "create_service_ticket": "创建投诉工单",
}

# 中文按子串匹配（"同意"/"批准"在句子里出现即可）；英文按整词匹配（见 _from_words）。
# "不通过" 必须单独列出：它包含"通过"这个批准词，只靠批准词表会误判为同意。
_APPROVE_WORDS = ("确认", "同意", "批准", "通过", "可以", "好的", "执行",
                  "yes", "y", "ok", "confirm", "confirmed", "approve")
_REJECT_WORDS = ("拒绝", "取消", "不同意", "不通过", "驳回", "不要", "no", "n", "reject", "deny")

# 英文审批语按整词切分用
_WORD_SPLIT = re.compile(r"[^a-z0-9]+")

# 用户明确开口要人工
HANDOFF_KEYWORDS = ("转人工", "人工客服", "找客服", "找人工", "人工处理", "真人")


def user_requests_human(question: str) -> bool:
    """用户是否明确要人工。子串匹配足够：这些短语几乎没有歧义用法。"""
    return any(kw in (question or "") for kw in HANDOFF_KEYWORDS)


def should_handoff(question: str, intent_agent: str = "") -> bool:
    """是否需要转人工的唯一判定口径。

    Agent 的实际决策和离线评测都走这里，避免两边各写一套规则对不上。
    投诉域本身就要求人工跟进（工单已建或在建），不必等用户开口。
    """
    return user_requests_human(question) or intent_agent == "complaint"


def is_high_risk(tool_name: str) -> bool:
    """单个工具是否高危。注意只认工具名，不看参数（同一工具的任何调用都要审批）。"""
    return tool_name in HIGH_RISK_TOOLS


def high_risk_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从本轮待执行工具里挑出高危的。

    返回的是"调用"本身（含参数），因为审批人需要看到"动哪个订单、退多少钱"。
    """
    return [tc for tc in tool_calls if is_high_risk(tc.get("name", ""))]


def pending_payload(calls: list[dict[str, Any]], agent_name: str, conversation_id: int) -> dict[str, Any]:
    """交给前端/审批人的确认单。参数一并给出，审批人看得见要动哪个订单。

    这个 dict 会：
    - 作为 interrupt() 的 payload 抛给调用层 → 前端渲染确认卡片；
    - 被 ensure_pending 落进 approval_requests.actions → 后台审批页展示。
    所以字段命名要同时照顾前端与后台，保持一致（tool/label/arguments）。
    """
    return {
        "agent": agent_name,
        "conversation_id": conversation_id,
        # 用带时区的 ISO 时间给前端展示；注意这与数据库里的 naive UTC 是两套表示
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
    返回 (是否批准, 备注)。
    """
    if isinstance(resume_value, dict):
        approved = resume_value.get("approved")
        note = str(resume_value.get("note", "") or "")
        if isinstance(approved, bool):
            # 正常路径：后台/前端传的就是布尔
            return approved, note
        # 兼容 {"approved": "同意"} 这种把布尔写成文字的情况
        approved, text = _from_words(str(approved) or note)
        return approved, note
    # 用户直接回了一句"确认"/"拒绝"
    return _from_words(str(resume_value or ""))


def _from_words(text: str) -> tuple[bool, str]:
    """自然语言 → 布尔。判定顺序：先拒绝词，再批准词，都不中→拒绝。"""
    lowered = text.strip().lower()
    # 英文词按整词匹配：否则 "n" 会命中任何含字母 n 的句子（"confirmed" 被当成拒绝）
    tokens = {t for t in _WORD_SPLIT.split(lowered) if t}

    def hit(words: tuple[str, ...]) -> bool:
        for word in words:
            if word.isascii():
                if word in tokens:      # 英文：必须是独立单词
                    return True
            elif word in lowered:       # 中文：子串即可
                return True
        return False

    # 先判拒绝：混合表述（"同意退款不通过"）必须落在拒绝侧，宁可少做不可错做
    if hit(_REJECT_WORDS):
        return False, text
    if hit(_APPROVE_WORDS):
        return True, text
    # 语义不明时一律不执行高危操作，转人工
    return False, text
