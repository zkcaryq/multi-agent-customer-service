"""Intent classification and routing to domain agents.

Two-tier strategy:
  1. LLM classifier（模型由 LLM_MODEL_INTENT 配置）— primary, high accuracy
  2. Keyword match — fallback on LLM error or timeout
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# ── Intent result ────────────────────────────────────────────────────────────

VALID_TARGETS = ("pre_sales", "during_sales", "after_sales", "complaint")


@dataclass
class IntentResult:
    intent_name: str
    target_agent: str  # one of VALID_TARGETS
    confidence: float
    classifier: str = "keyword"  # "llm" | "keyword"


# ── LLM classification prompt ────────────────────────────────────────────────

_INTENT_SYSTEM_PROMPT = """你是电商客服意图分类器。根据用户消息，判断应该由哪个 Agent 处理。

Agent 列表及其职责：

1. **pre_sales（售前）**：商品咨询、规格参数、价格查询、库存、促销活动、优惠券、会员权益、积分、商品对比和推荐
2. **during_sales（售中）**：下单问题、支付失败、付款方式、修改订单、取消订单、收货地址修改、发票开具
3. **after_sales（售后）**：物流查询、快递进度、退货申请、退款进度、换货等常规售后处理
4. **complaint（投诉）**：用户表达强烈不满、要维权、要求赔偿/道歉、提到 315、工商、市场监管、消协、媒体曝光、起诉、商品质量问题造假，或反复投诉同一问题

输出规则：
- 只返回 JSON，不要解释
- 常规办事归售后，带情绪/索赔/监管/法律色彩才归 complaint；拿不准时归 after_sales
- 问候语、感谢等非业务消息归类为 pre_sales

输出格式：
{"intent_name": "query_product", "target_agent": "pre_sales", "confidence": 0.95}
"""

_FEW_SHOT_EXAMPLES = [
    # 售前
    ("iPhone 15 Pro 多少钱？", '{"intent_name": "query_product", "target_agent": "pre_sales", "confidence": 0.98}'),
    ("这个手机和那个手机哪个好？", '{"intent_name": "query_product", "target_agent": "pre_sales", "confidence": 0.92}'),
    ("有什么优惠活动？", '{"intent_name": "query_product", "target_agent": "pre_sales", "confidence": 0.95}'),
    ("PLUS 会员有什么权益？", '{"intent_name": "query_product", "target_agent": "pre_sales", "confidence": 0.97}'),
    ("256G 的有货吗？", '{"intent_name": "query_product", "target_agent": "pre_sales", "confidence": 0.94}'),
    ("你好", '{"intent_name": "query_product", "target_agent": "pre_sales", "confidence": 0.90}'),
    # 售中
    ("我付了钱但显示没成功", '{"intent_name": "place_order", "target_agent": "during_sales", "confidence": 0.96}'),
    ("帮我改一下收货地址", '{"intent_name": "place_order", "target_agent": "during_sales", "confidence": 0.97}'),
    ("我要开发票", '{"intent_name": "place_order", "target_agent": "during_sales", "confidence": 0.98}'),
    ("支付失败了怎么重试？", '{"intent_name": "place_order", "target_agent": "during_sales", "confidence": 0.95}'),
    ("订单可以取消吗？", '{"intent_name": "place_order", "target_agent": "during_sales", "confidence": 0.93}'),
    # 售后
    ("快递到哪了？", '{"intent_name": "check_logistics", "target_agent": "after_sales", "confidence": 0.99}'),
    ("我要退货退款", '{"intent_name": "check_logistics", "target_agent": "after_sales", "confidence": 0.98}'),
    ("收到的东西是坏的", '{"intent_name": "check_logistics", "target_agent": "after_sales", "confidence": 0.97}'),
    ("退款什么时候到账？", '{"intent_name": "check_logistics", "target_agent": "after_sales", "confidence": 0.96}'),
    ("快递显示签收了但我没收到", '{"intent_name": "check_logistics", "target_agent": "after_sales", "confidence": 0.94}'),
    # 投诉
    ("你们这是欺诈，我要找315投诉", '{"intent_name": "complaint", "target_agent": "complaint", "confidence": 0.97}'),
    ("东西是假货，我要求赔偿并投诉", '{"intent_name": "complaint", "target_agent": "complaint", "confidence": 0.96}'),
    ("这已经是第三次了，我要投诉你们客服", '{"intent_name": "complaint", "target_agent": "complaint", "confidence": 0.93}'),
]


async def _classify_llm(user_query: str) -> IntentResult | None:
    """用配置指定的轻量模型分类。失败返回 None，由关键词规则兜底。"""
    try:
        from app.providers import get_chat

        messages: list[dict] = [
            {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
        ]
        for user_msg, assistant_msg in _FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": user_query})

        result = await get_chat().chat(
            role="intent",
            messages=messages,
            temperature=0.0,
            max_tokens=128,
            json_mode=True,
        )
        data = json.loads(result.content)
    except Exception:
        return None

    target = data.get("target_agent", "pre_sales")
    if target not in VALID_TARGETS:
        target = "pre_sales"

    return IntentResult(
        intent_name=data.get("intent_name", "query_product"),
        target_agent=_guard_routing(user_query, target),
        confidence=float(data.get("confidence", 0.7)),
        classifier="llm",
    )


# 用户明确要退款/退货/换货时，即使提到"商品和描述不符"也不该落到投诉域：
# 一旦判成 complaint，Agent 会去建投诉工单，退款流程整个跑偏。
_AFTER_SALES_ACTIONS = ("退款", "退货", "换货", "退钱")
_COMPLAINT_WORDS = ("投诉", "赔偿", "维权", "道歉", "举报", "起诉", "欺诈", "假货", "315", "曝光")


def _guard_routing(user_query: str, target: str) -> str:
    if target != "complaint":
        return target
    wants_refund = any(k in user_query for k in _AFTER_SALES_ACTIONS)
    wants_complaint = any(k in user_query for k in _COMPLAINT_WORDS)
    return "after_sales" if wants_refund and not wants_complaint else target


# ── Keyword fallback (kept from original implementation) ──────────────────────

_INTENT_RULES: list[tuple[list[str], str, str]] = [
    (
        ["投诉", "维权", "赔偿", "道歉", "315", "12315", "工商", "市场监管", "消协",
         "曝光", "媒体", "起诉", "律师", "欺诈", "假货", "造假", "态度差", "骗人"],
        "complaint",
        "complaint",
    ),
    (
        ["商品", "有没有", "多少钱", "价格", "规格", "颜色", "内存", "配置",
         "对比", "哪个好", "推荐", "优惠", "优惠券", "促销", "打折", "活动",
         "会员", "积分", "权益", "库存", "有货", "有现货"],
        "query_product",
        "pre_sales",
    ),
    (
        ["下单", "支付", "付款", "付了", "没成功", "地址", "改地址",
         "发票", "开票", "未支付", "取消订单", "修改订单",
         "花呗", "分期", "扣款", "扣了", "限额", "抬头"],
        "place_order",
        "during_sales",
    ),
    (
        ["物流", "快递", "发货", "到哪了", "货到哪", "单号", "退货",
         "退款", "换货", "质量问题", "坏了", "退钱",
         "配送", "运费", "签收", "到货", "几天到",
         "订单详情", "查订单", "我的订单", "转人工", "人工客服"],
        "check_logistics",
        "after_sales",
    ),
]


def _classify_keyword(user_query: str) -> IntentResult:
    """Simple keyword-based intent classification (fallback)."""
    best_score = 0.0
    best_intent = "query_product"
    best_agent = "pre_sales"

    for keywords, intent_name, agent in _INTENT_RULES:
        score = sum(1 for kw in keywords if kw in user_query) / len(keywords)
        if score > best_score:
            best_score = score
            best_intent = intent_name
            best_agent = agent

    confidence = min(best_score * 3, 0.98) if best_score > 0 else 0.5
    return IntentResult(
        intent_name=best_intent,
        target_agent=best_agent,
        confidence=round(confidence, 2),
        classifier="keyword",
    )


# ── Public API ────────────────────────────────────────────────────────────────


async def classify_intent(user_query: str) -> IntentResult:
    """LLM 优先 + 关键词兜底的意图分类。"""
    result = await _classify_llm(user_query)
    if result is not None:
        return result
    return _classify_keyword(user_query)


def agent_by_name(name: str):
    """按 Agent 名取实例。审批回调只带名字，需要据此找回同一条 LangGraph 线程。"""
    if name == "complaint":
        from app.agents.complaint.agent import complaint_agent

        return complaint_agent
    if name == "during_sales":
        from app.agents.during_sales.agent import during_sales_agent

        return during_sales_agent
    if name == "after_sales":
        from app.agents.after_sales.agent import after_sales_agent

        return after_sales_agent
    from app.agents.pre_sales.agent import pre_sales_agent

    return pre_sales_agent


def route_to_agent(intent: IntentResult):
    """Return the appropriate agent instance based on intent."""
    return agent_by_name(intent.target_agent)


def collaboration_candidates(user_query: str, primary_agent: str, limit: int = 2) -> list[str]:
    """识别复合诉求涉及的领域，主 Agent 始终排第一。

    LLM 负责主路由；这里使用确定性关键词补充协作 Agent，避免为了找辅助 Agent
    再增加一次不稳定的模型分类。辅助 Agent 只提供建议，不执行有副作用的工具。
    """
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for keywords, _intent_name, agent_name in _INTENT_RULES:
        if agent_name in seen:
            continue
        score = sum(1 for keyword in keywords if keyword in user_query)
        if score:
            scored.append((score, agent_name))
            seen.add(agent_name)

    scored.sort(key=lambda item: item[0], reverse=True)
    agents = [primary_agent]
    for _score, agent_name in scored:
        if agent_name not in agents:
            agents.append(agent_name)
        if len(agents) >= max(1, limit):
            break
    return agents
