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

# re 只服务英文审批语的整词切分；datetime 只给确认单打展示用时间戳；都与判定逻辑无关
import re
from datetime import datetime, timezone
from typing import Any

# 工具名 -> 展示动作。这是“需要审批的工具名单”，未登记的工具默认不审批。
# 新增写工具必须同时审查风险策略；忘记登记确实会绕过审批，不能称为默认安全白名单。
HIGH_RISK_TOOLS: dict[str, str] = {
    # 三项都是"写"操作：动资金、改履约地址、占用真人坐席；查询类工具一律不进名单
    "request_refund": "发起退款",
    "modify_address": "修改收货地址",
    "create_service_ticket": "创建投诉工单",
    # value 只用于确认卡片的 label（pending_payload），判定只看 key 是否存在
}

# 中文按子串匹配（"同意"/"批准"在句子里出现即可）；英文按整词匹配（见 _from_words）。
# "不通过" 必须单独列出：它包含"通过"这个批准词，只靠批准词表会误判为同意。
_APPROVE_WORDS = ("确认", "同意", "批准", "通过", "可以", "好的", "执行",
                  "yes", "y", "ok", "confirm", "confirmed", "approve")
# "可以"按子串命中，所以"不可以"会被判成批准；这类否定式短语要靠拒绝词表显式覆盖
_REJECT_WORDS = ("拒绝", "取消", "不同意", "不通过", "驳回", "不要", "no", "n", "reject", "deny")
# 拒绝词表比批准词表短：漏词只会落到"歧义默认拒绝"，不会放行，所以两侧宽严刻意不对称

# 英文审批语按整词切分用
_WORD_SPLIT = re.compile(r"[^a-z0-9]+")   # 非字母数字都当分隔符；"y1"、"ok?" 里的粘连串不会被切开

# 用户明确开口要人工
HANDOFF_KEYWORDS = ("转人工", "人工客服", "找客服", "找人工", "人工处理", "真人")
# 只有这六个字面量："有人工吗""换个真人来"靠子串命中，"有没有活人"这类同义词不命中


def user_requests_human(question: str) -> bool:
    """用户是否明确要人工。当前用子串匹配；“不要转人工”也会命中，不具备否定语义理解。"""
    # or "" 兜住 state 里 user_query 缺失的 None，避免 in 运算抛 TypeError
    return any(kw in (question or "") for kw in HANDOFF_KEYWORDS)


def should_handoff(question: str, intent_agent: str = "") -> bool:
    """是否需要转人工的唯一判定口径。

    Agent 的实际决策和离线评测都走这里，避免两边各写一套规则对不上。
    投诉域本身就要求人工跟进（状态标志本身不证明工单已建），不必等用户开口。
    """
    # 字面量比 AGENT_NAME：投诉 Agent 改名后这里静默失效，不会报错
    # or 短路只看两件事；工具全失败、无知识等其它转人工理由由调用方自己补
    return user_requests_human(question) or intent_agent == "complaint"


def is_high_risk(tool_name: str) -> bool:
    """单个工具是否高危。注意只认工具名，不看参数（同一工具的任何调用都要审批）。"""
    # 成员判定不看参数：退款额度多大、改哪个地址都同样要走审批，也意味着查订单这类只读工具永不审批
    # 名单外的名字一律 False（放行），所以"忘记登记"就是绕过审批，本函数不是默认拒绝
    return tool_name in HIGH_RISK_TOOLS


def high_risk_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从本轮待执行工具里挑出高危的。

    返回的是"调用"本身（含参数），因为审批人需要看到工具名与参数；当前退款参数不含金额，金额由执行时查询订单确定。
    """
    # get("name","") 让畸形调用（缺 name）落到名单外=放行，同样是 fail-open
    # 保留原 dict 引用而非拷贝：调用方直接拿返回值取 arguments 落审批表
    return [tc for tc in tool_calls if is_high_risk(tc.get("name", ""))]


def pending_payload(calls: list[dict[str, Any]], agent_name: str, conversation_id: int) -> dict[str, Any]:
    """交给前端/审批人的确认单。参数一并给出，审批人看得见要动哪个订单。

    这个 dict 会：
    - 作为 interrupt() 的 payload 抛给调用层 → 前端渲染确认卡片；
    - 被 ensure_pending 落进 approval_requests.actions → 后台审批页展示。
    所以字段命名要同时照顾前端与后台，保持一致（tool/label/arguments）。
    """
    return {
        # 键名对齐 ConfirmRequest.agent：前端把这份 payload 原样回填审批接口即可定位
        "agent": agent_name,
        # 审批表幂等键的第二半，必须是已落库的真实会话号，否则恢复时找不到 pending 行
        "conversation_id": conversation_id,
        # 用带时区的 ISO 时间给前端展示；注意这与数据库里的 naive UTC 是两套表示
        # timespec 去掉微秒：确认单只给人看秒，超时判定用的是库里的 requested_at，不读这个字段
        "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actions": [
            # c["name"] 用直接下标：calls 已由 high_risk_calls 过滤，名字必在名单内；
            # 这里若换成 .get() 会把名单漏登的工具静默渲染成无 label 的空动作
            {"tool": c["name"], "label": HIGH_RISK_TOOLS[c["name"]], "arguments": c.get("arguments", {})}
            for c in calls
        ],
        # arguments 原样透出（含订单号）：审批人看不到参数就无法判断风险，但这只是展示，不作为执行入参
        "prompt": "以下操作有风险，需人工确认后才执行。请回复\"确认\"或\"拒绝\"。",
        # 这句会被 _respond 当作 reply 落库；文案里的"确认/拒绝"必须能被上面的两张词表命中
    }


def parse_decision(resume_value: Any) -> tuple[bool, str]:
    """审批结果归一化。

    既接受结构化 {"approved": true/false, "note": ...}，也接受人工直接敲的自然语言。
    返回 (是否批准, 备注)。
    """
    if isinstance(resume_value, dict):
        # 结构化分支；list、对象等非 dict 一律走最后的字符串分支，靠 str() 兜住
        # 先检查是否真实 bool，避免用 bool("false") 把拒绝变批准。
        approved = resume_value.get("approved")   # 键缺失得到 None 而不是 False，所以只能靠 isinstance 判
        # note 只归一 None→""，不做真假判断："false"、"0" 这类文本备注要原样留给审计
        note = str(resume_value.get("note", "") or "")
        if isinstance(approved, bool):
            # 正常路径：后台/前端传的就是布尔
            return approved, note
        # 兼容 {"approved": "同意"} 这种把布尔写成文字的情况
        # approved 缺失时 str(None) 非空，不会回退到 note；自然语言兼容只覆盖部分表达。
        approved, text = _from_words(str(approved) or note)
        return approved, note
    # 用户直接回了一句"确认"/"拒绝"
    # 公共 HTTP 确认接口实际接收 Pydantic bool；自由文字解析主要用于直接调用 resume 或测试。
    # None 会变成空串，走不到任何词表，最终按歧义=拒绝处理
    return _from_words(str(resume_value or ""))


def _from_words(text: str) -> tuple[bool, str]:
    """自然语言 → 布尔。判定顺序：先拒绝词，再批准词，都不中→拒绝。"""
    lowered = text.strip().lower()   # 只影响英文；中文原样保留，所以词表里的英文必须写成小写
    # 英文词按整词匹配：否则 "n" 会命中任何含字母 n 的句子（"confirmed" 被当成拒绝）
    # 用集合去重并过滤空 token，多词句子的查找才是 O(1)
    tokens = {t for t in _WORD_SPLIT.split(lowered) if t}

    # 两侧共用的命中函数：批准与拒绝走同一套匹配规则，优先级完全由下面的调用顺序决定
    def hit(words: tuple[str, ...]) -> bool:
        for word in words:      # 同侧命中第一个词就返回，结果都是 True，词表内部顺序无影响
            # 用 isascii() 区分中英文两套匹配方式；词表若混入全角字母或带重音的词会走错分支
            if word.isascii():
                if word in tokens:      # 英文：必须是独立单词
                    return True
            elif word in lowered:       # 中文：子串即可
                return True
        return False        # 未命中不等于拒绝，由调用方决定（两个调用点都按拒绝处理）

    # 先判拒绝：混合表述（"同意退款不通过"）必须落在拒绝侧，宁可少做不可错做
    # 词典拒绝优先，但词表不完整，例如“不同意”已覆盖、“不确认”未覆盖；不能当自然语言安全证明。
    if hit(_REJECT_WORDS):
        return False, text
    if hit(_APPROVE_WORDS):
        return True, text
    # 语义不明时一律不执行高危操作，转人工
    return False, text
