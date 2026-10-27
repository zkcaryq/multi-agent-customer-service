"""意图分类与领域路由。

两层策略（先准后稳）：
  1. LLM 分类（岗位 `intent`，模型由配置决定，temperature=0）—— 主路径，可理解本次消息的改写；当前没有传入历史对话
  2. 关键词规则 —— 兜底：模型超时/限流/返回非法 JSON 时仍能分诊

外加一条**确定性路由守卫**（`_guard_routing`）：模型偶尔把"我要退款，商品和描述不符"
判成投诉域，一旦判错，下游会去建投诉工单而不是走退款流程。这类错误不能靠调提示词碰运气，
用规则硬纠。

【为什么分类不放在 Agent 图里】
分类是所有领域共用的前置步骤，放进任一领域的图都会导致其余三个域重复实现；
而且它决定"进哪张图"，天然属于图之前的一层。
"""

from __future__ import annotations  # 注解延迟求值，让 IntentResult | None 这种写法不依赖运行时类型

import json  # 解析模型返回的 JSON 文本；语法合法不等于字段和取值可用，字段仍要逐个清洗
from dataclasses import dataclass  # IntentResult 用普通 dataclass：不做运行时校验、不裁剪取值范围

# ── Intent result ────────────────────────────────────────────────────────────

# 合法目标集中在一处：新增领域时这里改一行，下面的校验和兜底都跟着生效
# 用 tuple 不用 list：模块级常量不该被运行期改写；这里只做成员判断，元素顺序不承载优先级
VALID_TARGETS = ("pre_sales", "during_sales", "after_sales", "complaint")


@dataclass
class IntentResult:
    """一次分诊的结论。

    classifier 字段很关键：排查"为什么答得不对"时，先看这次是模型判的还是关键词判的。
    """
    intent_name: str      # 模型自报的意图标签，无枚举校验；只进日志和 trace，不参与任何路由判断
    target_agent: str  # one of VALID_TARGETS  # 唯一真正决定执行的字段：route_to_agent、协作候选、会话归属都读它
    confidence: float     # 两条来源口径不同（LLM 自报 vs 关键词命中比例），不可跨 classifier 比较，也没做 0..1 裁剪
    classifier: str = "keyword"  # "llm" | "keyword"
    # 默认取 keyword 而不是 llm：手工构造（单测/脚本）没标来源时，宁可低估可信度，别让一条规则结果看起来像模型判定


# ── LLM classification prompt ────────────────────────────────────────────────

# 提示词的三段结构：角色与候选 → 判定原则 → 输出格式。
# "拿不准时归 after_sales" 是刻意写的：售后能兜住退换/物流/退款，
# 误判到售前会导致用户问退款却被推销商品。
# 这段文本里只有 target_agent 会被代码校验；意图名和 confidence 全靠模型自律，改域名必须同步 VALID_TARGETS。
# "问候语归 pre_sales" 意味着"你好"也会占用一次售前图（检索+生成），并不存在闲聊旁路。
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

# Few-shot 不是随便挑的：每个域 3-6 条，覆盖"典型说法 + 容易混淆的边界"。
# 例如"收到的东西是坏的"放在售后，而"东西是假货，我要求赔偿"放在投诉——
# 二者的差别是"有没有索赔/维权诉求"，模型靠这几组对照学会边界。
# 每条是 (user 消息, assistant 消息) 二元组，assistant 侧必须是合法 JSON：这里的键名写错不会报错，只会把错误示范教给模型。
# 代价：样例每次分类都全量重放（没有 prompt 缓存），这部分 token 和延迟是每条请求的固定开销。
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
        from app.providers import get_chat   # 延迟导入：避免 agents 与 providers 循环依赖

        messages: list[dict] = [
            {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
        ]
        # 规则与候选域先入上下文，样例才有解释对象；system 放后面会让模型把样例当独立问答
        # few-shot 用标准的 user/assistant 对话形式喂进去，比塞在 system 里更稳
        for user_msg, assistant_msg in _FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": user_query})
        # 本次待判消息放在全部样例之后：样例给出对照，最后一条才是要回答的
        # 这里不做长度裁剪，用户消息再长也整条送进；也没有多轮历史（跨轮省略语无法理解）

        # get_chat() 返回进程级客户端；未配置/凭据错误在这里抛 ProviderError，被下面的 except 吞成 None
        # 所以意图岗位挂掉的表现是"退化成关键词分类"，而不是 5xx——这条降级是静默的
        result = await get_chat().chat(
            role="intent",        # 岗位 → 小模型（分类不需要大模型，省钱又低延迟）
            messages=messages,
            temperature=0.0,      # 降低采样随机性；temperature=0 也不保证不同请求完全一致
            max_tokens=128,       # 只要一小段 JSON
            json_mode=True,       # 请求 JSON 输出；对象类型、字段和值仍需要业务侧校验
        )
        # result.content 为空（超长输入被截断、只返回 tool_calls）时这里抛 JSONDecodeError，走兜底
        # 只保证解析成某种 JSON 值；合法数组、null 也能解析成功，后面的 .get 可能抛错。
        data = json.loads(result.content)
    except Exception:
        # 超时、限流、返回非 JSON、模型未配置……全部走兜底。
        # 这里不记日志是原实现的取舍；排查时可在此加 log.warning。
        # 覆盖面过宽也意味着代码 bug（AttributeError/TypeError）同样被吞掉，只看结果分不出是模型差还是代码坏。
        return None

    # 下面这段已经在 try 之外：清洗阶段的异常不会再被关键词兜底接住，会一路冒到接口层
    # 这行已在 try 块外；当前非对象 JSON 不会走关键词兜底，是已知输入校验缺口。
    target = data.get("target_agent", "pre_sales")  # 字段缺失也落售前：和另外两处回落同向，"模型没说完话"时行为可预测
    if target not in VALID_TARGETS:
        # 模型幻觉出不存在的 Agent 名：宁可回落到售前，也不要让下游 KeyError
        target = "pre_sales"
        # 回落是无条件静默的，trace.classifier 仍然写 llm，看日志会以为是模型判对/判错，看不出这里被改过

    return IntentResult(
        # 缺省意图名沿用样例里售前的粗标签；样例只教出 query_product/place_order/check_logistics/complaint 四种值，
        # 所以 intent_name 实际上就是域名的另一种写法，粒度并不比 target_agent 细
        intent_name=data.get("intent_name", "query_product"),
        target_agent=_guard_routing(user_query, target),   # ← 确定性纠偏，见下
        # 这是模型自报分数，未校准也未做 0..1 范围检查；不可作为统计准确率。
        confidence=float(data.get("confidence", 0.7)),
        # float() 在 try 外：模型返回 "confidence": "高" 这类非数字值会抛 ValueError 直接 500，不会降级到关键词
        # 0.7 是人工经验值（"大方向可信但不保证"），只被写入 trace，没有任何代码按它做分支
        classifier="llm",
    )


# 用户明确要退款/退货/换货时，即使提到"商品和描述不符"也不该落到投诉域：
# 一旦判成 complaint，Agent 会去建投诉工单，退款流程整个跑偏。
# 两张表都比 _INTENT_RULES 短：守卫只做"否决"，不承担分类；词表越窄，改动越少误伤。
_AFTER_SALES_ACTIONS = ("退款", "退货", "换货", "退钱")  # 办事类动词：指向退款流程而不是建单
_COMPLAINT_WORDS = ("投诉", "赔偿", "维权", "道歉", "举报", "起诉", "欺诈", "假货", "315", "曝光")
# 这里的"315"是子串，所以"12315"也能命中；但"律师""市场监管""消协"只在 _INTENT_RULES 里有，守卫不认识它们


def _guard_routing(user_query: str, target: str) -> str:
    """只纠正"投诉 → 售后"这一种误判，其余目标原样返回。

    为什么只用规则改这一种？
    别的方向（售前↔售中、售前↔售后）误判的代价是"答非所问"，用户重问一句即可；
    而"退款被判成投诉"会真的走错业务流程（建单而非退款），代价高，必须兜住。
    """
    if target != "complaint":
        # 单向纠偏：规则从不把别的域改成 complaint，也从不把 complaint 改成第三个域
        # 原因是"升级成投诉"意味着要建单、要真人跟进，这个决定权只留给模型，代码没有资格加码
        return target
    wants_refund = any(k in user_query for k in _AFTER_SALES_ACTIONS)
    # 纯子串匹配，没有否定语义："我不想退款了"同样算 wants_refund
    wants_complaint = any(k in user_query for k in _COMPLAINT_WORDS)
    # 两个条件必须同时成立才纠偏：只要用户喊过投诉类词就尊重模型判定，规则只在"要办事、没喊维权"时才降级
    # 例："东西和描述不符，我要退款" → after_sales；"东西是假货，要求赔偿" → 仍留在 complaint
    return "after_sales" if wants_refund and not wants_complaint else target


# ── Keyword fallback (kept from original implementation) ──────────────────────

# 关键词表：每条是（词表, 意图名, Agent 名）。
# 打分方式见 _classify_keyword：命中数 / 词表长度，偏向"短词表且全中"的类别。
# 一处两读：_classify_keyword 用它兜底分类，collaboration_candidates 用它挑辅助域——新增域只改这里就同时生效。
# 元素是可变 list（不是 tuple），图的是字面量好写；运行期没人改它，也没有防御性拷贝。
_INTENT_RULES: list[tuple[list[str], str, str]] = [
    (
        # 放在第一条：同分时靠前的域胜出（见下面的严格大于），把最高风险域排第一是刻意的
        ["投诉", "维权", "赔偿", "道歉", "315", "12315", "工商", "市场监管", "消协",
         "曝光", "媒体", "起诉", "律师", "欺诈", "假货", "造假", "态度差", "骗人"],
        # 18 个词的分母：只命中 1 个词时得分 1/18，高于 22 个词的售前/售后表——归一化其实偏袒短词表
        "complaint",
        "complaint",
    ),
    (
        ["商品", "有没有", "多少钱", "价格", "规格", "颜色", "内存", "配置",
         "对比", "哪个好", "推荐", "优惠", "优惠券", "促销", "打折", "活动",
         "会员", "积分", "权益", "库存", "有货", "有现货"],
        # "优惠券/促销/会员"也归这里：售前的职责含权益与活动咨询，不是只管商品参数
        "query_product",
        "pre_sales",
    ),
    (
        ["下单", "支付", "付款", "付了", "没成功", "地址", "改地址",
         "发票", "开票", "未支付", "取消订单", "修改订单",
         "花呗", "分期", "扣款", "扣了", "限额", "抬头"],
        # "地址""抬头"这类单词单独出现也算售中：改地址、开票都是订单未完成阶段的动作
        "place_order",
        "during_sales",
    ),
    (
        ["物流", "快递", "发货", "到哪了", "货到哪", "单号", "退货",
         "退款", "换货", "质量问题", "坏了", "退钱",
         "配送", "运费", "签收", "到货", "几天到",
         "订单详情", "查订单", "我的订单", "转人工", "人工客服"],
        # 兜底路径不经过 _guard_routing，"退款/退货/换货"在这里已经直接归售后，等效覆盖了守卫想纠的那类误判
        # "转人工/人工客服"也落售后而不是单独分支：进图后由 user_requests_human 在 generate 阶段打 handoff 标记
        # 副作用："查订单"这类本属售中的说法会被算进售后，两个域的分界只靠词表
        "check_logistics",
        "after_sales",
    ),
]


def _classify_keyword(user_query: str) -> IntentResult:
    """Simple keyword-based intent classification (fallback)."""
    # 三个 best_* 的初值本身就是"什么都没命中"时的答案：售前 + query_product + 0 分
    # 初值和 LLM 路径非法域名的回落方向一致，两条路径的"最坏情况"落在同一个域
    best_score = 0.0
    best_intent = "query_product"
    best_agent = "pre_sales"

    for keywords, intent_name, agent in _INTENT_RULES:
        # 归一化打分：命中数 / 词表长度。这样长词表不会因为"词多"而天然占优
        score = sum(1 for kw in keywords if kw in user_query) / len(keywords)
        # 子串匹配、不区分重叠："退货运费"同时命中"退货"和"运费"，两个词都计入售后表
        # 严格大于才替换，所以同分时规则表里靠前的领域胜出。
        if score > best_score:
            best_score = score
            # 三个值必须一起换：只换 agent 会让 intent_name 和执行域对不上，trace 里就成了假数据
            best_intent = intent_name
            best_agent = agent

    # 放大 3 倍再截断到 0.98：命中 1 个词通常已足够可信，但绝不报 1.0。
    # 一个词都没命中时给 0.5（相当于"我不确定"），但仍然要返回一个可用的默认路由。
    # 乘 3 只是把"比例分"压进看起来合理的区间；1/18 的单词命中放大后约 0.17，短消息的分数普遍偏低
    confidence = min(best_score * 3, 0.98) if best_score > 0 else 0.5
    return IntentResult(
        intent_name=best_intent,
        target_agent=best_agent,
        # 这里不调 _guard_routing：兜底路径靠词表把"退款/退货/换货"直接归售后，纠偏需求已经在词表里消化
        confidence=round(confidence, 2),  # 两位小数：去掉浮点尾噪（0.30000000000000004），也贴合 trace 列的定点精度
        classifier="keyword",
    )


# ── Public API ────────────────────────────────────────────────────────────────


async def classify_intent(user_query: str) -> IntentResult:
    """LLM 优先 + 关键词兜底的意图分类。"""
    # 仅本次 query 入参；对“那帮我退了吧”这种跨轮省略语，没有订单槽位或历史注入。
    # 入参是接口层脱敏后的文本（mask_sensitive 的结果），不是用户原话
    result = await _classify_llm(user_query)
    # 每条业务消息都要付一次模型调用：这里没有缓存、没有短路，也没有按长度跳过
    if result is not None:
        # None 是唯一的兜底信号：模型"成功返回但判错"不会走关键词，判错只能靠 _guard_routing 纠那一类
        return result
    return _classify_keyword(user_query)


def agent_by_name(name: str):
    """按 Agent 名取实例。审批回调只带名字，需要据此找回同一条 LangGraph 线程。

    函数内 import 而不是模块顶部 import：四个 agent 模块都会 import base_agent，
    顶部导入会形成 agents 包内的循环依赖。
    """
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

    例："订单到哪了？另外能开发票吗" → ["after_sales", "during_sales"]
    """
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for keywords, _intent_name, agent_name in _INTENT_RULES:
        if agent_name in seen:
            continue
        # 这里用"命中个数"而不是归一化比例：多领域比较时，命中词更多的领域更相关
        score = sum(1 for keyword in keywords if keyword in user_query)
        if score:
            scored.append((score, agent_name))
            seen.add(agent_name)

    # 按命中词数降序，稳定排序让同分保持规则表原顺序；不是 LLM 的置信度排序。
    scored.sort(key=lambda item: item[0], reverse=True)
    agents = [primary_agent]          # 主 Agent 恒第一，后面的才是辅助
    for _score, agent_name in scored:
        if agent_name not in agents:
            agents.append(agent_name)
        # 检查在 append 后：limit=1 时仍可能加进一个辅助域；不要把该参数说成已验证的硬上限。
        if len(agents) >= max(1, limit):
            break
    return agents
