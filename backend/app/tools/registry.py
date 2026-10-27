"""工具注册表与入参校验。

两件事：
1. `ToolRegistry`：进程内的工具登记处。Agent 启动时注册，运行时按 agent_id 取用。
2. `validate_arguments`：把模型给的"野生参数"校验成"可执行参数"，并注入可信上下文。

【为什么工具要有注册表，而不是 Agent 里直接 import 函数】
- 模型需要 function schema（名称/描述/参数类型）才能决定调哪个工具，schema 由定义生成；
- 后台"工具管理"页要列出全部工具；离线评测要统计工具调用；
- 四个 Agent 需要按域隔离工具（售前不该看到 request_refund）。
这些需求都要求"工具是一等公民"，不能只是散落的函数。

【为什么注册键是 name:version】
同名工具升级参数时，旧会话可能还在用旧版本。带版本号允许并存，
`get(name)` 默认取 1.0，是刻意保留的简单默认。
"""

from __future__ import annotations  # 注解延迟求值，本文件的 int | None 等写法在定义期不求值

import json  # 当前模块没有用到它（schema 生成不做序列化），保留导入是历史遗留
from dataclasses import dataclass, field  # 本文件只用到 dataclass，field 目前是未使用的导入（ToolSpec 三个字段都是普通默认值）
from typing import Any, Callable  # Callable 描述 handler 签名，只做类型提示，不做运行时签名检查

from app.tools.base import ToolDefinition, ToolResult  # 定义（数据）与结果（数据）；校验函数在本文件，执行在 executor


@dataclass
class ToolSpec:
    """一条注册记录：定义（给模型看）+ 可执行函数（给服务端用）+ 归属域。

    agent_id=None 表示公共工具（如 web_search、CRM 查询），所有域都能用。
    """

    definition: ToolDefinition  # 校验与 schema 的唯一真源；handler 的形参名要和它一致，否则执行时 TypeError
    handler: Callable[..., ToolResult]  # 同步或异步都允许，差异由 executor._call_handler 抹平
    agent_id: int | None = None  # 只影响 to_openai_tools 的导出范围；invoke_tool 不读它，所以不是执行端权限


class ToolRegistry:
    """Central registry for all tools. Agent startup pulls its tools from here."""

    def __init__(self) -> None:
        # 用 "name:version" 作为 key：同名不同版本可以并存
        # 字典序＝注册顺序，list_for_agent 的返回顺序因此由 load_all_agents 的调用顺序决定，不是稳定契约
        # 只属于单个进程：多 worker 各自注册一份，后台看到的工具数不会跨进程合并
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: Callable[..., ToolResult],
        agent_id: int | None = None,
    ) -> None:
        """登记一个工具。幂等：同 name+version 重复注册会覆盖（不会报错）。

        覆盖而不是报错，是为了让 `load_all_agents()` 在测试/重载场景反复调用也安全。
        """
        # 版本只存在注册键中；function schema 不带版本，执行器 get(name) 默认仍取 1.0。
        key = f"{definition.name}:{definition.version}"
        # 整体替换 ToolSpec：definition 与 handler 必须成对更新，只换其中一个会让 schema 和执行不一致
        self._tools[key] = ToolSpec(
            definition=definition, handler=handler, agent_id=agent_id
        )

    def get(self, name: str, version: str = "1.0") -> ToolSpec | None:
        """按名字取工具；不存在返回 None（调用方负责转成"工具未注册"的结果）。"""
        # 只有 name+默认版本两个条件，既不看 agent_id 也不看调用方是谁；这是"按域导出 ≠ 执行端授权"的根因
        return self._tools.get(f"{name}:{version}")

    def list_for_agent(self, agent_id: int) -> list[ToolSpec]:
        """本域可见的工具 = 公共工具（agent_id=None）+ 本域专属工具。"""
        # 这里才是域隔离的唯一实现处：只作用于给模型的 schema，执行路径 invoke_tool 完全不经过它
        # 同 name 多版本会在这里各导出一份，schema 数组里出现两个同名 function（当前只注册 1.0 所以没暴露）
        return [t for t in self._tools.values() if t.agent_id is None or t.agent_id == agent_id]

    def list_all(self) -> list[ToolSpec]:
        """全部工具，后台"工具管理"页用它（当前 17 个）。"""
        # 17 = 私有 14（售前 4 / 售中 4 / 售后 4 / 投诉 2）+ 公共 3（CRM 两个 + web_search）
        # web_search 被三个域各注册一次，同 key 覆盖成一条，所以不会数出 19
        return list(self._tools.values())

    def to_openai_tools(self, agent_id: int) -> list[dict[str, Any]]:
        """把工具定义导出成 OpenAI 兼容的 function schema。

        注意：**只导出 parameters，不导出 context_params** ——
        user_id / conversation_id 这类可信参数由运行时注入，
        不给模型正常填这些字段的入口；恶意或错误输出仍可能包含它们，由 validate_arguments 拒绝。
        """
        tools: list[dict[str, Any]] = []
        for spec in self.list_for_agent(agent_id):
            # 这些字段描述模型可填写的 JSON 对象；真正执行前仍走本地校验，不能只信模型遵循 schema。
            params: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for p in spec.definition.parameters:
                # 只导出 type 和 description：ToolParameterDef 没有 enum/min/max/pattern 字段，
                # 所以"只能是电子发票/纸质发票"这类约束模型收不到，只能在 handler 里兜住或靠文案提示
                params["properties"][p.param_name] = {
                    "type": p.param_type,
                    "description": p.param_desc or "",
                }
                # required 是名字数组，与 properties 分开维护；两者必须同一循环里配对，否则会声明出不存在的字段
                if p.is_required:
                    params["required"].append(p.param_name)
            # 没有必填项时删掉 required 字段：部分供应商对空数组会报 schema 非法
            if not params["required"]:
                # 无参数工具（CRM 两个、query_unpaid_order、query_my_complaints）就落在这条分支，模型 arguments 给 {}
                del params["required"]
            tools.append({
                "type": "function",
                "function": {
                    # 不导出 version/display_name/timeout_ms/context_params：模型只需名字和描述，
                    # 反过来 context_params 一旦导出，模型就有了正常填 user_id 的入口，注入防线就废了
                    "name": spec.definition.name,
                    "description": spec.definition.description,
                    "parameters": params,
                },
            })
        return tools


# param_type → Python 转换器。用于把模型常见的"字符串数字"归一化。
# 不在表里的类型原样透传；object/array 在表内，但这里只做 Python 转换，不递归检查子字段。
_TYPE_COERCE = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(
    spec: ToolSpec,
    arguments: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """按 ToolDefinition 校验并归一化模型给出的参数，再注入可信上下文。

    返回 (可用参数, 错误列表)。错误非空就不该执行 handler——模型幻觉出未知参数、
    漏填必填项、把数字写成字符串、试图伪造 user_id，都属于这一层要拦住的。

    四类校验，顺序很重要：
    ① 未声明参数 → ② 伪造可信参数 → ③ 逐个声明参数做类型/必填 → ④ 注入上下文
    """
    errors: list[str] = []
    # `or {}` 而不是 `if context is None`：调用方不传上下文时按"无任何可信参数"处理，
    # 后面第 ④ 步会把它缺什么报出来，而不是在取值时抛 AttributeError
    context = context or {}
    # 声明参数：模型"可以"提供的
    # 参数定义是校验真源；业务 handler 的 Python 默认值只在对应参数未传入时起效。
    declared = {p.param_name: p for p in spec.definition.parameters}
    # 可信参数：只能由系统注入的
    # “可信”是相对模型输出而言；必须由入口认证用户后注入，当前聊天入口仍缺这一步。
    trusted = set(spec.definition.context_params)
    # 只收集校验通过的字段；handler 最终只拿到这个 dict，模型原始 arguments 不会透传
    normalized: dict[str, Any] = {}

    # ① 模型多给了不存在的参数（幻觉/旧 schema）：直接判错，不静默丢弃。
    #    静默丢弃会让"模型以为传了、实际没传"的 bug 很难查。
    # 同一个伪造字段可能同时触发 unknown 与 forged，两条报错不表示执行了两遍。
    # 用列表推导保留 arguments 的键序（可能重复的键由 dict 语义天然去重），输出时再 sorted 保证文案稳定
    unknown = [k for k in arguments if k not in declared]
    if unknown:
        errors.append(f"未声明的参数: {', '.join(sorted(unknown))}")

    # ② 模型试图自己提供 user_id：这是越权尝试，必须显式报错（而不是覆盖后放行）
    # 只和 context_params 声明取交集；没声明成可信参数的字段（如 check_membership 的 user_id）走不到这里
    forged = sorted(set(arguments) & trusted)
    if forged:
        errors.append(f"这些参数由系统注入，模型不能提供: {', '.join(forged)}")

    # ③ 逐个检查声明参数：缺必填 / 用默认值 / 类型转换
    # 遍历 declared（不是 arguments）：一次把"缺失"和"取值"两件事都覆盖，多余字段已在 ① 报过
    for name, p in declared.items():
        # 空字符串视同"没填"：模型经常给 order_id="" 这种占位
        # 只把 None 和 "" 当空：0、False、[] 都会当"填了"继续走转换，所以 order_id=0 不会报缺失
        if name not in arguments or arguments[name] in (None, ""):
            if p.is_required:
                errors.append(f"缺少必填参数 {name}（{p.param_desc}）")
            elif p.default_value is not None:
                # 默认值在此直接使用，不经过下面 caster；声明默认值时需要保证与 handler 类型匹配。
                normalized[name] = p.default_value
            # 非必填且没声明默认值就不写入：让 handler 的 Python 形参默认值生效（两套默认值，勿混淆）
            continue

        value = arguments[name]
        caster = _TYPE_COERCE.get(p.param_type)
        if caster is None:
            # param_type 拼错（如写成 "int"）时不报错，原样透传：静默降级，只能靠注册时的自检发现
            normalized[name] = value
            continue
        # 布尔要单独处理："true"/"false"/"1"/"0" 这类字符串先转成 bool，
        # 否则 bool("false") 会得到 True（非空字符串恒真）
        # 已知 true/false 文本被处理；其他非空文本会落到 bool(value) 变成 True，不是严格布尔校验。
        if caster is bool and isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                value = True
            elif value.lower() in ("false", "0", "no"):
                value = False
        # 到这里值一定是 bool（原本就是，或由上一分支转换来），直接写入并跳过通用 caster
        if caster is bool and isinstance(value, bool):
            normalized[name] = value
            continue
        try:
            # int(1.9) 会截成 1、list("ab") 会变字符列表；这是宽松转换，不等于完整 JSON Schema 验证。
            normalized[name] = caster(value)
        except (TypeError, ValueError):
            # 只记错误、不写入 normalized：handler 拿不到这个字段，会走它自己的 Python 默认值，
            # 但 errors 非空时执行器根本不会调 handler，所以这条路径只是给模型看的说明
            errors.append(f"参数 {name} 类型错误：期望 {p.param_type}，实际 {value!r}")

    # ④ 注入可信上下文；缺了就报错（不能静默传空串给下游，那会变成越权或脏数据）
    # 放在最后：先把模型能给的东西验完，再补模型不该给的东西，顺序调换会让伪造检查失去意义
    # trusted 是 set，遍历顺序不稳定；多条上下文都缺时，报错文案里的先后顺序会变（不影响判定）
    for name in trusted:
        # 同样只拦 None 和 ""：conversation_id=0 会被当成有效值注入，边界要 handler 自己挡
        if context.get(name) in (None, ""):
            errors.append(f"缺少系统上下文 {name}")
        else:
            # 注入值不会重新验证身份或类型；订单归属检查还在 handler，HTTP 认证也需独立完成。
            normalized[name] = context[name]

    # 四道检查都不提前 return，全部跑完再一次性返回：模型能同轮看到所有问题并一次改对
    return normalized, errors


# 模块级单例：全进程共用一张表。工具在 Agent 构造时注册（幂等）
# executor 和各 Agent 都 import 同一个对象而非构造注入，所以测试里想重置只能直接清 _tools，没法换一个干净实例
tool_registry = ToolRegistry()
