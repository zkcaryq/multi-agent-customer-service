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

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from app.tools.base import ToolDefinition, ToolResult


@dataclass
class ToolSpec:
    """一条注册记录：定义（给模型看）+ 可执行函数（给服务端用）+ 归属域。

    agent_id=None 表示公共工具（如 web_search、CRM 查询），所有域都能用。
    """

    definition: ToolDefinition
    handler: Callable[..., ToolResult]
    agent_id: int | None = None


class ToolRegistry:
    """Central registry for all tools. Agent startup pulls its tools from here."""

    def __init__(self) -> None:
        # 用 "name:version" 作为 key：同名不同版本可以并存
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
        key = f"{definition.name}:{definition.version}"
        self._tools[key] = ToolSpec(
            definition=definition, handler=handler, agent_id=agent_id
        )

    def get(self, name: str, version: str = "1.0") -> ToolSpec | None:
        """按名字取工具；不存在返回 None（调用方负责转成"工具未注册"的结果）。"""
        return self._tools.get(f"{name}:{version}")

    def list_for_agent(self, agent_id: int) -> list[ToolSpec]:
        """本域可见的工具 = 公共工具（agent_id=None）+ 本域专属工具。"""
        return [t for t in self._tools.values() if t.agent_id is None or t.agent_id == agent_id]

    def list_all(self) -> list[ToolSpec]:
        """全部工具，后台"工具管理"页用它（当前 17 个）。"""
        return list(self._tools.values())

    def to_openai_tools(self, agent_id: int) -> list[dict[str, Any]]:
        """把工具定义导出成 OpenAI 兼容的 function schema。

        注意：**只导出 parameters，不导出 context_params** ——
        user_id / conversation_id 这类可信参数由运行时注入，
        模型看不到，也就无从伪造。
        """
        tools: list[dict[str, Any]] = []
        for spec in self.list_for_agent(agent_id):
            params: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for p in spec.definition.parameters:
                params["properties"][p.param_name] = {
                    "type": p.param_type,
                    "description": p.param_desc or "",
                }
                if p.is_required:
                    params["required"].append(p.param_name)
            # 没有必填项时删掉 required 字段：部分供应商对空数组会报 schema 非法
            if not params["required"]:
                del params["required"]
            tools.append({
                "type": "function",
                "function": {
                    "name": spec.definition.name,
                    "description": spec.definition.description,
                    "parameters": params,
                },
            })
        return tools


# param_type → Python 转换器。用于把模型常见的"字符串数字"归一化。
# 不在表里的类型（如自定义 object）原样透传。
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
    context = context or {}
    # 声明参数：模型"可以"提供的
    declared = {p.param_name: p for p in spec.definition.parameters}
    # 可信参数：只能由系统注入的
    trusted = set(spec.definition.context_params)
    normalized: dict[str, Any] = {}

    # ① 模型多给了不存在的参数（幻觉/旧 schema）：直接判错，不静默丢弃。
    #    静默丢弃会让"模型以为传了、实际没传"的 bug 很难查。
    unknown = [k for k in arguments if k not in declared]
    if unknown:
        errors.append(f"未声明的参数: {', '.join(sorted(unknown))}")

    # ② 模型试图自己提供 user_id：这是越权尝试，必须显式报错（而不是覆盖后放行）
    forged = sorted(set(arguments) & trusted)
    if forged:
        errors.append(f"这些参数由系统注入，模型不能提供: {', '.join(forged)}")

    # ③ 逐个检查声明参数：缺必填 / 用默认值 / 类型转换
    for name, p in declared.items():
        # 空字符串视同"没填"：模型经常给 order_id="" 这种占位
        if name not in arguments or arguments[name] in (None, ""):
            if p.is_required:
                errors.append(f"缺少必填参数 {name}（{p.param_desc}）")
            elif p.default_value is not None:
                normalized[name] = p.default_value
            continue

        value = arguments[name]
        caster = _TYPE_COERCE.get(p.param_type)
        if caster is None:
            normalized[name] = value
            continue
        # 布尔要单独处理："true"/"false"/"1"/"0" 这类字符串先转成 bool，
        # 否则 bool("false") 会得到 True（非空字符串恒真）
        if caster is bool and isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                value = True
            elif value.lower() in ("false", "0", "no"):
                value = False
        if caster is bool and isinstance(value, bool):
            normalized[name] = value
            continue
        try:
            normalized[name] = caster(value)
        except (TypeError, ValueError):
            errors.append(f"参数 {name} 类型错误：期望 {p.param_type}，实际 {value!r}")

    # ④ 注入可信上下文；缺了就报错（不能静默传空串给下游，那会变成越权或脏数据）
    for name in trusted:
        if context.get(name) in (None, ""):
            errors.append(f"缺少系统上下文 {name}")
        else:
            normalized[name] = context[name]

    return normalized, errors


# 模块级单例：全进程共用一张表。工具在 Agent 构造时注册（幂等）
tool_registry = ToolRegistry()
