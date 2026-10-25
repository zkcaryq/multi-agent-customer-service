from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from app.tools.base import ToolDefinition, ToolResult


@dataclass
class ToolSpec:
    """Internal representation of a registered tool callable."""

    definition: ToolDefinition
    handler: Callable[..., ToolResult]
    agent_id: int | None = None


class ToolRegistry:
    """Central registry for all tools. Agent startup pulls its tools from here."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: Callable[..., ToolResult],
        agent_id: int | None = None,
    ) -> None:
        key = f"{definition.name}:{definition.version}"
        self._tools[key] = ToolSpec(
            definition=definition, handler=handler, agent_id=agent_id
        )

    def get(self, name: str, version: str = "1.0") -> ToolSpec | None:
        return self._tools.get(f"{name}:{version}")

    def list_for_agent(self, agent_id: int) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.agent_id is None or t.agent_id == agent_id]

    def list_all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def to_openai_tools(self, agent_id: int) -> list[dict[str, Any]]:
        """Export tools as OpenAI-compatible function definitions."""
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
    """
    errors: list[str] = []
    context = context or {}
    declared = {p.param_name: p for p in spec.definition.parameters}
    trusted = set(spec.definition.context_params)
    normalized: dict[str, Any] = {}

    unknown = [k for k in arguments if k not in declared]
    if unknown:
        errors.append(f"未声明的参数: {', '.join(sorted(unknown))}")

    forged = sorted(set(arguments) & trusted)
    if forged:
        errors.append(f"这些参数由系统注入，模型不能提供: {', '.join(forged)}")

    for name, p in declared.items():
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

    for name in trusted:
        if context.get(name) in (None, ""):
            errors.append(f"缺少系统上下文 {name}")
        else:
            normalized[name] = context[name]

    return normalized, errors


# Global singleton
tool_registry = ToolRegistry()
