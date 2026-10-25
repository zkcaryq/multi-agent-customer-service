"""统一的工具执行管线：注册检查 → 入参校验 → 熔断 → 超时 → 退避重试 → 记录成败。

三个 Agent 的 call_tools 节点都调这一个函数，策略改动只需改这里。
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import Any

from app.core.resilience import circuit_check, retry_with_backoff
from app.tools.base import ToolResult
from app.tools.registry import ToolSpec, tool_registry, validate_arguments


def _fail(name: str, error: str, t0: float, **extra: Any) -> dict[str, Any]:
    out = {
        "tool": name,
        "success": False,
        "error": error,
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }
    out.update(extra)
    return out


async def _call_handler(spec: ToolSpec, arguments: dict[str, Any]) -> ToolResult:
    handler = spec.handler
    if inspect.iscoroutinefunction(handler):
        return await handler(**arguments)
    return await asyncio.to_thread(functools.partial(handler, **arguments))


async def invoke_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行一个工具，永远返回结构化结果，不向调用方抛异常。

    context 是运行时可信参数（如 user_id），不会出现在给模型的 schema 里。
    """
    t0 = time.monotonic()

    spec = tool_registry.get(tool_name)
    if spec is None:
        return _fail(tool_name, "工具未注册", t0)

    normalized, errors = validate_arguments(spec, arguments or {}, context)
    if errors:
        return _fail(tool_name, "入参校验失败: " + "; ".join(errors), t0, validation_errors=errors)

    breaker = circuit_check(tool_name)
    if breaker.is_open():
        return _fail(tool_name, "熔断中，请稍后重试", t0, degraded=True)

    timeout_s = max(spec.definition.timeout_ms, 1) / 1000.0

    async def attempt() -> ToolResult:
        # 超时算一次失败，交给外层退避重试；重试耗尽后计入熔断
        return await asyncio.wait_for(_call_handler(spec, normalized), timeout=timeout_s)

    try:
        result = await retry_with_backoff(
            attempt, max_retries=spec.definition.retry_count, base_delay=1.0
        )
    except asyncio.TimeoutError:
        breaker.record_failure()
        return _fail(tool_name, f"工具调用超时（>{spec.definition.timeout_ms}ms）", t0)
    except Exception as exc:
        breaker.record_failure()
        return _fail(tool_name, str(exc), t0)

    breaker.record_success()
    return {
        "tool": tool_name,
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }
