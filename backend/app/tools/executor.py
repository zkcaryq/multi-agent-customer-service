"""统一的工具执行管线。

一条工具调用要穿过 6 道关卡，全部集中在这里，业务 Agent 只管"调哪个工具"：

    注册检查 → 入参校验 → 熔断 → 超时 → 退避重试 → 记录成败

【为什么必须集中】
四个域 Agent、9 个自检脚本、后台"工具测试"如果要各自实现一遍超时和重试，
迟早出现"Agent 里有超时、脚本里没有"的偏差。集中之后：
- 改策略只改这一个文件；
- 测试可以直接调 `invoke_tool` 验证策略，不需要跑整个 Agent。

【主要契约：预期调用失败转结构化结果】
注册缺失、校验报错、熔断和 handler 的常规异常转成结果字典。
校验函数自身异常、非法返回对象、任务取消等仍可能传播，不能承诺“永不抛异常”。
调用方（`base_agent._call_tools`）按顺序收集正常返回的结果；
模型也能从 `success=false` 里看到失败原因，进而决定"换个工具/转人工"。
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
    """统一的失败结果。extra 用来附加机器可读的细节（如 validation_errors、degraded）。"""
    out = {
        "tool": name,
        "success": False,
        "error": error,
        # monotonic 不受系统校时影响；执行器总耗时包括各次尝试和退避等待。
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }
    out.update(extra)
    return out


async def _call_handler(spec: ToolSpec, arguments: dict[str, Any]) -> ToolResult:
    """执行 handler，兼容同步/异步两种签名。

    同步 handler 丢到线程池（asyncio.to_thread）：否则一个阻塞的同步函数
    会卡死整个事件循环，所有并发请求一起变慢。
    """
    handler = spec.handler
    if inspect.iscoroutinefunction(handler):
        return await handler(**arguments)
    # partial 在这里预绑定参数；to_thread 本身也支持 *args/**kwargs，这不是它的限制。
    # 取消等待不能强行终止已启动的线程；同步写工具超时后可能仍执行，重试必须考虑业务幂等。
    return await asyncio.to_thread(functools.partial(handler, **arguments))


async def invoke_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行一个工具，把预期失败和 handler 的常规异常转成结构化结果。

    context 是运行时可信参数（如 user_id、conversation_id），不会出现在给模型的
    function schema 里 —— 校验层显式拒绝模型提供的上下文字段；真实用户认证和资源授权仍由入口/handler 负责。
    """
    t0 = time.monotonic()

    # ① 注册检查：模型可能幻觉出一个不存在的工具名（甚至在脏输出里给出 "[]"），
    #    这里直接判失败，而不是 KeyError
    # 这里只按全局名字取工具，没有 agent_id 授权参数；按域导出 schema 不等于执行端访问控制。
    spec = tool_registry.get(tool_name)
    if spec is None:
        return _fail(tool_name, "工具未注册", t0)

    # ② 入参校验：未声明参数、缺必填、类型不符、伪造可信上下文，全部在这里拦下。
    #    返回 (归一化后的参数, 错误列表)；有错误就不执行 handler。
    # 校验发生在 try 外；若传入非字典等意外类型，可能直接抛异常，调用协议仍有前置假设。
    normalized, errors = validate_arguments(spec, arguments or {}, context)
    if errors:
        return _fail(tool_name, "入参校验失败: " + "; ".join(errors), t0, validation_errors=errors)

    # ③ 熔断：连续失败达到阈值后直接快速失败，不再打下游。
    #    否则下游一直超时，会拖慢每一个请求（每轮都要等满超时）。
    breaker = circuit_check(tool_name)
    if breaker.is_open():
        return _fail(tool_name, "熔断中，请稍后重试", t0, degraded=True)

    # 超时值来自 ToolDefinition（默认取配置 TOOL_TIMEOUT_MS），不是写死的常量。
    # max(..., 1) 防呆：配成 0 或负数时退化为 1ms，而不是永远立即超时。
    timeout_s = max(spec.definition.timeout_ms, 1) / 1000.0

    async def attempt() -> ToolResult:
        # 超时算一次失败，交给外层退避重试；重试耗尽后计入熔断
        # 单次尝试超时；wait_for 会等待取消完成，所以不是整个调用绝不超过 5 秒的硬保证。
        return await asyncio.wait_for(_call_handler(spec, normalized), timeout=timeout_s)

    try:
        # ④⑤ 退避重试 + 超时：retry_with_backoff 内部按 1s→2s→4s 退避，
        #    每次尝试都受 wait_for 的超时约束。retry_count 来自工具定义/配置。
        result = await retry_with_backoff(
            attempt, max_retries=spec.definition.retry_count, base_delay=1.0
        )
    except asyncio.TimeoutError:
        # 全部重试都超时：记一次熔断失败，返回可读原因（带具体阈值，便于排障）
        breaker.record_failure()
        return _fail(tool_name, f"工具调用超时（>{spec.definition.timeout_ms}ms）", t0)
    except Exception as exc:  # noqa: BLE001
        # handler 内部抛异常（网络错误、SQL 错误…）：同样计入熔断
        breaker.record_failure()
        return _fail(tool_name, str(exc), t0)

    # ⑥ 业务级失败（handler 正常返回但 result.success=False）不算熔断失败：
    #    比如"订单不属于当前用户"是预期内的业务结果，不是系统故障。
    # 业务失败也清零系统失败计数；熔断统计的是技术调用是否抛错，而非业务是否办理成功。
    breaker.record_success()
    return {
        "tool": tool_name,
        "success": result.success,
        "data": result.data,
        "error": result.error,
        # monotonic 不受系统校时影响；执行器总耗时包括各次尝试和退避等待。
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }
