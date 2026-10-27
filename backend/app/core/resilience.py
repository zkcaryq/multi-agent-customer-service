"""Retry with exponential backoff and circuit breaker for tool calls."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import settings


def is_retryable(error: Exception) -> bool:
    """网络/超时类可重试；业务错误不重试。

    asyncio.wait_for 抛的 TimeoutError 的 str() 是空串，只按消息匹配会漏掉，
    所以先按类型判，再看消息关键词。
    """
    # OSError 范围很广，某些本地错误也会被当成可重试；这只是当前简单策略，不是精确错误分类器。
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    msg = str(error).lower()
    retryable = ["timeout", "connection", "network", "unreachable", "refused", "reset"]
    # 字符串兜底兼容第三方异常，但可能误判；生产适配应按 SDK 异常类型与错误码收敛。
    return any(kw in msg for kw in retryable)


async def retry_with_backoff(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int | None = None,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Call fn with retry on retryable errors. Delay: 1s → 2s → 4s …"""
    # None 才取配置，0 表示不重试；这是“额外重试次数”，不是总尝试次数。
    retries = max_retries if max_retries is not None else settings.tool_retry_count
    # 保存最后一次错误，重试耗尽后交回执行器归一化。
    last_error: Exception | None = None

    # retry_count=3 时最多 4 次调用；每次 5 秒加 1+2+4 秒退避，预算约 27 秒，取消开销另计。
    for attempt in range(retries + 1):
        try:
            # 执行器传入的是 async attempt；若直接传阻塞同步函数，这个通用函数自身不会自动开线程。
            return await fn(*args, **kwargs) if asyncio.iscoroutinefunction(fn) else fn(*args, **kwargs)
        except Exception as e:
            # 仅捕获 Exception，asyncio.CancelledError 会继续传播，允许上层取消任务。
            last_error = e
            # 剩余次数和错误类别同时满足才重试；ToolResult(success=False) 没有抛错，不会进入这里。
            if attempt < retries and is_retryable(e):
                # 从 attempt=0 开始，得到 1、2、4 秒；当前没有随机抖动 jitter。
                delay = base_delay * (2 ** attempt)
                # 等待时让出事件循环，其他请求可继续；不使用阻塞的 time.sleep。
                await asyncio.sleep(delay)
            else:
                break

    # 负数重试配置会使循环不执行并导致 raise None；配置层目前没有对该边界做校验。
    raise last_error  # type: ignore[misc]


@dataclass
class CircuitBreaker:
    """Tracks consecutive failures; opens circuit when threshold exceeded."""

    # 按一次 invoke_tool 的最终技术失败计数，内部四次尝试失败通常只累计一次。
    failure_count: int = 0
    # 默认连续五次技术调用失败开路；业务失败会通过 record_success 清零。
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    last_failure_time: float = 0.0
    state: str = "closed"  # closed | open | half_open

    def record_success(self) -> None:
        # 清除已有失败计数；在半开探测时也会清零，因此首次探测失败不会立刻重开。
        self.failure_count = 0
        # 只在半开成功时关闭；这是简化熔断器，并未实现独占探针或完整并发状态保护。
        if self.state == "half_open":
            self.state = "closed"

    def record_failure(self) -> None:
        # 每次最终异常累加；不是每个业务拒绝都累加。
        self.failure_count += 1
        # 记录最近失败的单调时钟，用来计算是否已过恢复窗口。
        self.last_failure_time = time.monotonic()
        # 达到阈值才打开；半开失败也遵循这个阈值，和常见“失败即开路”设计不同。
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def is_open(self) -> bool:
        if self.state == "open":
            # 只在下次调用检查时切换半开，没有后台定时器主动改变 state。
            if time.monotonic() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                # 清除已有失败计数；在半开探测时也会清零，因此首次探测失败不会立刻重开。
                self.failure_count = 0
                # 表示这次允许调用，不代表下游健康；半开也返回 False，让请求实际尝试。
                return False
            return True
        # 表示这次允许调用，不代表下游健康；半开也返回 False，让请求实际尝试。
        return False


# Per-tool circuit breakers (tool_name → CircuitBreaker)
_circuits: dict[str, CircuitBreaker] = {}


def circuit_check(tool_name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a tool."""
    # 字典仅属于本进程，多 worker 彼此不共享计数，也未限制半开探测并发数。
    if tool_name not in _circuits:
        _circuits[tool_name] = CircuitBreaker()
    return _circuits[tool_name]
