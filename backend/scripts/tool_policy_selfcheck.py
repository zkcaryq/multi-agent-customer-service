"""工具执行管线自检：入参校验 / 超时 / 退避重试 / 熔断。

跑法：backend 目录下 `python scripts/tool_policy_selfcheck.py`（不需要任何外部服务）
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.base import ToolDefinition, ToolParameterDef, ToolResult  # noqa: E402
from app.tools.executor import invoke_tool  # noqa: E402
from app.tools.registry import tool_registry  # noqa: E402

PASS = 0
TOTAL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, TOTAL
    TOTAL += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label} {detail}")
    PASS += 1 if ok else 0


def reg(name: str, handler, *, timeout_ms: int, retry: int, required=True):
    tool_registry.register(
        ToolDefinition(
            name=name,
            display_name=name,
            description="test",
            parameters=[ToolParameterDef("order_id", "string", "订单号", is_required=required)],
            timeout_ms=timeout_ms,
            retry_count=retry,
        ),
        handler,
        agent_id=None,
    )


async def main() -> int:
    # 1) 必填缺失
    reg("t_required", lambda order_id="": ToolResult(success=True, data=order_id), timeout_ms=500, retry=0)
    r = await invoke_tool("t_required", {})
    check("缺必填参数被拦下", not r["success"] and "缺少必填参数" in r["error"], f"→ {r['error']}")

    # 2) 未知参数（模型幻觉）
    r = await invoke_tool("t_required", {"order_id": "A1", "tracking_no": "SF999"})
    check("未声明参数被拦下", not r["success"] and "未声明的参数" in r["error"], f"→ {r['error']}")

    # 3) 类型归一化
    seen = {}

    def _echo(order_id=""):
        seen["value"] = order_id
        return ToolResult(success=True, data=order_id)

    reg("t_type", _echo, timeout_ms=500, retry=0)
    r = await invoke_tool("t_type", {"order_id": 12345})
    check("非字符串参数归一化为 string", r["success"] and seen.get("value") == "12345", f"→ {seen.get('value')!r}")

    # 4) 超时真正生效
    async def _slow(order_id=""):
        await asyncio.sleep(1.0)
        return ToolResult(success=True, data="never")

    reg("t_slow", _slow, timeout_ms=150, retry=0)
    t0 = time.monotonic()
    r = await invoke_tool("t_slow", {"order_id": "A1"})
    elapsed = time.monotonic() - t0
    check("超时被截断", not r["success"] and "超时" in r["error"], f"→ {r['error']} 耗时{elapsed:.2f}s")
    check("耗时受 timeout 约束（含重试等待）", elapsed < 1.0, f"{elapsed:.2f}s < 1.0s")

    # 5) 可重试错误：第一次连接失败，第二次成功
    attempts = {"n": 0}

    def _flaky(order_id=""):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("connection refused")
        return ToolResult(success=True, data=order_id)

    reg("t_flaky", _flaky, timeout_ms=800, retry=1)
    r = await invoke_tool("t_flaky", {"order_id": "A1"})
    check("可重试错误自动重试后成功", r["success"] and attempts["n"] == 2, f"→ 尝试{attempts['n']}次")

    # 6) 业务错误不重试
    biz = {"n": 0}

    def _biz(order_id=""):
        biz["n"] += 1
        return ToolResult(success=False, error="订单不属于该用户")

    reg("t_biz", _biz, timeout_ms=500, retry=2)
    r = await invoke_tool("t_biz", {"order_id": "A1"})
    check("业务失败按结果返回，不抛异常", not r["success"] and "不属于" in r["error"], f"→ {r['error']}")

    # 7) 熔断：连续失败达阈值后不再打下游
    boom = {"n": 0}

    def _always_fail(order_id=""):
        boom["n"] += 1
        raise ConnectionError("connection reset")

    reg("t_breaker", _always_fail, timeout_ms=300, retry=0)
    for _ in range(5):
        await invoke_tool("t_breaker", {"order_id": "A1"})
    r = await invoke_tool("t_breaker", {"order_id": "A1"})
    check("连续失败后熔断打开", r.get("degraded") and "熔断" in r["error"], f"→ {r['error']}")
    check("熔断期间不再打下游", boom["n"] == 5, f"下游被调用 {boom['n']} 次")

    # 8) 默认超时来自配置
    from app.tools.base import ToolDefinition as D
    from app.config import settings

    check("TOOL_TIMEOUT_MS 配置真正生效", D(name="x", display_name="x", description="x").timeout_ms == settings.tool_timeout_ms,
          f"→ {settings.tool_timeout_ms}ms")

    print(f"\n{PASS}/{TOTAL} 通过")
    return 0 if PASS == TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
