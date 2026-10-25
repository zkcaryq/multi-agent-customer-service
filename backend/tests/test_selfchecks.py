"""把 scripts/*_selfcheck.py 纳入 pytest 发现。

用 subprocess 而不是 import 脚本模块：脚本靠"导入 app 之前先设 os.environ"来关掉
真实出网，而 app.config 的 settings 是模块级单例——同一进程里跑第二个脚本时
app.* 早已绑到旧的 settings 上（reload 也修不好，因为各模块持有的是旧对象引用）。
子进程隔离才等价于手工 `python scripts/xxx.py`。

退出码约定：0 = 全绿；1 = 有断言失败（输出里能看到 FAIL）；其他 = 脚本自己炸了。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent

# 需要 PostgreSQL + 会话态的端到端脚本
INTEGRATION = (
    "offline_selfcheck.py",            # 四域全链路
    "hitl_selfcheck.py",               # 审批暂停/恢复 + 超时
    "hitl_api_selfcheck.py",           # /chat → /chat/confirm HTTP 闭环
    "approval_queue_selfcheck.py",     # 后台队列 + 鉴权 + 审计 + 工单
    "order_tools_selfcheck.py",        # 订单/退款/CRM 工具与越权拦截
    "complaint_selfcheck.py",          # 投诉域建单与定级
    "memory_selfcheck.py",             # Redis 会话记忆
    "tool_policy_selfcheck.py",        # 超时/重试/熔断/入参校验
    "metrics_selfcheck.py",            # 指标计算与落库
)

TIMEOUT_S = 1800


@pytest.mark.integration
@pytest.mark.parametrize("script", INTEGRATION)
def test_selfcheck_passes(script: str) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, str(BACKEND / "scripts" / script)],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_S,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    # 只看退出码和 FAIL 标记：pymilvus/grpc 会往 stderr 打自己的 traceback 噪声，
    # 脚本本身仍然全绿，用 "Traceback" 判失败会误报。
    assert "FAIL" not in output, f"{script} 有断言失败\n{output[-2500:]}"
    assert proc.returncode == 0, (
        f"{script} 退出码 {proc.returncode}\n{output[-2500:]}"
    )
