"""开发/局域网启动入口：先把事件循环策略设对，再交给 uvicorn。

为什么需要这个文件：uvicorn 自己用 asyncio.run() 建循环，Windows 上默认就是
ProactorEventLoop，而 psycopg3 异步不支持它。于是 LangGraph 的 Postgres 检查点
（interrupt/resume 靠它落盘）会静默退回内存版——审批状态重启即丢，HITL 看着能跑、
换个进程就接不上。main.py 里那句 set_event_loop_policy 救不了，因为那时循环已经建好；
必须在 uvicorn.run() 之前设策略。

跑法（backend/ 目录下）：
    uv run python run.py              # 等价于 python run.py
    APP_PORT=8080 uv run python run.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform == "win32":
    # psycopg3 异步要求 Selector 循环（NamedPipe/AFUNIX 那套 Proactor IO 它用不了）
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn  # noqa: E402

from app.config import settings  # noqa: E402


def main() -> int:
    reload = os.getenv("APP_RELOAD", "0") in ("1", "true", "yes", "on")
    if reload:
        # reloader 会另起子进程，而 uvicorn 只有在子进程模式下才给 Selector 循环，
        # 所以 --reload 恰好绕开了 Proactor 问题，直接交给它。
        uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port,
                    reload=True, loop="asyncio")
        return 0

    config = uvicorn.Config(
        "app.main:app", host=settings.app_host, port=settings.app_port, loop="asyncio"
    )
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        # 见 uvicorn/loops/asyncio.py：非子进程时它硬返回 ProactorEventLoop，
        # 光 set_event_loop_policy 改不动，必须自己指定 loop_factory。
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
