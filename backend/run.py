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

from __future__ import annotations          # 注解延迟求值；本文件只用到 -> int，留这句与项目其它模块同一写法

import asyncio                             # 设策略与下面手动 asyncio.run 都用它
import os                                  # 只用于读 APP_RELOAD / APP_HOST 之外的开关（端口仍走 settings）
import sys                                 # sys.platform 判 Windows；sys.path 补模块搜索路径
from pathlib import Path                    # 用 __file__ 反推 backend 目录，避免依赖启动时的 cwd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 插到 0 而不是 append：本项目 app 包优先于 site-packages 里的同名包；"app.main:app" 字符串能否解析也靠这句

if sys.platform == "win32":
    # psycopg3 异步要求 Selector 循环（NamedPipe/AFUNIX 那套 Proactor IO 它用不了）
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())   # 顺序关键：必须早于 import uvicorn，之后任何 asyncio.run() 造的循环都由这句决定

import uvicorn  # noqa: E402               # 放策略之后是刻意的；E402 只是行级豁免"import 不在文件顶部"
from app.config import settings  # noqa: E402   # 这次导入顺带把 backend/.env 灌进 os.environ——main() 读 APP_RELOAD 能不能看到 .env 的值，取决于它在不在后面


def main() -> int:
    # 配置类没有 app_reload 字段，所以只能直读环境变量；比较是大小写敏感的，APP_RELOAD=TRUE 会落到"不重载"分支
    reload = os.getenv("APP_RELOAD", "0") in ("1", "true", "yes", "on")
    if reload:
        # reloader 会另起子进程，而 uvicorn 只有在子进程模式下才给 Selector 循环，
        # 所以 --reload 恰好绕开了 Proactor 问题，直接交给它。
        uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port,
                    reload=True, loop="asyncio")   # 必须传 import 字符串：对象无法跨 reloader 子进程传递；host/port 来自配置，不写死端口
        return 0                              # run() 正常返回即退出进程；反复重启期间的失败不会体现在这个 0 上

    config = uvicorn.Config(
        "app.main:app", host=settings.app_host, port=settings.app_port, loop="asyncio"   # loop="asyncio" 只负责选中 loops/asyncio.py；它不带 reload，所以 use_subprocess=False
    )
    server = uvicorn.Server(config)           # 先建 Server 而不是直接 uvicorn.run()：下面要自己接管 asyncio.run 的循环来源
    if sys.platform == "win32":
        # 见 uvicorn/loops/asyncio.py：非子进程时它硬返回 ProactorEventLoop，
        # 光 set_event_loop_policy 改不动，必须自己指定 loop_factory。
        # 补充：uvicorn 0.34 的 Server.run() 走的是不带 loop_factory 的 asyncio.run()，会沿用上面设好的策略；显式指定是不依赖第三方内部实现的双保险
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)   # 跳过 Server.run() 就不经它的 setup_event_loop；Ctrl+C 触发的取消会从这句向上抛，不走下面的 return 0
    else:
        server.run()                          # 非 Windows 不存在 Proactor 问题，走 uvicorn 完整默认路径
    return 0                                  # 只有正常停止（should_exit 置位后 serve 返回）才会走到这里


if __name__ == "__main__":
    raise SystemExit(main())                  # 把返回值变成进程退出码，便于脚本/CI 判断；比 sys.exit(main()) 少一个导入依赖
