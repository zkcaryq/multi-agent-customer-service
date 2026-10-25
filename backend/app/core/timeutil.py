"""时间工具。

本库所有 datetime 列都是 TIMESTAMP WITHOUT TIME ZONE（naive），约定统一存 UTC。
因此应用侧一律用 utcnow_naive()，不要直接 datetime.now(timezone.utc)——
把 tz-aware 对象交给 asyncpg 会报 "can't subtract offset-naive and offset-aware"。

超时/有效期这类判定优先交给数据库的 now()：应用机与数据库主机可能存在时钟漂移。
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
