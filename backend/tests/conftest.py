"""测试夹具。

单元测试不碰外部服务；集成自检脚本要连真实 PostgreSQL，所以在这里统一探测，
不可达就整批跳过——否则换台机器 `pytest` 会红一片，看不出是真坏了还是没起服务。
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

# backend/ 目录进 sys.path，测试里 import app.* 不用再各写一遍 sys.path.insert
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _reachable(url: str, default_port: int) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False
    try:
        with socket.create_connection((host, parsed.port or default_port), timeout=1.5):
            return True
    except OSError:
        return False


def pg_reachable() -> bool:
    from app.config import settings

    return _reachable(settings.database_url, 5432)


def pytest_collection_modifyitems(config, items):
    if pg_reachable():
        return
    skip = pytest.mark.skip(reason="PostgreSQL 不可达，集成自检需要真实库")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
