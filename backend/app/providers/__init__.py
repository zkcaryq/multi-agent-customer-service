"""供应商协议的对外出口：业务只通过这里拿 get_* 与统一数据结构。"""

from app.providers.base import (
    ChatResult,
    ToolCall,
    ProviderError,
)
from app.providers.registry import describe, get_chat, get_embedding, get_rerank, preflight

__all__ = [
    "ChatResult",
    "ToolCall",
    "ProviderError",
    "describe",
    "get_chat",
    "get_embedding",
    "get_rerank",
    "preflight",
]
