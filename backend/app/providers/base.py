"""供应商无关的能力协议。

Agent / RAG 代码只依赖这一层的数据结构，不碰 openai / httpx 等 SDK 类型，
换供应商（百炼、OpenAI、DeepSeek、自建网关）只改配置，不改业务代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """模型请求调用的工具。arguments 已解析成 dict，业务侧不必再 json.loads。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = ""


class ProviderError(RuntimeError):
    """供应商调用失败。业务侧可据此走降级路径（关键词兜底、原序返回等）。"""


@runtime_checkable
class ChatProvider(Protocol):
    async def chat(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """同步接口：入库脚本是批量离线任务，检索侧现有调用链也是同步的。"""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


@runtime_checkable
class RerankProvider(Protocol):
    name: str

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """返回 (原始下标, 相关性分) 按分数降序。供应商不可用时返回空列表，由上层兜底。"""
        ...
