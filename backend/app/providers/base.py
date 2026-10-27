"""供应商无关的能力协议。

Agent / RAG 代码只依赖这一层的数据结构，不碰 openai / httpx 等 SDK 类型，
换供应商（百炼、OpenAI、DeepSeek、自建网关）只改配置，不改业务代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
# 这层只 import 标准库：一旦引到 openai / httpx 的类型，"换供应商不改业务代码"就不成立了。


@dataclass
class ToolCall:
    """模型请求调用的工具。arguments 已解析成 dict，业务侧不必再 json.loads。"""

    id: str  # 供应商回填的调用号，写回 assistant 消息时靠它对齐结果；langchain 分支可能给成空串
    name: str  # 执行器按全局工具名查 handler，没有"这个 Agent 是否允许调用"的授权维度
    arguments: dict[str, Any] = field(default_factory=dict)  # 可变默认必须走 factory；解析不出来时也是 {}，由执行器报缺必填


@dataclass
class ChatResult:
    content: str = ""  # 纯工具调用时就是空串，调用方不能用"content 为空"判定失败
    tool_calls: list[ToolCall] = field(default_factory=list)  # 非空表示模型想先调工具，轮次受 max_tool_rounds 约束
    model: str = ""  # 回显本次真正使用的模型 id，用来区分"岗位配错"和"网关回落"
    prompt_tokens: int = 0  # 供应商没回 usage 时统一记 0，不做估算，不能当账单数据用
    completion_tokens: int = 0  # 同上，只统计单次模型响应，不含整条 Agent 链路
    provider: str = ""  # 由各实现写死自己的 PROVIDER_NAME，与 registry 表键一致，trace 里靠它分辨走的哪家


class ProviderError(RuntimeError):
    """供应商调用失败。业务侧可据此走降级路径（关键词兜底、原序返回等）。"""

    # 继承 RuntimeError，所以任何 except Exception 的老代码都能兜住；代价是"缺凭据"和"远端 5xx"
    # 被压成同一类型，调用方无法据此判断该不该重试。


@runtime_checkable
class ChatProvider(Protocol):
    # 结构契约：实现类不继承也能对上。runtime_checkable 的 isinstance 只查方法名，不校验签名和返回类型。
    async def chat(
        self,
        *,  # 强制关键字传参，防止位置参数把 role 和 messages 传反；三个协议里只有 chat 这么约束
        role: str,  # 岗位名（intent/summary/pre_sales/...），实现内部再用 settings.model_for 换成模型 id
        messages: list[dict[str, Any]],  # 已是 OpenAI 风格的 {role, content} 列表，本层不做截断或裁剪
        temperature: float = 0.7,  # 协议级默认值；实现类若不支持该参数会被静默忽略而不是报错
        max_tokens: int = 2048,  # 默认值与 config 无耦合，调用方（意图分类等）通常显式传更小值
        json_mode: bool = False,  # True 时请求 JSON 输出，兼容网关支持度不一，失败会包成 ProviderError
        tools: list[dict[str, Any]] | None = None,  # 只把工具描述交给模型，本层不执行 handler
    ) -> ChatResult: ...  # 协议没有 name/dim 属性：与下面两个 Provider 不对称，调用方只能靠 ChatResult.provider 反推供应商


@runtime_checkable
class EmbeddingProvider(Protocol):
    """同步接口：入库脚本是批量离线任务，检索侧现有调用链也是同步的。"""

    # 代价：在事件循环里直接调用会占住该 worker 线程，目前检索链路没有用 asyncio.to_thread 包一层。
    dim: int  # 属性不是方法，装配期就能和 Milvus 集合 schema 比对；改它不会把旧向量转换成新空间

    def embed(self, texts: list[str]) -> list[list[float]]: ...  # 返回顺序必须与入参一一对应，调用方按下标把向量贴回 chunk

    def embed_query(self, query: str) -> list[float]: ...  # 必须与 embed 同模型同维度，否则查询向量和库内向量不在同一空间


@runtime_checkable
class RerankProvider(Protocol):
    name: str  # 只是实现类自述；registry 路由读的是配置字符串，全项目没有任何地方读这个属性

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """返回 (原始下标, 相关性分) 按分数降序。供应商不可用时返回空列表，由上层兜底。"""
        ...
        # 实际两种失败路径并不统一：dashscope 实现是抛 ProviderError（由 Reranker 捕获降级），只有 noop 返回空列表。
        # 下标相对本次 documents，不是 chunk_id；越界下标由 Reranker 过滤。
