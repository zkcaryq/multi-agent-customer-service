"""按配置装配 provider。业务代码只调 get_*，不认识任何具体供应商类。

【为什么需要这一层】
项目要能在"百炼 / OpenAI 兼容网关 / 本地占位实现"之间切换，而不改业务代码。
做法是三步：
1. `base.py` 定义协议（ChatProvider / EmbeddingProvider / RerankProvider）；
2. 各实现类只依赖协议和方法签名；
3. 本文件根据配置字符串（如 `openai_compat`）选实现、缓存实例。

业务侧永远写 `get_chat()`，不写 `OpenAICompatChat()` —— 换供应商只改 `.env`。

【为什么要缓存实例】
每次请求新建 OpenAI client 会重复建连接池。这里按 `kind:provider` 缓存，
进程内一个实现只实例化一次；也可被自检脚本用 `override_chat` 覆盖。
"""

from __future__ import annotations

from app.config import settings
from app.providers.base import ChatProvider, EmbeddingProvider, ProviderError, RerankProvider
from app.providers.chat import OpenAICompatChat
from app.providers.embedding import HashEmbedding, OpenAICompatEmbedding
from app.providers.langchain_chat import LangChainChat
from app.providers.rerank import DashScopeRerank, NoopRerank

# provider 名 → 实现类。新增供应商只在这里登记一行。
# langchain_openai 是当前默认实现（用 LangChain 适配 Tool Calling），
# openai_compat 是裸 SDK 实现，留作对照与降级。
_CHAT = {"langchain_openai": LangChainChat, "openai_compat": OpenAICompatChat}
_EMBEDDING = {"openai_compat": OpenAICompatEmbedding, "hash": HashEmbedding}
_RERANK = {"dashscope": DashScopeRerank, "noop": NoopRerank}

_cache: dict[str, object] = {}

# 自检/测试用的显式覆盖。以前脚本直接往 _cache 里塞 "chat:openai_compat"，
# 一旦把 CHAT_PROVIDER 换成别的实现，键就对不上、桩被静默绕过，自检会去打真实模型
# （表现为：HITL 用例不再稳定触发审批、摘要自检拿到模型自由发挥的文本）。
_override: ChatProvider | None = None


def override_chat(provider: ChatProvider | None) -> None:
    """不走配置，直接指定 chat 实现；传 None 恢复按配置装配。"""
    global _override
    _override = provider


def _build(kind: str, table: dict, chosen: str):
    """按名字取实现，实例缓存复用。

    未知名字抛 ProviderError 而不是 KeyError：调用方需要的是"可读的配置错误"，
    例如 `未知的 chat provider: 'qwen'，可选: openai_compat`。
    """
    if chosen not in table:
        raise ProviderError(
            f"未知的 {kind} provider: {chosen!r}，可选: {', '.join(sorted(table))}"
        )
    key = f"{kind}:{chosen}"
    if key not in _cache:
        _cache[key] = table[chosen]()
    return _cache[key]


def get_chat() -> ChatProvider:
    # 覆盖优先：自检脚本注入的桩必须能压过配置，否则测试结果取决于 .env
    if _override is not None:
        return _override
    return _build("chat", _CHAT, settings.chat_provider)  # type: ignore[return-value]


def get_embedding() -> EmbeddingProvider:
    return _build("embedding", _EMBEDDING, settings.embedding_provider)  # type: ignore[return-value]


def get_rerank() -> RerankProvider:
    return _build("rerank", _RERANK, settings.rerank_provider)  # type: ignore[return-value]


def describe() -> str:
    """启动时打印一次，让"当前用的是哪家模型"在日志里可见而不是藏在代码里。

    排障第一步就是看这行：模型、向量模型、重排、向量库形态全在这。
    """
    from app.rag.vector_store import vector_store

    # 每个岗位用的模型都列出来（intent/summary/pre_sales/...）
    roles = ", ".join(f"{r}={settings.llm_models[r]}" for r in settings.llm_models)
    return (
        f"chat={settings.chat_provider}@{settings.chat_base_url} ({roles}) | "
        f"embedding={settings.embedding_provider}:{settings.embedding_model}/{settings.embedding_dim} | "
        f"rerank={settings.rerank_provider}:{settings.rerank_model} | "
        f"vectorstore={'Milvus Server/Zilliz' if vector_store.is_remote else 'Milvus Lite'}@{vector_store.uri}"
    )


# provider 名 → 缺了就无法调用的环境变量名。hash/noop 是本地占位实现，不需要凭据。
_KEYED = {
    "chat": ("openai_compat", "CHAT_API_KEY"),
    "embedding": ("openai_compat", "EMBEDDING_API_KEY"),
    "rerank": ("dashscope", "RERANK_API_KEY"),
}


def preflight() -> list[str]:
    """把"要到第一次调用才炸"的凭据缺失提前到启动时发现。

    只告警不阻塞：评测和自检要能在无 Key 的环境里跑（它们会用桩覆盖）。
    """
    issues = []
    for kind, (needs_key, var) in _KEYED.items():
        chosen = getattr(settings, f"{kind}_provider")
        if chosen == needs_key and not settings.key_for(kind):
            issues.append(f"{kind}_provider={chosen} 需要 {var}（当前为空）")
    return issues
