"""按配置装配 provider。业务代码只调 get_*，不认识任何具体供应商类。"""

from __future__ import annotations

from app.config import settings
from app.providers.base import ChatProvider, EmbeddingProvider, ProviderError, RerankProvider
from app.providers.chat import OpenAICompatChat
from app.providers.embedding import HashEmbedding, OpenAICompatEmbedding
from app.providers.langchain_chat import LangChainChat
from app.providers.rerank import DashScopeRerank, NoopRerank

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
    if chosen not in table:
        raise ProviderError(
            f"未知的 {kind} provider: {chosen!r}，可选: {', '.join(sorted(table))}"
        )
    key = f"{kind}:{chosen}"
    if key not in _cache:
        _cache[key] = table[chosen]()
    return _cache[key]


def get_chat() -> ChatProvider:
    if _override is not None:
        return _override
    return _build("chat", _CHAT, settings.chat_provider)  # type: ignore[return-value]


def get_embedding() -> EmbeddingProvider:
    return _build("embedding", _EMBEDDING, settings.embedding_provider)  # type: ignore[return-value]


def get_rerank() -> RerankProvider:
    return _build("rerank", _RERANK, settings.rerank_provider)  # type: ignore[return-value]


def describe() -> str:
    """启动时打印一次，让"当前用的是哪家模型"在日志里可见而不是藏在代码里。"""
    from app.rag.vector_store import vector_store

    roles = ", ".join(f"{r}={settings.llm_models[r]}" for r in settings.llm_models)
    return (
        f"chat={settings.chat_provider}@{settings.chat_base_url} ({roles}) | "
        f"embedding={settings.embedding_provider}:{settings.embedding_model}/{settings.embedding_dim} | "
        f"rerank={settings.rerank_provider}:{settings.rerank_model} | "
        f"vectorstore={'Milvus Server/Zilliz' if vector_store.is_remote else 'Milvus Lite'}@{vector_store.uri}"
    )


# provider 名 → 缺了就无法调用的环境变量名。hash/noop 是本地占位实现，不需要凭据。
_KEYED = {
    "chat": ({"langchain_openai", "openai_compat"}, "CHAT_API_KEY"),
    "embedding": ({"openai_compat"}, "EMBEDDING_API_KEY"),
    "rerank": ({"dashscope"}, "RERANK_API_KEY"),
}


def preflight() -> list[str]:
    """把"要到第一次调用才炸"的凭据缺失提前到启动时发现。"""
    issues = []
    for kind, (needs_key, var) in _KEYED.items():
        chosen = getattr(settings, f"{kind}_provider")
        if chosen in needs_key and not settings.key_for(kind):
            issues.append(f"{kind}_provider={chosen} 需要 {var}（当前为空）")
    return issues
