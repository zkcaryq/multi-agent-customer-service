"""按配置装配 provider。业务代码只调 get_*，不认识任何具体供应商类。

【为什么需要这一层】
项目要能在"百炼 / OpenAI 兼容网关 / 本地占位实现"之间切换，而不改业务代码。
做法是三步：
1. `base.py` 定义协议（ChatProvider / EmbeddingProvider / RerankProvider）；
2. 各实现类只依赖协议和方法签名；
3. 本文件根据配置字符串（如 `openai_compat`）选实现、缓存实例。

业务侧永远写 `get_chat()`，不写 `OpenAICompatChat()` —— 换供应商只改 `.env`。

【为什么要缓存实例】
这里按 kind:provider 缓存适配器实例，可由 override_chat 替换。
是否复用底层客户端取决于具体实现：OpenAICompatChat 延迟缓存客户端，
LangChainChat.chat 当前每次构造 ChatOpenAI，不能混称为都复用了连接池。
"""

from __future__ import annotations

from app.config import settings  # 三个 *_provider 配置项是本文件唯一的路由输入
from app.providers.base import ChatProvider, EmbeddingProvider, ProviderError, RerankProvider
from app.providers.chat import OpenAICompatChat
from app.providers.embedding import HashEmbedding, OpenAICompatEmbedding
from app.providers.langchain_chat import LangChainChat
from app.providers.rerank import DashScopeRerank, NoopRerank
# 这里是全项目唯一 import 具体供应商类的地方；业务侧只 import get_*，两者依赖方向不能反过来。

# provider 名 → 实现类。新增供应商只在这里登记一行。
# langchain_openai 是当前默认实现（用 LangChain 适配 Tool Calling），
# openai_compat 是裸 SDK 实现，留作对照与降级。
_CHAT = {"langchain_openai": LangChainChat, "openai_compat": OpenAICompatChat}
_EMBEDDING = {"openai_compat": OpenAICompatEmbedding, "hash": HashEmbedding}  # hash 是占位实现，无网络无凭据，可裸跑自检
_RERANK = {"dashscope": DashScopeRerank, "noop": NoopRerank}  # 表里没有 OpenAI 兼容 rerank：重排只对接了百炼原生端点
# 注意三张表只覆盖 chat/embedding/rerank：向量库不在 registry 里，
# vector_store._resolve_target 写死 Milvus，settings.vectorstore_provider 全项目没有读取方（死配置）。

_cache: dict[str, object] = {}  # 值类型是 object 而不是协议类型：这里不想静态担保实现合规，取用时靠 type: ignore 转协议

# 自检/测试用的显式覆盖。以前脚本直接往 _cache 里塞 "chat:openai_compat"，
# 一旦把 CHAT_PROVIDER 换成别的实现，键就对不上、桩被静默绕过，自检会去打真实模型
# （表现为：HITL 用例不再稳定触发审批、摘要自检拿到模型自由发挥的文本）。
_override: ChatProvider | None = None  # 进程级全局；多协程并发跑不同配置时没有隔离，测试必须自己记得复位


def override_chat(provider: ChatProvider | None) -> None:
    """不走配置，直接指定 chat 实现；传 None 恢复按配置装配。"""
    # 测试进程里临时替换 chat provider；传 None 恢复配置，不改变磁盘 .env。
    global _override  # 只覆盖 chat：embedding/rerank 没有对应的覆盖入口，自检要桩掉向量/重排只能改配置
    _override = provider


def _build(kind: str, table: dict, chosen: str):
    """按名字取实现，实例缓存复用。

    未知名字抛 ProviderError 而不是 KeyError：调用方需要的是"可读的配置错误"，
    例如 `未知的 chat provider: 'qwen'，可选: openai_compat`。
    """
    if chosen not in table:
        raise ProviderError(
            f"未知的 {kind} provider: {chosen!r}，可选: {', '.join(sorted(table))}"
        )  # sorted 让可选列表稳定，自检和文档能断言同一句文案；拼错的配置值在装配期就炸，不留到首次调用
    # kind 防止 chat 和 embedding 同名实现撞缓存；配置变化通常需要重启/清缓存才能生效。
    key = f"{kind}:{chosen}"
    if key not in _cache:
        _cache[key] = table[chosen]()  # 无参构造：构造时不校验凭据、不建连，所以"装配成功"不等于"能用"；构造若抛异常不会留下半初始化实例
    return _cache[key]


def get_chat() -> ChatProvider:
    # 覆盖优先：自检脚本注入的桩必须能压过配置，否则测试结果取决于 .env
    if _override is not None:
        return _override  # 覆盖实例不进 _cache，也不做合规检查（runtime_checkable 的 isinstance 只查方法名）
    # CHAT_PROVIDER 默认 langchain_openai；配置改成 openai_compat 才走裸 SDK 那份实现
    return _build("chat", _CHAT, settings.chat_provider)  # type: ignore[return-value]


def get_embedding() -> EmbeddingProvider:
    # 返回协议对象，HashEmbedding 仅作联调占位，不能用来证明语义召回质量。
    # EMBEDDING_PROVIDER 默认 openai_compat；它是同步能力，retriever/ingestion 直接在事件循环里调用会占住 worker
    return _build("embedding", _EMBEDDING, settings.embedding_provider)  # type: ignore[return-value]


def get_rerank() -> RerankProvider:
    # noop 返回空排名，由业务 Reranker 执行分数降级。
    # RERANK_PROVIDER 默认 dashscope；真实实现每次调用都可能抛 ProviderError，降级判定在 reranker 层而不在这里
    return _build("rerank", _RERANK, settings.rerank_provider)  # type: ignore[return-value]


def describe() -> str:
    """启动时打印一次，让"当前用的是哪家模型"在日志里可见而不是藏在代码里。

    排障第一步就是看这行：模型、向量模型、重排、向量库形态全在这。
    """
    from app.rag.vector_store import vector_store  # 函数内导入避免 providers→rag 反向依赖；导入即实例化 VectorStore，但只解析 uri、不真正连接

    # 每个岗位用的模型都列出来（intent/summary/pre_sales/...）
    roles = ", ".join(f"{r}={settings.llm_models[r]}" for r in settings.llm_models)
    return (
        f"chat={settings.chat_provider}@{settings.chat_base_url} ({roles}) | "  # base_url 会进日志：不含凭据，但仍是内部端点信息，别外发
        f"embedding={settings.embedding_provider}:{settings.embedding_model}/{settings.embedding_dim} | "  # 打维度是为了和集合 schema 肉眼对上（改 dim 必须重建集合）
        f"rerank={settings.rerank_provider}:{settings.rerank_model} | "
        f"vectorstore={'Milvus Server/Zilliz' if vector_store.is_remote else 'Milvus Lite'}@{vector_store.uri}"  # 形态由端点是否为 http(s) 推断，不是读 vectorstore_provider
    )


# provider 名 → 缺了就无法调用的环境变量名。hash/noop 是本地占位实现，不需要凭据。
_KEYED = {
    "chat": ("openai_compat", "CHAT_API_KEY"),  # 只列 openai_compat：当前默认 langchain_openai 不在这里，缺 key 不会在启动时告警（要到第一次真实调用才炸）
    "embedding": ("openai_compat", "EMBEDDING_API_KEY"),
    "rerank": ("dashscope", "RERANK_API_KEY"),
}  # 每项只有一个受检实现和一个提示用变量名；var 只是给人看的，实际检查走 settings.key_for（会算上回落凭据）


def preflight() -> list[str]:
    """把"要到第一次调用才炸"的凭据缺失提前到启动时发现。

    只告警不阻塞：评测和自检要能在无 Key 的环境里跑（它们会用桩覆盖）。
    """
    issues = []  # 返回文案列表而不抛异常：由 main.py 的 lifespan 逐条 log.warning，启动继续
    for kind, (needs_key, var) in _KEYED.items():
        chosen = getattr(settings, f"{kind}_provider")  # 靠命名约定取配置（chat_provider/embedding_provider/rerank_provider），配置项改名会 AttributeError
        # 当前 _KEYED 的 chat 仅列 openai_compat，langchain_openai 缺 key 可能到真实调用时才发现。
        if chosen == needs_key and not settings.key_for(kind):
            issues.append(f"{kind}_provider={chosen} 需要 {var}（当前为空）")  # key_for 的回落在这里生效：chat 空但 DASHSCOPE_API_KEY 有值时不算问题
    return issues  # 空列表不代表凭据可用，只代表这三项检查没命中
