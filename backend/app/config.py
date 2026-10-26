"""配置中心：所有环境变量收敛在这里，业务代码只读 settings。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Auto-load .env file from backend/ directory
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # python-dotenv not installed — parse .env manually
        with open(_env_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _val = _line.split("=", 1)
                    os.environ.setdefault(_key.strip(), _val.strip())


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    return float(raw) if raw else default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# 调用大模型的岗位（role）。role 决定用哪个模型，模型 id 只在配置里出现。
LLM_ROLES = (
    "intent",
    "summary",
    "pre_sales",
    "during_sales",
    "after_sales",
    "complaint",
)


@dataclass
class Settings:
    # ── 基础设施 ────────────────────────────────────────────────────────────
    database_url: str = _env(
        "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/customer_service"
    )
    redis_url: str = _env("REDIS_URL", "redis://localhost:6379/0")

    # 向量库：MILVUS_ENDPOINT 给了就是 Milvus Server / Zilliz Cloud，
    # 否则用 MILVUS_LITE_PATH 指定的本地文件（Milvus Lite，嵌入式）。
    # 注意：不要用 MILVUS_URI —— 它是 pymilvus 的保留环境变量，import 时就会被
    # 当作服务端地址解析，填本地路径会抛 ConnectionConfigException。
    vectorstore_provider: str = _env("VECTORSTORE_PROVIDER", "milvus")
    milvus_endpoint: str = _env("MILVUS_ENDPOINT")
    milvus_lite_path: str = _env("MILVUS_LITE_PATH")
    milvus_host: str = _env("MILVUS_HOST", "localhost")
    milvus_port: int = _env_int("MILVUS_PORT", 19530)
    milvus_token: str = _env("MILVUS_TOKEN")  # Zilliz Cloud: "<user>:<password>" 或 API key
    milvus_db_name: str = _env("MILVUS_DB_NAME")
    milvus_collection: str = _env("MILVUS_COLLECTION", "knowledge_chunks")

    # ── Chat 模型（provider 无关）──────────────────────────────────────────
    chat_provider: str = _env("CHAT_PROVIDER", "langchain_openai")
    chat_api_key: str = _env("CHAT_API_KEY") or _env("DASHSCOPE_API_KEY")
    chat_base_url: str = _env(
        "CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    chat_timeout_s: float = _env_float("CHAT_TIMEOUT_S", 30.0)
    chat_max_retries: int = _env_int("CHAT_MAX_RETRIES", 2)
    # 客服、意图分类与工具规划优先低延迟和结构化稳定性，默认关闭模型思考。
    chat_enable_thinking: bool = _env_bool("CHAT_ENABLE_THINKING", False)
    # role -> model id，优先级：LLM_MODEL_<ROLE> > LLM_MODEL_DEFAULT
    llm_model_default: str = _env("LLM_MODEL_DEFAULT", "qwen-plus")
    llm_models: dict[str, str] = field(default_factory=dict)

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_provider: str = _env("EMBEDDING_PROVIDER", "openai_compat")
    embedding_api_key: str = _env("EMBEDDING_API_KEY")
    embedding_base_url: str = _env("EMBEDDING_BASE_URL")
    embedding_model: str = _env("EMBEDDING_MODEL", "text-embedding-v3")
    embedding_dim: int = _env_int("EMBEDDING_DIM", 1024)

    # ── Rerank ─────────────────────────────────────────────────────────────
    rerank_provider: str = _env("RERANK_PROVIDER", "dashscope")
    rerank_api_key: str = _env("RERANK_API_KEY")
    rerank_model: str = _env("RERANK_MODEL", "qwen3-rerank")
    # DashScope 原生端点（rerank 不在 OpenAI 兼容路径上）
    rerank_endpoint: str = _env(
        "RERANK_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
    rerank_top_n: int = _env_int("RERANK_TOP_N", 5)
    rerank_timeout_s: float = _env_float("RERANK_TIMEOUT_S", 10.0)

    # ── 后台鉴权 ───────────────────────────────────────────────────────────
    # JWT 签名密钥。不设就用开发默认值并在启动时告警——生产必须显式配置。
    jwt_secret: str = _env("JWT_SECRET")
    jwt_ttl_s: int = _env_int("JWT_TTL_S", 28800)          # 8 小时
    jwt_algorithm: str = "HS256"
    # 首次启动时引导出的后台账号（局域网演示用；生产应改密码或直接建自己的账号）
    bootstrap_admin_user: str = _env("ADMIN_BOOTSTRAP_USER", "admin")
    bootstrap_admin_password: str = _env("ADMIN_BOOTSTRAP_PASSWORD")

    # ── 服务监听 ───────────────────────────────────────────────────────────
    app_host: str = _env("APP_HOST", "127.0.0.1")
    app_port: int = _env_int("APP_PORT", 8000)

    # ── 日志 ───────────────────────────────────────────────────────────────
    # 级别与格式都从环境走，本地调试改成 DEBUG 不用动代码。
    log_level: str = _env("LOG_LEVEL", "INFO")
    # 使用项目专用变量名，避免 CI / IDE 常见的 LOG_FORMAT=json 污染
    # logging.Formatter 并导致应用在导入阶段崩溃。
    log_format: str = _env(
        "APP_LOG_FORMAT",
        "%(asctime)s %(levelname)-7s %(name)s [%(trace)s] %(message)s",
    )

    # ── Agent defaults ─────────────────────────────────────────────────────
    max_conversation_turns: int = _env_int("MAX_CONVERSATION_TURNS", 10)
    summary_trigger_turns: int = _env_int("SUMMARY_TRIGGER_TURNS", 8)
    tool_retry_count: int = _env_int("TOOL_RETRY_COUNT", 3)
    tool_timeout_ms: int = _env_int("TOOL_TIMEOUT_MS", 5000)
    max_handoff_failures: int = _env_int("MAX_HANDOFF_FAILURES", 3)
    max_tool_rounds: int = _env_int("MAX_TOOL_ROUNDS", 3)
    max_collaboration_agents: int = _env_int("MAX_COLLABORATION_AGENTS", 2)
    # 高危操作审批的等待上限，超时自动拒绝（原来只有"人工确认"，没有"超时拒绝"）
    approval_timeout_s: int = _env_int("APPROVAL_TIMEOUT_S", 900)

    # ── Retrieval defaults ─────────────────────────────────────────────────
    retrieval_top_k: int = _env_int("RETRIEVAL_TOP_K", 5)
    # 向量候选的相似度下限。这个值跟 embedding 模型强相关，换模型必须重新标定：
    # 在 40 条评测集上用当前 qwen 向量实测 —— 0.75 只召回 52% 的相关切片、平均每问
    # 留下 0.9 条（等于把向量召回关掉）；0.50 召回 88%、平均保留 7.5 条；0.45 起
    # 进入平台期（90%）。取 0.50，剩下的交给重排序。
    similarity_threshold: float = _env_float("SIMILARITY_THRESHOLD", 0.50)
    bm25_weight: float = _env_float("BM25_WEIGHT", 0.3)
    vector_weight: float = _env_float("VECTOR_WEIGHT", 0.7)

    def __post_init__(self) -> None:
        for role in LLM_ROLES:
            self.llm_models[role] = _env(f"LLM_MODEL_{role.upper()}") or self.llm_model_default

    def model_for(self, role: str) -> str:
        """岗位对应的模型 id。未知岗位直接报错，避免静默用错模型。"""
        if role not in self.llm_models:
            raise KeyError(f"未配置的模型岗位: {role}，可选: {', '.join(LLM_ROLES)}")
        return self.llm_models[role]

    def key_for(self, kind: str) -> str:
        """各能力可以各用一份凭据；没单独配就回落到 chat 的凭据。"""
        if kind == "chat":
            return self.chat_api_key
        dedicated = {"embedding": self.embedding_api_key, "rerank": self.rerank_api_key}[kind]
        return dedicated or self.chat_api_key

    def base_url_for(self, kind: str) -> str:
        if kind == "chat":
            return self.chat_base_url
        return {"embedding": self.embedding_base_url or self.chat_base_url, "rerank": self.rerank_endpoint}[kind]


settings = Settings()
