"""配置中心：所有环境变量收敛在这里，业务代码只读 settings。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Auto-load .env file from backend/ directory
_env_path = Path(__file__).parent.parent / ".env"    # 锚定包文件位置，与启动时的 cwd 无关（run.py 从 backend/ 起也命中同一个）
if _env_path.exists():                                # 文件不存在就静默跳过：环境变量必须能独立支撑运行（容器/CI 全靠它）
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)                        # 默认 override=False：真实环境变量优先于 .env，不会被文件反盖
    except ImportError:
        # python-dotenv not installed — parse .env manually
        with open(_env_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _val = _line.split("=", 1)   # 只切第一个 =：值里带 = （连接串、URL）不会被截断
                    os.environ.setdefault(_key.strip(), _val.strip())   # setdefault 与 load_dotenv 同语义：环境优先
                    # 这段兜底不认 export / 引号 / 多行值，也不解析空值差异；正式部署仍建议装 python-dotenv


# 全部 _env_* 都在"类体求值"时被调用，也就是 import 本模块的那一刻读环境；
# 之后改 os.environ 不会生效，测试想换值得用 monkeypatch + 重载模块。
def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()    # `or` 让"显式设成空串"与"没设"不可区分；顺带吃掉首尾空白/换行（.env 常见）


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    # 空串回落默认；填了非数字（比如 TOOL_TIMEOUT_MS=5s）直接 ValueError，且栈指向 dataclass 字段、发生在导入期
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    return float(raw) if raw else default         # 同上：SIMILARITY_THRESHOLD 写错一个字符就是整包 import 失败


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if not raw:
        return default
    # 白名单之外的任何值（含 "flase" 这种拼错）都判 False，因此这里无法区分"未配置"和"配置为假"
    return raw.lower() in {"1", "true", "yes", "on"}


# 调用大模型的岗位（role）。role 决定用哪个模型，模型 id 只在配置里出现。
# 这份元组就是 model_for() 的合法取值域：新增岗位要同时改这里、LLM_MODEL_<ROLE> 与 agents 目录名。
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
    # 字段顺序不影响取值（每个默认值都在类体里独立求值），只有 __post_init__ 依赖 llm_model_default 先算好。
    # ── 基础设施 ────────────────────────────────────────────────────────────
    database_url: str = _env(
        "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/customer_service"
    )   # 必须带 +asyncpg：engine.py 用 create_async_engine；填同步驱动要等首次建连接才报错
    redis_url: str = _env("REDIS_URL", "redis://localhost:6379/0")   # 会话记忆；Redis 读失败时 memory.py 退化成"本次无历史"的进程内对象

    # 向量库：MILVUS_ENDPOINT 给了就是 Milvus Server / Zilliz Cloud，
    # 否则用 MILVUS_LITE_PATH 指定的本地文件（Milvus Lite，嵌入式）。
    # 注意：不要用 MILVUS_URI —— 它是 pymilvus 的保留环境变量，import 时就会被
    # 当作服务端地址解析，填本地路径会抛 ConnectionConfigException。
    vectorstore_provider: str = _env("VECTORSTORE_PROVIDER", "milvus")   # 目前只有 milvus 一种实现，本字段没有被任何代码读取
    milvus_endpoint: str = _env("MILVUS_ENDPOINT")    # 空串 = 走本地文件；vector_store._resolve_target 用 .strip() 后判非空来决定形态
    milvus_lite_path: str = _env("MILVUS_LITE_PATH")  # 相对路径会按 backend/ 解析；pymilvus 在 Windows 上只收正斜杠路径
    milvus_host: str = _env("MILVUS_HOST", "localhost")   # 供 pymilvus 自身的环境变量约定留档，本项目代码不读它（连接只认 endpoint）
    milvus_port: int = _env_int("MILVUS_PORT", 19530)      # 同上：改它不会改变实际连接地址，容易误以为生效
    milvus_token: str = _env("MILVUS_TOKEN")  # Zilliz Cloud: "<user>:<password>" 或 API key
    milvus_db_name: str = _env("MILVUS_DB_NAME")           # 只在 endpoint 非空（远端）时才传给 client
    milvus_collection: str = _env("MILVUS_COLLECTION", "knowledge_chunks")   # 改名 = 换一个空集合，检索立刻 0 命中，需重新入库

    # ── Chat 模型（provider 无关）──────────────────────────────────────────
    chat_provider: str = _env("CHAT_PROVIDER", "langchain_openai")   # registry._CHAT 的键；填错在首次 get_chat() 才 KeyError，不是 import 期
    chat_api_key: str = _env("CHAT_API_KEY") or _env("DASHSCOPE_API_KEY")   # 兼容历史变量名：两个都空时 preflight() 会提前告警
    chat_base_url: str = _env(
        "CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )   # OpenAI 兼容路径；embedding 没单独配 BASE_URL 时也回落到这里
    chat_timeout_s: float = _env_float("CHAT_TIMEOUT_S", 30.0)   # 传给 provider 客户端；它管不到工具执行器的 TOOL_TIMEOUT_MS 预算
    chat_max_retries: int = _env_int("CHAT_MAX_RETRIES", 2)      # 模型层重试，与工具层 TOOL_RETRY_COUNT 各自独立，最坏耗时是两者相乘
    # 客服、意图分类与工具规划优先低延迟和结构化稳定性，默认关闭模型思考。
    chat_enable_thinking: bool = _env_bool("CHAT_ENABLE_THINKING", False)   # 作为 extra_body 透传，供应商不认这个键时行为由对端决定
    # role -> model id，优先级：LLM_MODEL_<ROLE> > LLM_MODEL_DEFAULT
    llm_model_default: str = _env("LLM_MODEL_DEFAULT", "qwen-plus")   # 兜底：六个岗位都没单独配时全用它
    llm_models: dict[str, str] = field(default_factory=dict)   # 可变默认必须走 default_factory，然后在 __post_init__ 里按 role 填

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_provider: str = _env("EMBEDDING_PROVIDER", "openai_compat")   # 填 hash 只是本地占位（自检用），不能证明语义召回质量
    embedding_api_key: str = _env("EMBEDDING_API_KEY")     # 空则 key_for("embedding") 回落到 chat 的凭据
    embedding_base_url: str = _env("EMBEDDING_BASE_URL")   # 空则 base_url_for("embedding") 回落到 chat_base_url
    embedding_model: str = _env("EMBEDDING_MODEL", "text-embedding-v3")
    embedding_dim: int = _env_int("EMBEDDING_DIM", 1024)   # 建集合时写进向量列 schema：与模型真实维度不符会在插入时报维度错，改它必须重灌库

    # ── Rerank ─────────────────────────────────────────────────────────────
    rerank_provider: str = _env("RERANK_PROVIDER", "dashscope")   # noop 返回空排名，由 rag/reranker 走分数降级
    rerank_api_key: str = _env("RERANK_API_KEY")                  # 空则回落 chat 凭据
    rerank_model: str = _env("RERANK_MODEL", "qwen3-rerank")
    # DashScope 原生端点（rerank 不在 OpenAI 兼容路径上）
    rerank_endpoint: str = _env(
        "RERANK_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
    rerank_top_n: int = _env_int("RERANK_TOP_N", 5)   # 交给重排模型的候选上限；比融合后的候选少时，后面的候选根本没机会翻盘
    rerank_timeout_s: float = _env_float("RERANK_TIMEOUT_S", 10.0)   # httpx 客户端超时；ProviderError 会降级为融合分数排序，不抛给请求

    # ── 后台鉴权 ───────────────────────────────────────────────────────────
    # JWT 签名密钥。不设就用开发默认值并在启动时告警——生产必须显式配置。
    jwt_secret: str = _env("JWT_SECRET")     # 空值时 auth._secret() 回落内置开发密钥（只告警一次）；泄漏即可伪造 admin 令牌
    jwt_ttl_s: int = _env_int("JWT_TTL_S", 28800)          # 8 小时
    jwt_algorithm: str = "HS256"             # 故意写死、不走环境：避免被换成 none/其他算法做令牌伪造
    # 首次启动时引导出的后台账号（局域网演示用；生产应改密码或直接建自己的账号）
    bootstrap_admin_user: str = _env("ADMIN_BOOTSTRAP_USER", "admin")
    bootstrap_admin_password: str = _env("ADMIN_BOOTSTRAP_PASSWORD")   # 无默认值：不配则 seed_admin 直接跳过，后台无人可登录

    # ── 服务监听 ───────────────────────────────────────────────────────────
    app_host: str = _env("APP_HOST", "127.0.0.1")   # 只回环，局域网/容器访问需显式改 0.0.0.0（run.py 用的就是这两个值）
    app_port: int = _env_int("APP_PORT", 8000)

    # ── 日志 ───────────────────────────────────────────────────────────────
    # 级别与格式都从环境走，本地调试改成 DEBUG 不用动代码。
    log_level: str = _env("LOG_LEVEL", "INFO")   # core/log.py 会 .upper() 后交给 logging；填错等级在 init_logging 时抛 ValueError
    # 使用项目专用变量名，避免 CI / IDE 常见的 LOG_FORMAT=json 污染
    # logging.Formatter 并导致应用在导入阶段崩溃。
    log_format: str = _env(
        "APP_LOG_FORMAT",
        "%(asctime)s %(levelname)-7s %(name)s [%(trace)s] %(message)s",
    )   # 里面的 %(trace)s 只有 _TraceFormatter 会注入；换成标准 Formatter 或自定义格式漏掉它都会让每条日志报 KeyError

    # ── Agent defaults ─────────────────────────────────────────────────────
    max_conversation_turns: int = _env_int("MAX_CONVERSATION_TURNS", 10)   # 会话记忆保留的轮数上限，超出丢最老的
    summary_trigger_turns: int = _env_int("SUMMARY_TRIGGER_TURNS", 8)      # 必须小于上一项，才能"先压缩成摘要再截断"，否则老对话直接被扔掉
    tool_retry_count: int = _env_int("TOOL_RETRY_COUNT", 3)   # 是"额外重试次数"：3 → 最多 4 次尝试（resilience.retry_with_backoff）
    tool_timeout_ms: int = _env_int("TOOL_TIMEOUT_MS", 5000)  # 单次尝试上限；executor 用 max(...,1) 兜底，填 0 不等于关掉超时
    # 上面两个默认值合起来是工具最坏预算：4×5s + (1+2+4)s 退避 ≈ 27s；超时只能取消协程，杀不掉线程里已经发起的外部写操作
    max_handoff_failures: int = _env_int("MAX_HANDOFF_FAILURES", 3)   # 当前没有任何代码读取它，转人工失败次数实际由 Agent 逻辑决定
    max_tool_rounds: int = _env_int("MAX_TOOL_ROUNDS", 3)   # base_agent 的路由用它决定继续 decide 还是转 generate，防止工具无限自循环
    max_collaboration_agents: int = _env_int("MAX_COLLABORATION_AGENTS", 2)   # coordinator 只取排序前 2 个领域（主 + 1 辅），再多成本线性上涨
    # 高危操作审批的等待上限，超时自动拒绝（原来只有"人工确认"，没有"超时拒绝"）
    approval_timeout_s: int = _env_int("APPROVAL_TIMEOUT_S", 900)   # 15 分钟；比较用 PG 的 now()（应用与库有实测约 1 分钟时钟漂移，Python 的钟不可信）

    # ── Retrieval defaults ─────────────────────────────────────────────────
    retrieval_top_k: int = _env_int("RETRIEVAL_TOP_K", 5)   # 融合后送重排/拼提示的条数；向量召回实际取 top_k*2 再按阈值筛
    # 向量候选的相似度下限。这个值跟 embedding 模型强相关，换模型必须重新标定：
    # 在 40 条评测集上用当前 qwen 向量实测 —— 0.75 只召回 52% 的相关切片、平均每问
    # 留下 0.9 条（等于把向量召回关掉）；0.50 召回 88%、平均保留 7.5 条；0.45 起
    # 进入平台期（90%）。取 0.50，剩下的交给重排序。
    similarity_threshold: float = _env_float("SIMILARITY_THRESHOLD", 0.50)   # 只过滤向量一路（COSINE 分数），BM25 命中不受它约束
    bm25_weight: float = _env_float("BM25_WEIGHT", 0.3)   # 加权融合系数
    vector_weight: float = _env_float("VECTOR_WEIGHT", 0.7)   # 两者设计上和为 1，但没有任何校验：各调一半会让总分尺度漂移

    def __post_init__(self) -> None:
        # 只能放在这里：类体里逐个字段求值时 llm_model_default 还没保证可用，且 role 集合要能在运行时扩
        for role in LLM_ROLES:
            self.llm_models[role] = _env(f"LLM_MODEL_{role.upper()}") or self.llm_model_default   # 变量名用大写岗位，与 .env.example 约定一致

    def model_for(self, role: str) -> str:
        """岗位对应的模型 id。未知岗位直接报错，避免静默用错模型。"""
        if role not in self.llm_models:
            # KeyError 而非回落默认模型：拼错的 Agent 名（agents 表里的手工行）会在这里暴露成 500
            raise KeyError(f"未配置的模型岗位: {role}，可选: {', '.join(LLM_ROLES)}")
        return self.llm_models[role]

    def key_for(self, kind: str) -> str:
        """各能力可以各用一份凭据；没单独配就回落到 chat 的凭据。"""
        if kind == "chat":
            return self.chat_api_key
        # 字典取值：kind 拼错直接 KeyError，不静默返回空串把缺凭据推到首次远程调用才炸
        dedicated = {"embedding": self.embedding_api_key, "rerank": self.rerank_api_key}[kind]
        return dedicated or self.chat_api_key

    def base_url_for(self, kind: str) -> str:
        if kind == "chat":
            return self.chat_base_url
        # embedding 没配就复用 chat 的兼容端点；rerank 反过来"没有独立端点概念"，恒用原生 RERANK_ENDPOINT
        return {"embedding": self.embedding_base_url or self.chat_base_url, "rerank": self.rerank_endpoint}[kind]


settings = Settings()   # 模块级单例：import 本模块即读一次环境，全项目共享同一快照
