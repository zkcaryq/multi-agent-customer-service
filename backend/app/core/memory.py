"""会话记忆：最近 N 条消息 + LLM 摘要压缩 + Redis 持久化。

【为什么不能把历史全塞给模型】
多轮客服对话动辄十几轮，全量拼进提示词会导致：
token 成本线性增长、关键信息被稀释、超出上下文窗口。所以要"压缩"。

【两级结构】
- `turns`：最近的消息（默认保留 4 条，不是 4 对问答，也不保证所有细节保留）；
- `summary`：更早内容的摘要（LLM 生成；LLM 不可用时退化为字符串折叠）。

【三个刻意的设计】
1. `add_turn` **只标记不压缩**：早期版本在 add_turn 里直接丢老轮次，
   导致 LLM 摘要拿到的是空历史——所以压缩统一放在 `summarize_async` 或硬上限兜底里。
2. **摘要失败不丢内容**：LLM 调用失败时退回 `_fold`（把老轮次拼成文本），
   每条折叠最多 120 字，合并可能截掉旧内容；降级也会有信息损失。
3. **合并用追加 + 截断**：原来是"取更长的那个"，多次折叠后最早那段摘要会被整段丢掉。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.log import get_logger


@dataclass
class ConversationMemory:
    """Holds the current session context for a conversation.

    Short-term memory: last N turns, with LLM summary compression when exceeding
    the trigger threshold. Summary is persisted so long-running sessions stay compact.

    压缩只在 summarize_async / 硬上限兜底里发生；add_turn 只做标记，
    否则老轮次会被静默丢掉，LLM 摘要拿到的就是空历史。
    """

    # 作为 Redis 键和进程内字典键，与 PostgreSQL 会话主键对应。
    conversation_id: int
    # 每个会话独立分配列表；不能把可变的 [] 直接用作所有实例共享的默认值。
    turns: list[Turn] = field(default_factory=list)
    # 摘要是派生上下文，不是可靠业务事实源；订单状态仍应查数据库工具。
    summary: str = ""
    # 阈值都比较 len(turns)，一个问答对通常占两条消息。
    max_turns: int = 10               # 硬上限：超过就强制折叠（不等异步摘要）
    summary_trigger_turns: int = 8    # 软阈值：超过就标记"该压缩了"
    keep_recent_turns: int = 4        # 压缩时保留最近几轮原文
    needs_compaction: bool = False    # 标记位，由 summarize_async 消费

    def add_turn(self, role: str, content: str) -> None:
        """追加一条消息。只做标记与硬上限兜底，不在这里调 LLM（同步函数没法 await）。"""
        # 先保存当前消息，再判断阈值；第九条追加后满足 >8，而不是 >=8。
        self.turns.append(Turn(role=role, content=content))
        # 默认四对问答有八条消息，第五次 user 提问才在正常流程中触发软阈值。
        if len(self.turns) > self.summary_trigger_turns:
            # 超软阈值：标记待压缩，真正的摘要由 _generate 里的 await summarize_async 完成
            # 只做待处理标记，让同步 add_turn 不承担模型网络调用。
            self.needs_compaction = True
        # 即使调用方忘记 summarize_async，也在第十一条消息时触发同步兜底。
        if len(self.turns) > self.max_turns:
            # 硬上限：调用方没跑异步摘要时，也只能折叠而不是丢弃
            # 使用文本折叠而非调用 LLM；硬上限是消息条数，单条超长文本仍可能占很多 token。
            self._compress_sync()

    def _fold(self, turns: list[Turn]) -> str:
        """无 LLM 的降级摘要：把轮次拼成一段文本。截断每条到 120 字，控制长度。"""
        return "对话摘要: " + "; ".join(f"[{t.role}]: {t.content[:120]}" for t in turns)

    def _compress_sync(self) -> None:
        """无 LLM 的降级压缩：把早期消息折进摘要文本；长度裁剪会损失细节。"""
        # 负数切片取得最近四条之前的消息；切分按条数，边界可能落在问答对中间。
        older = self.turns[:-self.keep_recent_turns]
        # 没有旧消息就直接返回，避免无内容时也付一次摘要调用费用。
        if not older:
            return
        # 保留最近四条原文；后续 assistant 追加后可能暂时变成五条。
        self.turns = self.turns[-self.keep_recent_turns:]
        # 将旧摘要与截断文本合并；原始消息已移出 turns，摘要质量下降会影响后续生成。
        self.summary = _merge_summaries(self.summary, self._fold(older))

    async def summarize_async(self) -> str:
        """用 LLM 把早期轮次并进摘要；失败则退回字符串折叠。

        返回新的 summary。未触发压缩时原样返回。
        注意：即使没触发，也返回当前 summary（调用方拿它拼 system prompt）。
        """
        # 既没标记、又没超阈值才跳过；允许调用者在已有标记时主动压缩。
        if not self.needs_compaction and len(self.turns) <= self.summary_trigger_turns:
            return self.summary

        # 负数切片取得最近四条之前的消息；切分按条数，边界可能落在问答对中间。
        older = self.turns[:-self.keep_recent_turns]
        # 没有旧消息就直接返回，避免无内容时也付一次摘要调用费用。
        if not older:
            # 轮次还不够多，没什么可压的
            return self.summary

        # 先切分：保留最近 keep_recent_turns 轮原文，其余交给摘要
        # 保留最近四条原文；后续 assistant 追加后可能暂时变成五条。
        self.turns = self.turns[-self.keep_recent_turns:]
        # 在模型调用前清标记，成功或降级都视为完成本次压缩；取消时仍可能留下已裁剪状态。
        self.needs_compaction = False

        try:
            # 输入旧摘要加这批旧消息，要求模型增量整合；没有额外校验订单号是否遗漏。
            new_summary = await _summarize_llm(self.summary, older)
        except Exception:
            # 以空字符串统一表示摘要失败，下面转到确定性的折叠路径。
            new_summary = ""
        if new_summary:
            # LLM 摘要成功：用它替换旧摘要（提示词里要求"把新信息融入已有摘要"）
            # 成功摘要整体替换旧摘要；LLM 的 256 token 输出限制和折叠的 1200 字符限制是两套边界。
            self.summary = new_summary
        else:
            # 失败：追加截断后的历史文本，保留部分线索，但不是无损压缩。
            # 将旧摘要与截断文本合并；原始消息已移出 turns，摘要质量下降会影响后续生成。
            self.summary = _merge_summaries(self.summary, self._fold(older))
        return self.summary

    def to_messages(self, system_prompt: str = "") -> list[dict[str, str]]:
        """Build the message list ready for LLM injection.

        顺序：system（角色+知识+摘要）→ 最近若干轮原文。
        摘要放在 system 里而不是单独一条 user 消息，是为了让模型把它当"背景事实"。
        """
        # 构造模型协议消息；存储结构 Turn 不直接传给供应商。
        messages: list[dict[str, str]] = []
        # 只有给了非空 system_prompt 才会把摘要拼进去；空提示词调用会遗漏 summary。
        if system_prompt:
            prompt = system_prompt
            if self.summary:
                prompt += f"\n\n[历史对话摘要]\n{self.summary}"
            # 用户历史被摘要后放进 system 并不会自动变可信，仍存在历史内容污染提示词的风险。
            messages.append({"role": "system", "content": prompt})
        # 保持消息原顺序；不在这里补齐可能被切开的 user/assistant 配对。
        for t in self.turns:
            messages.append({"role": t.role, "content": t.content})
        return messages

    def to_messages_for_classification(self) -> str:
        """Return the last user message for intent classification."""
        # 从后往前找最近 user 文本；当前 router 没调用这个辅助方法。
        for t in reversed(self.turns):
            if t.role == "user":
                return t.content
        return ""

    def reset(self) -> None:
        # 只清现有消息列表；reset 当前未重置 needs_compaction，要留意标记残留。
        self.turns.clear()
        self.summary = ""

    def to_dict(self) -> dict:
        """序列化成可 JSON 存储的结构（Redis 用）。"""
        return {
            "conversation_id": self.conversation_id,
            "turns": [{"role": t.role, "content": t.content, "metadata": t.metadata} for t in self.turns],
            "summary": self.summary,
            "max_turns": self.max_turns,
            "summary_trigger_turns": self.summary_trigger_turns,
            "keep_recent_turns": self.keep_recent_turns,
            "needs_compaction": self.needs_compaction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConversationMemory:
        """从 Redis 恢复。缺字段用默认值，兼容老版本的存储结构。"""
        mem = cls(
            conversation_id=data["conversation_id"],
            summary=data.get("summary", ""),
            max_turns=data.get("max_turns", 10),
            summary_trigger_turns=data.get("summary_trigger_turns", 8),
            keep_recent_turns=data.get("keep_recent_turns", 4),
            needs_compaction=data.get("needs_compaction", False),
        )
        # 从 JSON 字典恢复 Turn 对象；默认 metadata 兼容以前没有该字段的快照。
        mem.turns = [
            Turn(role=t["role"], content=t["content"], metadata=t.get("metadata", {}))
            for t in data.get("turns", [])
        ]
        return mem


@dataclass
class Turn:
    """一条消息（一次 user 或 assistant）。role 沿用 OpenAI 的取值：user / assistant / tool。"""
    role: str  # "user" | "assistant" | "tool"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── LLM summarization ────────────────────────────────────────────────────────

_SUMMARY_MAX_CHARS = 1200  # 降级折叠时摘要的保留上限

# 摘要提示词强调"保留订单号/商品名/关键结论"——
# 这些正是后续多轮里最容易丢、也最影响业务判断的信息。
_SUMMARY_PROMPT = """你是一个对话摘要助手。请用 1-2 句话提取以下对话的关键信息。

要求：
- 突出用户的核心诉求和已提供的信息（如订单号、商品名、问题描述）
- 突出助手已给出的关键结论或操作
- 如果已有历史摘要，将新信息融入进去
- 只返回摘要内容，不要解释

已有摘要: {previous}
"""


async def _summarize_llm(previous_summary: str, turns: list[Turn]) -> str:
    """生成对话摘要；模型由 LLM_MODEL_SUMMARY 配置。

    失败返回空串（不是抛异常）：调用方据此走 _fold 降级。
    """
    # 摘要模型同时看到角色和原文，避免把用户诉求误当成助手已经完成的操作。
    dialogue = "\n".join(f"[{t.role}]: {t.content}" for t in turns)

    try:
        from app.providers import get_chat

        result = await get_chat().chat(
            role="summary",            # 摘要岗位 → 便宜的小模型
            messages=[
                {"role": "system", "content": _SUMMARY_PROMPT.format(previous=previous_summary or "无")},
                {"role": "user", "content": dialogue},
            ],
            temperature=0.2,           # 摘要要稳定，不要发散
            max_tokens=256,
        )
        return result.content.strip()
    except Exception:
        return ""


def _merge_summaries(old: str, new: str) -> str:
    """追加式合并，超过上限时裁掉最早的片段。

    原来的"取更长的那个"会在多次折叠后把最早那段摘要整丢掉。
    LLM 摘要路径不受影响，这里兜的是 LLM 不可用时的连续折叠。

    取尾部而不是头部：最新的信息对当前轮更有用。
    """
    # 第一次合并直接返回 new，不经过尾部裁剪；因此 1200 字符不是所有分支的硬上限。
    if not old:
        return new
    # 仅用子串包含避免重复追加，不是语义去重。
    if new in old:
        return old
    # 用分隔符保留先后关系；旧摘要仍会被下一行的长度上限裁剪。
    merged = f"{old} | {new}"
    # 保留末尾 1200 字符，不是 1200 token；较早的订单号可能被截断。
    return merged[-_SUMMARY_MAX_CHARS:]


# ── Memory stores ─────────────────────────────────────────────────────────────

class MemoryStore:
    """In-memory store — fast, suitable for dev / single-process."""

    def __init__(self) -> None:
        # 内存降级没有 TTL，也不跨进程共享；进程重启即丢。
        self._store: dict[int, ConversationMemory] = {}

    def get_or_create(
        self, conversation_id: int, max_turns: int = 10, summary_trigger: int = 8
    ) -> ConversationMemory:
        # 内存版已有对象不会自动更新 max_turns 等配置；和 Redis 恢复路径的行为有差别。
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationMemory(
                conversation_id=conversation_id,
                max_turns=max_turns,
                summary_trigger_turns=summary_trigger,
            )
        # 返回同一个可变对象，并发处理相同会话时仍可能竞争。
        return self._store[conversation_id]

    def remove(self, conversation_id: int) -> None:
        # 不存在时无操作，不抛 KeyError。
        self._store.pop(conversation_id, None)


class RedisMemoryStore:
    """Redis 中共享的记忆快照；没有会话锁或 CAS，同会话并发写可能覆盖，不能据此声称生产并发安全。"""

    # 键前缀带项目名：同一台 Redis 可能被多个项目共用（本项目就是分库/分前缀隔离）
    _KEY_PREFIX = "ma-cs:memory:"
    _TTL = 86400  # 24 hours

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or settings.redis_url
        # 延迟创建同步 Redis 客户端；同步 get/setex 在异步请求里会阻塞事件循环。
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            import redis
            # decode_responses=True：读出来就是 str，不用手动 decode
            self._client = redis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _key(self, conversation_id: int) -> str:
        # 键只有会话号，因此不同领域 Agent 共享该会话记忆，而检查点按 agent:conversation 分开。
        return f"{self._KEY_PREFIX}{conversation_id}"

    def get_or_create(
        self, conversation_id: int, max_turns: int = 10, summary_trigger: int = 8
    ) -> ConversationMemory:
        key = self._key(conversation_id)
        try:
            # 每次读取 Redis 快照；没有本地缓存/乐观锁，也不会仅凭 GET 刷新 TTL。
            raw = self.client.get(key)
            if raw:
                # JSON 是这里 Redis 快照的编码；不能推广成 LangGraph 所有检查点都用 JSON。
                data = json.loads(raw)
                # 每次读取重新构造对象，不是持有数据库连接的 ORM 对象。
                mem = ConversationMemory.from_dict(data)
                # 上限/阈值以当前配置为准（改配置后老记忆也能跟上），不沿用存储里的旧值
                mem.max_turns = max_turns
                mem.summary_trigger_turns = summary_trigger
                return mem
        except Exception:
            # Redis 抖动时不能让对话失败：退化成"这条会话本次无历史"
            pass

        mem = ConversationMemory(
            conversation_id=conversation_id,
            max_turns=max_turns,
            summary_trigger_turns=summary_trigger,
        )
        # 初次创建和 persist 都写回完整快照并刷新 24 小时 TTL。
        self._save(mem)
        return mem

    def _save(self, memory: ConversationMemory) -> None:
        try:
            key = self._key(memory.conversation_id)
            # setex：带 TTL 写入，避免无人会话的键永久堆积
            # 一次替换整段 JSON，不会原子合并另一请求刚追加的消息；最后一次写可能覆盖前一次。
            self.client.setex(key, self._TTL, json.dumps(memory.to_dict(), ensure_ascii=False))
        except Exception:
            pass

    def persist(self, memory: ConversationMemory) -> None:
        """Persist a memory instance back to Redis (call after updates)."""
        self._save(memory)

    def remove(self, conversation_id: int) -> None:
        try:
            self.client.delete(self._key(conversation_id))
        except Exception:
            pass


# ── Global singleton (auto-selects Redis if available) ────────────────────────

def _create_store():
    """启动时探测一次 Redis：可用就用 Redis，不可用退回进程内存。

    这是"能跑"和"能持久化"之间的降级：单机演示无所谓，
    多进程/重启场景下内存版会丢上下文，所以启动日志会明确打印用的哪个。
    """
    store = RedisMemoryStore()
    try:
        # 只在模块导入选择实现时探测一次；选 Redis 后运行中失败会返回新空记忆，不会切成共享内存缓存。
        store.client.ping()
        get_logger(__name__).info("using Redis memory store")
        return store
    except Exception:
        get_logger(__name__).warning("Redis unavailable, using in-memory store")
        # 启动时退回内存后不会自动重新探测 Redis；恢复服务通常需要重启当前进程。
        return MemoryStore()


memory_store: MemoryStore | RedisMemoryStore = _create_store()
