"""会话记忆：最近 N 轮 + LLM 摘要压缩 + Redis 持久化。

【为什么不能把历史全塞给模型】
多轮客服对话动辄十几轮，全量拼进提示词会导致：
token 成本线性增长、关键信息被稀释、超出上下文窗口。所以要"压缩"。

【两级结构】
- `turns`：最近的完整轮次（默认保留 4 轮，保证当前话题的细节不丢）；
- `summary`：更早内容的摘要（LLM 生成；LLM 不可用时退化为字符串折叠）。

【三个刻意的设计】
1. `add_turn` **只标记不压缩**：早期版本在 add_turn 里直接丢老轮次，
   导致 LLM 摘要拿到的是空历史——所以压缩统一放在 `summarize_async` 或硬上限兜底里。
2. **摘要失败不丢内容**：LLM 调用失败时退回 `_fold`（把老轮次拼成文本），
   宁可摘要粗糙也不能把用户说过的话弄丢。
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

    conversation_id: int
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    max_turns: int = 10               # 硬上限：超过就强制折叠（不等异步摘要）
    summary_trigger_turns: int = 8    # 软阈值：超过就标记"该压缩了"
    keep_recent_turns: int = 4        # 压缩时保留最近几轮原文
    needs_compaction: bool = False    # 标记位，由 summarize_async 消费

    def add_turn(self, role: str, content: str) -> None:
        """追加一轮。只做标记与硬上限兜底，不在这里调 LLM（同步函数没法 await）。"""
        self.turns.append(Turn(role=role, content=content))
        if len(self.turns) > self.summary_trigger_turns:
            # 超软阈值：标记待压缩，真正的摘要由 _generate 里的 await summarize_async 完成
            self.needs_compaction = True
        if len(self.turns) > self.max_turns:
            # 硬上限：调用方没跑异步摘要时，也只能折叠而不是丢弃
            self._compress_sync()

    def _fold(self, turns: list[Turn]) -> str:
        """无 LLM 的降级摘要：把轮次拼成一段文本。截断每条到 120 字，控制长度。"""
        return "对话摘要: " + "; ".join(f"[{t.role}]: {t.content[:120]}" for t in turns)

    def _compress_sync(self) -> None:
        """无 LLM 的降级压缩：把早期轮次折进摘要文本，不丢内容。"""
        older = self.turns[:-self.keep_recent_turns]
        if not older:
            return
        self.turns = self.turns[-self.keep_recent_turns:]
        self.summary = _merge_summaries(self.summary, self._fold(older))

    async def summarize_async(self) -> str:
        """用 LLM 把早期轮次并进摘要；失败则退回字符串折叠。

        返回新的 summary。未触发压缩时原样返回。
        注意：即使没触发，也返回当前 summary（调用方拿它拼 system prompt）。
        """
        if not self.needs_compaction and len(self.turns) <= self.summary_trigger_turns:
            return self.summary

        older = self.turns[:-self.keep_recent_turns]
        if not older:
            # 轮次还不够多，没什么可压的
            return self.summary

        # 先切分：保留最近 keep_recent_turns 轮原文，其余交给摘要
        self.turns = self.turns[-self.keep_recent_turns:]
        self.needs_compaction = False

        try:
            new_summary = await _summarize_llm(self.summary, older)
        except Exception:
            new_summary = ""
        if new_summary:
            # LLM 摘要成功：用它替换旧摘要（提示词里要求"把新信息融入已有摘要"）
            self.summary = new_summary
        else:
            # 失败：追加折叠文本，保证历史不丢
            self.summary = _merge_summaries(self.summary, self._fold(older))
        return self.summary

    def to_messages(self, system_prompt: str = "") -> list[dict[str, str]]:
        """Build the message list ready for LLM injection.

        顺序：system（角色+知识+摘要）→ 最近若干轮原文。
        摘要放在 system 里而不是单独一条 user 消息，是为了让模型把它当"背景事实"。
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            prompt = system_prompt
            if self.summary:
                prompt += f"\n\n[历史对话摘要]\n{self.summary}"
            messages.append({"role": "system", "content": prompt})
        for t in self.turns:
            messages.append({"role": t.role, "content": t.content})
        return messages

    def to_messages_for_classification(self) -> str:
        """Return the last user message for intent classification."""
        for t in reversed(self.turns):
            if t.role == "user":
                return t.content
        return ""

    def reset(self) -> None:
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
        mem.turns = [
            Turn(role=t["role"], content=t["content"], metadata=t.get("metadata", {}))
            for t in data.get("turns", [])
        ]
        return mem


@dataclass
class Turn:
    """一轮对话。role 沿用 OpenAI 的取值：user / assistant / tool。"""
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
    if not old:
        return new
    if new in old:
        return old
    merged = f"{old} | {new}"
    return merged[-_SUMMARY_MAX_CHARS:]


# ── Memory stores ─────────────────────────────────────────────────────────────

class MemoryStore:
    """In-memory store — fast, suitable for dev / single-process."""

    def __init__(self) -> None:
        self._store: dict[int, ConversationMemory] = {}

    def get_or_create(
        self, conversation_id: int, max_turns: int = 10, summary_trigger: int = 8
    ) -> ConversationMemory:
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationMemory(
                conversation_id=conversation_id,
                max_turns=max_turns,
                summary_trigger_turns=summary_trigger,
            )
        return self._store[conversation_id]

    def remove(self, conversation_id: int) -> None:
        self._store.pop(conversation_id, None)


class RedisMemoryStore:
    """Redis-backed memory store — persistent, suitable for multi-process / production."""

    # 键前缀带项目名：同一台 Redis 可能被多个项目共用（本项目就是分库/分前缀隔离）
    _KEY_PREFIX = "ma-cs:memory:"
    _TTL = 86400  # 24 hours

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or settings.redis_url
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            import redis
            # decode_responses=True：读出来就是 str，不用手动 decode
            self._client = redis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _key(self, conversation_id: int) -> str:
        return f"{self._KEY_PREFIX}{conversation_id}"

    def get_or_create(
        self, conversation_id: int, max_turns: int = 10, summary_trigger: int = 8
    ) -> ConversationMemory:
        key = self._key(conversation_id)
        try:
            raw = self.client.get(key)
            if raw:
                data = json.loads(raw)
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
        self._save(mem)
        return mem

    def _save(self, memory: ConversationMemory) -> None:
        try:
            key = self._key(memory.conversation_id)
            # setex：带 TTL 写入，避免无人会话的键永久堆积
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
        store.client.ping()
        get_logger(__name__).info("using Redis memory store")
        return store
    except Exception:
        get_logger(__name__).warning("Redis unavailable, using in-memory store")
        return MemoryStore()


memory_store: MemoryStore | RedisMemoryStore = _create_store()
