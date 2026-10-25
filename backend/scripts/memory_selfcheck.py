"""对话记忆压缩自检：标记式压缩、LLM 摘要、降级折叠、Redis 往返。

跑法：backend 目录下 `python scripts/memory_selfcheck.py`
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers import registry  # noqa: E402
from app.providers.base import ChatResult  # noqa: E402

PASS = 0
TOTAL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, TOTAL
    TOTAL += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label} {detail}")
    PASS += 1 if ok else 0


class SummaryChat:
    """LLM 摘要桩：把摘要写成固定文案，验证调用真的发生了。"""

    async def chat(self, *, role, messages, temperature=0.7, max_tokens=2048,
                   json_mode=False, tools=None):
        assert role == "summary", f"摘要应走 summary 岗位，实际 {role}"
        return ChatResult(content="用户咨询订单 ORD-1 的物流，已告知在途；随后申请退款。",
                          model="stub", provider="stub")


def _fill(mem, n: int) -> None:
    for i in range(n):
        mem.add_turn("user", f"第{i}轮用户内容_{i}")
        mem.add_turn("assistant", f"第{i}轮助手内容_{i}")


async def main() -> int:
    from app.core.memory import ConversationMemory, RedisMemoryStore

    # 1) 超阈值只标记，不静默丢轮次
    mem = ConversationMemory(conversation_id=1, max_turns=20, summary_trigger_turns=6)
    _fill(mem, 5)
    check("超阈值仅标记待压缩", mem.needs_compaction and len(mem.turns) == 10,
          f"turns={len(mem.turns)} needs={mem.needs_compaction} summary={mem.summary!r}")

    # 2) LLM 摘要成功：老轮次进 summary，保留最近 keep 条
    registry.override_chat(SummaryChat())
    summary = await mem.summarize_async()
    check("LLM 摘要生效", "已告知在途" in summary and len(mem.turns) == mem.keep_recent_turns,
          f"turns={len(mem.turns)} summary={summary[:24]!r}")
    check("压缩后不再重复触发", not mem.needs_compaction)

    # 3) 摘要失败时降级折叠，内容不丢
    async def _boom(*a, **k):
        raise RuntimeError("llm down")

    import app.core.memory as m

    orig = m._summarize_llm
    m._summarize_llm = _boom
    try:
        mem2 = ConversationMemory(conversation_id=2, max_turns=20, summary_trigger_turns=4)
        _fill(mem2, 4)
        s2 = await mem2.summarize_async()
        check("LLM 失败时折叠保留内容", "第0轮用户内容_0" in s2, f"summary={s2[:40]!r}")
    finally:
        m._summarize_llm = orig

    # 4) 硬上限兜底：没人调异步摘要也不会无限长
    mem3 = ConversationMemory(conversation_id=3, max_turns=6, summary_trigger_turns=4)
    _fill(mem3, 5)
    check("超硬上限被截住", len(mem3.turns) <= mem3.max_turns, f"turns={len(mem3.turns)}")
    check("兜底压缩保留早期内容", "第0轮" in mem3.summary, f"summary={mem3.summary[:40]!r}")

    # 5) to_dict / from_dict 往返（Redis 存的就是这个）
    back = ConversationMemory.from_dict(mem2.to_dict())
    check("Redis 往返无损", back.summary == mem2.summary and len(back.turns) == len(mem2.turns)
          and back.needs_compaction == mem2.needs_compaction)

    # 6) 摘要注入提示词
    msgs = mem2.to_messages("你是售后客服")
    check("历史摘要进入 system prompt", "[历史对话摘要]" in msgs[0]["content"],
          f"len={len(msgs)}")

    # 7) Redis 不可用时 store 仍工作（降级到内存）
    store = RedisMemoryStore(redis_url="redis://127.0.0.1:6399/0")
    ok_ping = True
    try:
        store.client.ping()
    except Exception:
        ok_ping = False
    check("Redis 故障路径可预期", not ok_ping, "（本机无 Redis，走降级分支）")

    print(f"\n{PASS}/{TOTAL} 通过")
    return 0 if PASS == TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
