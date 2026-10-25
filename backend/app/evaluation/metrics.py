"""评测指标：纯函数，不碰数据库和网络，便于单测与复算。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    hit: bool
    recall: float
    precision: float
    reciprocal_rank: float


def retrieval_case(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> RetrievalMetrics:
    """单条检索用例。relevant 为空视为不可判定（不计入统计，由调用方过滤）。"""
    if not relevant_ids:
        return RetrievalMetrics(hit=False, recall=0.0, precision=0.0, reciprocal_rank=0.0)

    top = retrieved_ids[:k]
    found = [i for i in top if i in relevant_ids]
    rank = next((n + 1 for n, i in enumerate(top) if i in relevant_ids), 0)

    return RetrievalMetrics(
        hit=bool(found),
        recall=round(len(found) / len(relevant_ids), 4),
        precision=round(len(found) / len(top), 4) if top else 0.0,
        reciprocal_rank=round(1.0 / rank, 4) if rank else 0.0,
    )


def accuracy(flags: list[bool | None]) -> float | None:
    """None 表示"未参与评测"，不计入分母，避免把没跑的项算成错。"""
    vals = [f for f in flags if f is not None]
    if not vals:
        return None
    return round(sum(1 for f in vals if f) / len(vals) * 100, 2)


def mean(values: list[int | float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)
