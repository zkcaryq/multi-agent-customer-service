"""评测指标：纯函数，不碰数据库和网络，便于单测与复算。"""

from __future__ import annotations

from dataclasses import dataclass      # 用普通 dataclass：指标不做运行时校验、不裁范围，只当值容器传给汇总方


@dataclass
class RetrievalMetrics:
    hit: bool                          # 前 k 里"至少命中一条"，0/1 语义；被 accuracy([m.hit …]) 当布尔标志求命中率
    recall: float                      # 0..1，分母=标注相关集合大小 len(relevant_ids)
    precision: float                   # 0..1，分母=实际参与评估的候选数 len(top)，不是固定的 k
    reciprocal_rank: float             # 单条用例的首条相关倒数 1/rank；跨用例 mean() 之后才是 MRR


def retrieval_case(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> RetrievalMetrics:
    """单条检索用例。relevant 为空视为不可判定（不计入统计，由调用方过滤）。"""
    # k 由调用方传入（评测脚本用的是 HybridRetriever 自己的 top_k），本函数不读配置，保证纯函数可复算
    if not relevant_ids:
        # 全 0 只是"不可判定"的哨兵：调用方必须以 relevant_ids 非空为过滤条件，否则这些 0 会把均值拉低
        return RetrievalMetrics(hit=False, recall=0.0, precision=0.0, reciprocal_rank=0.0)

    # 只评前 k 个候选；当前调用方传融合结果，不会在这里自动执行 rerank。
    top = retrieved_ids[:k]            # 候选不足 k 条时切片不报错，分母随之变小——precision 口径要带上返回条数一起看
    # 按相关 ID 集判断命中；假设候选已去重，否则重复 ID 会夸大 recall/precision。
    found = [i for i in top if i in relevant_ids]   # 交集按 top 的顺序保留，供下面求名次；分母用的是 relevant_ids，不是 found
    # 首次相关候选的名次从 1 开始；没有相关项记 0，避免 reciprocal rank 除零。
    rank = next((n + 1 for n, i in enumerate(top) if i in relevant_ids), 0)   # 只取第一条相关项，后面的命中不再影响 MRR

    return RetrievalMetrics(
        hit=bool(found),               # 与 recall 同源但口径不同：命中一条即 True，哪怕只覆盖相关的十分之一
        # 召回率分母是已标注相关集合大小，和 hit@k 的“有没有至少一条”不同。
        recall=round(len(found) / len(relevant_ids), 4),   # relevant_ids 为空集时这行会 ZeroDivisionError，靠上面的提前 return 兜住
        # 本实现分母为实际返回数量，不固定为 k；解释指标时要带上这一定义。
        precision=round(len(found) / len(top), 4) if top else 0.0,   # 空候选（检索返回 []）走 0.0 分支，不做除零
        reciprocal_rank=round(1.0 / rank, 4) if rank else 0.0,       # rank 为 0 表示相关项落在 k 名之外（或不存在），记 0 而非 1/0
    )


def accuracy(flags: list[bool | None]) -> float | None:
    """None 表示"未参与评测"，不计入分母，避免把没跑的项算成错。"""
    # None 从分母剔除，False 才是已评但失败；有效样本数必须随百分比一起报告。
    vals = [f for f in flags if f is not None]   # 典型来源：没配模型时 tool_correct 整列是 None，不能算成答错
    if not vals:
        return None                    # 返回 None（落库成 NULL）而不是 0.0：区分"没评"与"评了全错"，0.0 会被误读成 0 分
    # accuracy 返回 0..100 的百分数，retrieval_case 的 recall/precision 则是 0..1。
    return round(sum(1 for f in vals if f) / len(vals) * 100, 2)   # *100 是它的口径；与 mean() 的 0..1 不能直接同列比较


def mean(values: list[int | float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]   # 同样剔除 None；float() 让 int/float 混合列表口径一致
    if not vals:
        return None                    # 与 accuracy 一致：空样本返回 None，避免把"没测"报成 0 分
    return round(sum(vals) / len(vals), 4)               # 微平均：每条用例等权，不按 relevant_count 加权
