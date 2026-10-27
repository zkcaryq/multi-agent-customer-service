"""重排序：调用 providers 层的 rerank 能力，ProviderError/空结果时使用分数惩罚策略降级。

不再在业务进程里加载 cross-encoder 权重（原 FlagEmbedding 分支），
模型换成配置项 RERANK_MODEL，默认百炼 qwen3-rerank。
"""

from __future__ import annotations

from app.config import settings
from app.core.log import get_logger
from app.providers.base import ProviderError
from app.providers.registry import get_rerank
from app.rag.retriever import RetrievalResult


# 无状态：不持有模型、连接或配置，provider 每次现取（get_rerank 内部按名字缓存实例）。
class Reranker:
    async def rerank(
        self, query: str, results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        # 候选是融合后已截到 top_k 的那批：重排只能在它们之间换序，补不回被截掉的结果。
        if not results:
            # 空输入原样返回同一个列表对象（不是新建），省掉一次拷贝。
            return results

        try:
            ranked = await get_rerank().rerank(
                # 把 query 与候选正文一起交给重排模型；返回 index 对应当前候选列表，不是知识库 chunk_id。
                # top_n 用 settings.rerank_top_n（默认 5）：模型路会按它截断，降级路不截断，条数口径不一致。
                query, [r.content for r in results], settings.rerank_top_n
            )
        except ProviderError as exc:
            # 只捕 ProviderError（含"未配置密钥""未知 provider 名"）：SDK 抛出的其他异常会直接上抛给 Agent 层。
            get_logger(__name__).warning("rerank provider 不可用，降级为融合分数排序: %s", exc)
            return self._score_rerank(results)

        if not ranked:
            # noop provider 正常返回空列表，也走降级：与抛错的区别只在有没有 warning。
            return self._score_rerank(results)

        out: list[RetrievalResult] = []
        for idx, score in ranked:
            # 防供应商返回越界下标；全部越界时 out 会为空，当前不会再触发降级。
            if not 0 <= idx < len(results):
                continue
            # 取得原 RetrievalResult 对象，下面直接改它的 score，调用方持有的同对象也会变化。
            item = results[idx]
            item.score = round(float(score), 4)
            out.append(item)
        # 供应商只返回 top_n 条：排名之外的候选在这里被丢弃，模型路因此带截断而降级路不带。
        return out

    def _score_rerank(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """降级排序：按融合分数降序，对同文档重复命中和排名靠后的结果做轻微惩罚。"""
        # 第一次排序按融合分；下面写的 i 就是这个"惩罚前名次"，与最终返回顺序不是同一回事。
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        seen_sources: dict[str, int] = {}
        for i, r in enumerate(sorted_results):
            src = r.source_document
            # 同一来源第一个罚 0、第二个罚 0.03；这是多样性启发式，不是另一个学习模型。
            penalty = seen_sources.get(src, 0) * 0.03
            # 同时扣排名与同源惩罚，可能改变顺序；降级分不可与真实 rerank 分直接横比。
            # max(0.0, …) 把负分夹到 0：低名次候选会一起变成 0 并并列，最终由稳定排序保留罚前相对序。
            r.score = round(max(0.0, r.score - i * 0.01 - penalty), 4)
            seen_sources[src] = seen_sources.get(src, 0) + 1

        # 降级分支未截断 rerank_top_n，通常依赖上游 retrieval_top_k 限制数量。
        # 第二次排序才反映惩罚后的序；返回新列表，但元素与入参是同一批对象（分数已被就地改掉）。
        return sorted(sorted_results, key=lambda r: r.score, reverse=True)
