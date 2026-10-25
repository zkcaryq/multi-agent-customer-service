"""重排序：调用 providers 层的 rerank 能力，失败则降级为融合分数原序。

不再在业务进程里加载 cross-encoder 权重（原 FlagEmbedding 分支），
模型换成配置项 RERANK_MODEL，默认百炼 qwen3-rerank。
"""

from __future__ import annotations

from app.config import settings
from app.core.log import get_logger
from app.providers.base import ProviderError
from app.providers.registry import get_rerank
from app.rag.retriever import RetrievalResult


class Reranker:
    async def rerank(
        self, query: str, results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        if not results:
            return results

        try:
            ranked = await get_rerank().rerank(
                query, [r.content for r in results], settings.rerank_top_n
            )
        except ProviderError as exc:
            get_logger(__name__).warning("rerank provider 不可用，降级为融合分数排序: %s", exc)
            return self._score_rerank(results)

        if not ranked:
            return self._score_rerank(results)

        out: list[RetrievalResult] = []
        for idx, score in ranked:
            if not 0 <= idx < len(results):
                continue
            item = results[idx]
            item.score = round(float(score), 4)
            out.append(item)
        return out

    def _score_rerank(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """降级排序：按融合分数降序，对同文档重复命中和排名靠后的结果做轻微惩罚。"""
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        seen_sources: dict[str, int] = {}
        for i, r in enumerate(sorted_results):
            src = r.source_document
            penalty = seen_sources.get(src, 0) * 0.03
            r.score = round(max(0.0, r.score - i * 0.01 - penalty), 4)
            seen_sources[src] = seen_sources.get(src, 0) + 1

        return sorted(sorted_results, key=lambda r: r.score, reverse=True)
