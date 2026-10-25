"""混合检索：BM25 关键词 + Milvus 向量语义，加权融合。

【为什么不能只用向量】
中文客服问题往往很短（"退款多久到账"），关键词命中比语义更稳：
- 向量对"退款/退货/退钱"这种同义词有优势，但对精确术语（订单号、单号）容易漂；
- BM25 对精确词强，但换个说法就失效。
两路各取所长，按权重融合（默认 0.7 向量 / 0.3 BM25）。

【融合方式：加权分数相加，不是 RRF】
两路的分数都先归一化到"越大越相关"（BM25 原始分、余弦相似度），
再乘权重相加。选了简单可解释的方案，换 RRF 只需改 `_fusion`。

【阈值只作用于向量候选】
`similarity_threshold` 是给向量召回的"最低相关度"。它跟 embedding 模型强相关：
当前模型下 0.75 会把几乎所有候选滤掉（实测相关切片距离中位数 0.759），
所以默认值标定为 0.5（详见第 05 章）。BM25 不过阈值——关键词命中本身就是强信号。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.rag.embedder import embedder
from app.rag.vector_store import vector_store
from app.rag.bm25_store import bm25_store


@dataclass
class RetrievalResult:
    """检索结果的最小载体。只带下游需要的四个字段，便于序列化进 state。"""
    chunk_id: str
    content: str
    score: float
    source_document: str = ""


class HybridRetriever:
    """Hybrid retrieval combining BM25 keyword + vector semantic search."""

    def __init__(
        self,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        # 构造参数优先（便于评测/实验临时改），否则取配置。
        # 注意：用 `is not None` 判断而不是 `or`，否则传 0 会被当成"没传"
        # ——权重传 0 是合法配置（比如只想用向量）。
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight
        self.vector_weight = vector_weight if vector_weight is not None else settings.vector_weight
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        """两路各取 top_k*2 个候选，融合后截到 top_k。

        为什么要取 2 倍：融合会去重（同一 chunk 两路都命中），
        候选池太小会导致融合后不够 top_k 条。
        """
        bm25_results = await self._bm25_search(query)
        vector_results = await self._vector_search(query)
        return self._fusion(bm25_results, vector_results)

    async def _bm25_search(self, query: str) -> list[RetrievalResult]:
        """BM25 keyword search over in-memory index."""
        if not bm25_store.is_ready:
            # 索引没加载不是错误：降级为只用向量。调用方会在 trace 里看到 retrieval_count
            return []
        hits = bm25_store.search(query, top_k=self.top_k * 2)
        return [
            RetrievalResult(
                chunk_id=doc.chunk_id,
                content=doc.content,
                # rank-bm25 返回 numpy.float64；必须转 float，否则进 LangGraph state 时
                # msgpack 序列化会失败（这个坑真实踩过）
                score=float(score),
                source_document=doc.source_document,
            )
            for doc, score in hits
        ]

    async def _vector_search(self, query: str) -> list[RetrievalResult]:
        """Vector semantic search via Milvus."""
        if not vector_store.is_ready:
            return []
        qv = embedder.embed_query(query)      # 同步调用：embedding 协议就是同步的（见第 03 章）
        hits = vector_store.search(qv, top_k=self.top_k * 2)
        return [
            RetrievalResult(
                chunk_id=h["entity"].get("chunk_id", ""),
                content=h["entity"].get("content", ""),
                # COSINE: 1 = 完全一致；pymilvus 返回 numpy.float64，同样要转 float
                score=float(h["distance"]),
                source_document=h["entity"].get("source_document", ""),
            )
            for h in hits
            # 阈值过滤只在这里做：低于阈值的向量候选不参与融合，
            # 避免"语义上勉强相关"的噪声压过 BM25 的精确命中
            if float(h["distance"]) >= self.similarity_threshold
        ]

    def _fusion(
        self,
        bm25: list[RetrievalResult],
        vector: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """加权融合：同一 chunk 两路分数相加，按总分排序取 top_k。

        用 chunk_id 作为合并键：同一段文本被两路同时召回时，
        说明它既关键词命中又语义相关，理应排更前。
        """
        contents: dict[str, tuple[str, str]] = {}
        scores: dict[str, float] = {}

        for r in bm25:
            scores[r.chunk_id] = self.bm25_weight * r.score
            contents[r.chunk_id] = (r.content, r.source_document)

        for r in vector:
            current = scores.get(r.chunk_id, 0.0)
            scores[r.chunk_id] = current + self.vector_weight * r.score
            # 只在 BM25 没收录该 chunk 时补内容，避免同一段文本被覆盖两次
            if r.chunk_id not in contents:
                contents[r.chunk_id] = (r.content, r.source_document)

        sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)[: self.top_k]
        return [
            RetrievalResult(
                chunk_id=cid,
                content=contents[cid][0],
                score=round(scores[cid], 4),      # 保留 4 位，trace 里可读
                source_document=contents[cid][1],
            )
            for cid in sorted_ids
        ]
