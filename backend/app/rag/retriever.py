"""混合检索：BM25 关键词 + Milvus 向量语义，加权融合。

【为什么不能只用向量】
中文客服问题往往很短（"退款多久到账"），关键词命中比语义更稳：
- 向量对"退款/退货/退钱"这种同义词有优势，但对精确术语（订单号、单号）容易漂；
- BM25 对精确词强，但换个说法就失效。
两路各取所长，按权重融合（默认 0.7 向量 / 0.3 BM25）。

【融合方式：加权分数相加，不是 RRF】
BM25 在 Bm25Store.search 里除以本次最大正分；向量直接使用 COSINE 相似度。
两者都越大越相关，但并非同分布、也不是概率，再按权重相加。选了简单可解释的方案，换 RRF 只需改 `_fusion`。

【阈值只作用于向量候选】
`similarity_threshold` 是给向量召回的"最低相关度"。它跟 embedding 模型强相关：
本项目配置使用 0.5；阈值效果必须通过当前模型/数据的标注集验证。
仅凭历史中位数不能推导 0.75 会过滤几乎所有结果。BM25 不走这个阈值。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings  # 权重/top_k/阈值都来自它，阈值 0.50 的标定过程写在 config.py
from app.rag.embedder import embedder  # 同步接口：查询向量在这里取，会阻塞当前事件循环
from app.rag.vector_store import vector_store  # Milvus 单例（COSINE 相似度）
from app.rag.bm25_store import bm25_store  # 进程内 BM25 单例（多 worker 各持一份）


@dataclass
class RetrievalResult:
    """检索结果的最小载体。只带下游需要的四个字段，便于序列化进 state。"""
    # 融合去重键："来源#chunkN"。向量命中若缺该字段会取到空串，多条空键候选会在 dict 里互相覆盖。
    chunk_id: str
    content: str
    # 这一个字段先后承载四种量纲：BM25 归一分 / COSINE 相似度 / 融合加权和 / rerank 分，不可跨阶段横比。
    score: float
    # 默认空串：重排的"同源惩罚"就是按它分组，来源缺失的候选会被当成同一份文档互相扣分。
    # 注意这里没有 heading 字段，虽然两路存储都保存了 heading，引用展示只能回落到正文。
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
        # 两个权重不校验和为 1，也没做归一化：默认 0.3/0.7 的融合分上限只有 1.0，改配置后绝对值不可比。
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        # 阈值只被 _vector_search 使用；BM25 路不受任何分数下限约束，关键词命中一定进融合。
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
        # 代码顺序先 BM25 后向量，不是 gather 并行召回；async 函数里仍可调用同步阻塞代码。
        # 两路耗时直接相加，整体没有超时保护：BM25 打分是 CPU 阻塞、向量路还叠加一次 embedding 网络请求。
        bm25_results = await self._bm25_search(query)
        # 若向量连接或 embedding 抛错，这里整体抛出；上层会清空结果，已取得的 BM25 也不会单独返回。
        vector_results = await self._vector_search(query)
        # 只有两路正常完成（允许某路因未就绪返回空）才融合；“未就绪”和“调用异常”不能混同。
        return self._fusion(bm25_results, vector_results)

    async def _bm25_search(self, query: str) -> list[RetrievalResult]:
        """BM25 keyword search over in-memory index."""
        if not bm25_store.is_ready:
            # 索引没加载不是错误：降级为只用向量。调用方会在 trace 里看到 retrieval_count
            # 未就绪的成因不止"没建索引"：main.py 启动 load pkl 抛错只记 warning，进程照常起来，
            # 于是关键词这一路长期静默关闭，线上表现为纯向量召回而无人报警。
            return []
        # 默认每路最多 10 条，融合后保留 5 条，重排只能重排这 5 条，不能补回已截掉候选。
        hits = bm25_store.search(query, top_k=self.top_k * 2)
        return [
            RetrievalResult(
                # 与向量路同构，便于 _fusion 用 chunk_id 对齐；这里不区分查询是否命中关键词。
                chunk_id=doc.chunk_id,
                content=doc.content,
                # rank-bm25 返回 numpy.float64；必须转 float，否则进 LangGraph state 时
                # msgpack 序列化会失败（这个坑真实踩过）
                score=float(score),
                source_document=doc.source_document,
            )
            # score 已由 Bm25Store.search 除以本次最大正分，落在 (0,1]，只对本次查询有意义。
            for doc, score in hits
        ]

    async def _vector_search(self, query: str) -> list[RetrievalResult]:
        """Vector semantic search via Milvus."""
        # is_ready 可能调用 Milvus has_collection 并抛错，并不总是一个纯内存布尔检查。
        if not vector_store.is_ready:
            # 集合不存在（例如从没跑过入库）时同样返回空；两路都空则整个检索返回 []，不是异常。
            return []
        qv = embedder.embed_query(query)      # 本项目适配器是同步接口，故会阻塞当前事件循环；并非 embedding 协议只能同步
        # 查询向量维度与集合 dim 不一致时由 Milvus 抛错，这里不做本地预校验。
        hits = vector_store.search(qv, top_k=self.top_k * 2)
        return [
            RetrievalResult(
                # entity 缺字段一律回落空串：chunk_id 为空的多条候选会在融合字典里塌缩成同一个键。
                chunk_id=h["entity"].get("chunk_id", ""),
                content=h["entity"].get("content", ""),
                # COSINE: 1 = 完全一致；pymilvus 返回 numpy.float64，同样要转 float
                score=float(h["distance"]),
                source_document=h["entity"].get("source_document", ""),
            )
            for h in hits
            # 阈值过滤只在这里做：低于阈值的向量候选不参与融合，
            # 避免"语义上勉强相关"的噪声压过 BM25 的精确命中
            # 注意这也会消耗融合池：向量全被过滤时，结果只剩 BM25，权重 0.7 的路直接失去话语权。
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
        # 两张表都以 chunk_id 为键，必须同步写入：scores 有而 contents 没有的键会在下面取值时 KeyError。
        contents: dict[str, tuple[str, str]] = {}
        scores: dict[str, float] = {}

        for r in bm25:
            # 这里是覆盖赋值而不是累加：BM25 结果内同一 chunk_id 出现两次（旧快照 + 重复入库造的同名片）
            # 只会保留最后一条的分数，前面的命中被丢弃。
            scores[r.chunk_id] = self.bm25_weight * r.score
            contents[r.chunk_id] = (r.content, r.source_document)

        for r in vector:
            # 某 chunk 只在向量路命中时，缺失的 BM25 分按 0；无需虚构另一条候选。
            current = scores.get(r.chunk_id, 0.0)
            # 例如 BM25=0.8、COSINE=0.6，默认得 0.3×0.8+0.7×0.6=0.66；这不是 RRF。
            scores[r.chunk_id] = current + self.vector_weight * r.score
            # 只在 BM25 没收录该 chunk 时补内容，避免同一段文本被覆盖两次
            # 两段循环因此不能调换：向量在前、BM25 在后时，那个覆盖赋值会把已累加的 0.7 分量抹掉。
            if r.chunk_id not in contents:
                contents[r.chunk_id] = (r.content, r.source_document)

        # 先按融合分降序，再截断；分数相同依赖字典插入顺序，不是额外语义判断。
        # Timsort 稳定 + BM25 先入 dict，所以同分时"关键词命中"的那条排在前面。
        sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)[: self.top_k]
        return [
            RetrievalResult(
                chunk_id=cid,
                # 正文取自 contents：两路文本不一致时（pkl 旧、Milvus 新），展示的是 BM25 快照那份。
                content=contents[cid][0],
                score=round(scores[cid], 4),      # 保留 4 位，trace 里可读
                source_document=contents[cid][1],
            )
            for cid in sorted_ids
        ]
