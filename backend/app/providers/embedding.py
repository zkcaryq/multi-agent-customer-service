"""Embedding 供应商实现。

`openai_compat` 覆盖百炼 text-embedding-v3/v4 及任意 OpenAI 兼容 embedding 网关；
`hash` 是确定性的非语义占位实现，只用于离线跑通链路和单测，不能上生产。
"""

from __future__ import annotations

import hashlib  # 只用于 hash 占位实现的确定性种子，不是任何安全用途
import random  # 只用来 new 局部 Random，绝不碰全局随机状态
from typing import Any

from app.config import settings
from app.providers.base import ProviderError

PROVIDER_NAME_OPENAI = "openai_compat"  # 与 registry._EMBEDDING 的键、EMBEDDING_PROVIDER 默认值一致
PROVIDER_NAME_HASH = "hash"  # 文档性常量：路由实际看配置字符串，这两个名字在本文件里没有读取方

# 百炼 embedding 接口单次请求的文本条数上限，超出必须分批
BATCH_SIZE = 10  # 决定入库请求数（切片数 / 10），与维度无关；没有配置项，换供应商必须回来改这里


class OpenAICompatEmbedding:
    def __init__(self) -> None:
        # 向量维度必须与 Milvus 集合 schema 一致；只改配置不能把旧向量转换成新空间。
        self.dim = settings.embedding_dim  # EMBEDDING_DIM 默认 1024，同时是请求参数和返回值校验值
        # 延迟构造同步 SDK 客户端；一次适配器实例内可复用该客户端。
        self._client: Any = None  # Any 是为了不把 openai 类型带进签名；实例由 registry 缓存，所以连接池进程内复用

    @property
    def client(self) -> Any:
        if self._client is None:
            api_key = settings.key_for("embedding")  # EMBEDDING_API_KEY 优先，没单独配就回落到 chat 的凭据（百炼同一把 key 通用）
            if not api_key:
                raise ProviderError("未配置 EMBEDDING_API_KEY / CHAT_API_KEY")  # preflight 只在 provider=openai_compat 时提前查，这里是最后一道
            from openai import OpenAI  # 同步版：协议里 embed 是 def 不是 async def，用 AsyncOpenAI 会要求上层改签名

            self._client = OpenAI(
                api_key=api_key,
                # base_url_for("embedding") 空则回落 CHAT_BASE_URL；这就是同一家网关时只需配一个 URL 的原因
                base_url=settings.base_url_for("embedding"),
                timeout=30.0,  # 硬编码，既不读 CHAT_TIMEOUT_S 也没有 EMBEDDING_TIMEOUT_S；一批卡住最多等 30 秒，批次串行所以总耗时按批数乘
            )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # 按本项目设置的每批十条分请求，避免一次塞入全部知识；换供应商应复核其上限。
        for i in range(0, len(texts), BATCH_SIZE):
            # 按批次顺序拼向量；检索和文档入库必须使用匹配的 embedding 模型。
            # 串行请求、无并发也无重试：任一批失败会整轮 raise，前面批次的结果一并丢弃。
            vectors.extend(self._embed_batch(texts[i : i + BATCH_SIZE]))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = self.client.embeddings.create(
                model=settings.embedding_model, input=batch, dimensions=self.dim
            )
        # 只捕获 Python TypeError；远端 HTTP 不支持 dimensions 返回的异常不一定属于该类型。
        except TypeError:
            # 部分兼容网关不认 dimensions 参数，退回默认维度再试一次
            resp = self.client.embeddings.create(model=settings.embedding_model, input=batch)  # 网关返回默认维度（如 1536）时，紧接着的维度校验就会抛错，这条降级只在默认值恰好等于 self.dim 时才真能救回来
        except Exception as exc:
            raise ProviderError(
                f"embedding 调用失败 model={settings.embedding_model}: {exc}"
            ) from exc  # 网络/HTTP/SDK 异常统一成 ProviderError；调用方（检索链路）不捕获它，会直接冒到接口层

        # 默认供应商按输入顺序返回；当前没有用响应 index 重排，也未校验总条数。
        vectors = [d.embedding for d in resp.data]  # 少了或多了一条都不会报错，错位会把向量和 chunk 配错，这比报错更难发现
        # 这里只检查首向量维度；空返回或后续向量长度异常尚未完整覆盖。
        if vectors and len(vectors[0]) != self.dim:
            raise ProviderError(
                f"向量维度不匹配：期望 {self.dim}，实际 {len(vectors[0])}。"
                f"改 EMBEDDING_DIM 需重建集合（维度写进 Milvus schema）。"
            )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        # 单条 query 复用批量协议，再取首向量；空响应会抛 IndexError。
        return (self.embed([query]))[0]  # 走完整 batch 逻辑（含 dimensions 降级与维度校验），查询与文档因此不会用到不同维度


class HashEmbedding:
    """确定性伪向量：同文本恒定同向量，用于离线联调与测试。不具备语义。"""

    def __init__(self) -> None:
        # 向量维度必须与 Milvus 集合 schema 一致；只改配置不能把旧向量转换成新空间。
        self.dim = settings.embedding_dim  # 与真实实现读同一个配置，所以能塞进同一个集合，仅维度对得上、语义对不上

    def _one(self, text: str) -> list[float]:
        # 文本哈希决定伪随机种子，使同文本得到同向量；不表示近义文本会接近。
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")  # 不用内置 hash()：它受 PYTHONHASHSEED 影响，跨进程/跨次运行就不确定了
        # 局部随机数生成器不污染全局 random 状态，便于确定性测试。
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]  # 逐维取值，dim=1024 时每条 1024 次调用，纯 CPU 但比真实网络请求便宜
        # L2 范数；除以它得到单位向量，or 1 防止除零。
        norm = sum(v * v for v in vec) ** 0.5 or 1.0  # 归一化后内积即余弦，和 Milvus COSINE 度量的行为一致
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]  # 本地计算，忽略 BATCH_SIZE（无网络请求也就无需分批）；顺序保持与入参一致

    def embed_query(self, query: str) -> list[float]:
        return self._one(query)  # 与 embed 同一套生成规则，所以自检时"查自己"相似度接近 1，容易给出假高分
