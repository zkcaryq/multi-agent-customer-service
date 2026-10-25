"""Embedding 供应商实现。

`openai_compat` 覆盖百炼 text-embedding-v3/v4 及任意 OpenAI 兼容 embedding 网关；
`hash` 是确定性的非语义占位实现，只用于离线跑通链路和单测，不能上生产。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from app.config import settings
from app.providers.base import ProviderError

PROVIDER_NAME_OPENAI = "openai_compat"
PROVIDER_NAME_HASH = "hash"

# 百炼 embedding 接口单次请求的文本条数上限，超出必须分批
BATCH_SIZE = 10


class OpenAICompatEmbedding:
    def __init__(self) -> None:
        self.dim = settings.embedding_dim
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            api_key = settings.key_for("embedding")
            if not api_key:
                raise ProviderError("未配置 EMBEDDING_API_KEY / CHAT_API_KEY")
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key,
                base_url=settings.base_url_for("embedding"),
                timeout=30.0,
            )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[i : i + BATCH_SIZE]))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = self.client.embeddings.create(
                model=settings.embedding_model, input=batch, dimensions=self.dim
            )
        except TypeError:
            # 部分兼容网关不认 dimensions 参数，退回默认维度再试一次
            resp = self.client.embeddings.create(model=settings.embedding_model, input=batch)
        except Exception as exc:
            raise ProviderError(
                f"embedding 调用失败 model={settings.embedding_model}: {exc}"
            ) from exc

        vectors = [d.embedding for d in resp.data]
        if vectors and len(vectors[0]) != self.dim:
            raise ProviderError(
                f"向量维度不匹配：期望 {self.dim}，实际 {len(vectors[0])}。"
                f"改 EMBEDDING_DIM 需重建集合（维度写进 Milvus schema）。"
            )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return (self.embed([query]))[0]


class HashEmbedding:
    """确定性伪向量：同文本恒定同向量，用于离线联调与测试。不具备语义。"""

    def __init__(self) -> None:
        self.dim = settings.embedding_dim

    def _one(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._one(query)
