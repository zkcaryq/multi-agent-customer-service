"""检索链路使用的向量化入口。

真正的供应商由 EMBEDDING_PROVIDER 决定（见 app/providers/registry.py），
这里只做"单条/多条"的形状适配，业务侧不再出现任何 SDK 与模型名。
"""

from __future__ import annotations

from app.providers.registry import get_embedding


class Embedder:
    def embed(self, texts: str | list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return get_embedding().embed(texts)

    def embed_query(self, query: str) -> list[float]:
        return get_embedding().embed_query(query)

    @property
    def dim(self) -> int:
        return get_embedding().dim


# Singleton
embedder = Embedder()
