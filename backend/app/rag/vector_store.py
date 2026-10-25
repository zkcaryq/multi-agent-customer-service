"""Milvus 向量库。三种形态由配置决定，代码不区分：

- MILVUS_ENDPOINT 为空       → Milvus Lite，本地文件（默认 backend/data/milvus.db）
- MILVUS_LITE_PATH            → 显式指定 Lite 的数据文件位置（挂 eMMC/SSD，别放 TF 卡）
- MILVUS_ENDPOINT=http(s)://… → Milvus Server 或 Zilliz Cloud，配合 MILVUS_TOKEN / MILVUS_DB_NAME
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pymilvus import (
    DataType,
    FieldSchema,
    CollectionSchema,
    MilvusClient,
)

from app.config import settings

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_LITE_FILE = (_BACKEND_DIR / "data" / "milvus.db").as_posix()

COLLECTION = settings.milvus_collection


def _resolve_target() -> tuple[str, str, str]:
    """返回 (uri, token, db_name)。uri 是 http(s):// 即远端，否则是本地文件。"""
    endpoint = settings.milvus_endpoint.strip()
    if endpoint:
        return endpoint, settings.milvus_token, settings.milvus_db_name

    # pymilvus 在 Windows 上只接受正斜杠的本地文件路径，反斜杠会被当成非法 uri
    path = Path(settings.milvus_lite_path or DEFAULT_LITE_FILE)
    if not path.is_absolute():
        path = _BACKEND_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.as_posix(), "", ""


class VectorStore:
    """Wraps Milvus for chunk storage and semantic search."""

    def __init__(self) -> None:
        self.uri, self.token, self.db_name = _resolve_target()
        self.is_remote = self.uri.startswith(("http://", "https://"))
        self._client: MilvusClient | None = None
        self._collection_ready = False

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            kwargs: dict[str, Any] = {"uri": self.uri}
            if self.token:
                kwargs["token"] = self.token
            if self.db_name:
                kwargs["db_name"] = self.db_name
            self._client = MilvusClient(**kwargs)
        return self._client

    def create_collection(self, drop_existing: bool = False) -> None:
        """Create the knowledge_chunks collection with schema."""
        if drop_existing and self.client.has_collection(COLLECTION):
            self.client.drop_collection(COLLECTION)

        if self.client.has_collection(COLLECTION):
            self._collection_ready = True
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="source_document", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(
                    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim
                ),
            ],
            description="CogniFlow knowledge base chunks",
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=COLLECTION,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(COLLECTION)
        self._collection_ready = True

    def insert(self, rows: list[dict[str, Any]]) -> int:
        """Insert chunk records. Each row: chunk_id, content, source_document, heading, chunk_index, embedding."""
        if not rows:
            return 0
        result = self.client.insert(collection_name=COLLECTION, data=rows)
        return result["insert_count"]

    def delete_by_source(self, source_document: str) -> None:
        """Delete every old chunk for one logical source before replacement."""
        if not self.client.has_collection(COLLECTION):
            return
        escaped = source_document.replace("\\", "\\\\").replace('"', '\\"')
        self.client.delete(
            collection_name=COLLECTION,
            filter=f'source_document == "{escaped}"',
        )

    def load(self) -> None:
        """Load collection into memory (required before search)."""
        if self.client.has_collection(COLLECTION):
            self.client.load_collection(COLLECTION)

    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Semantic search — returns top_k matched chunks with metadata."""
        self.load()  # ensure collection is in memory (idempotent)
        results = self.client.search(
            collection_name=COLLECTION,
            data=[query_vector],
            limit=top_k,
            output_fields=["chunk_id", "content", "source_document", "heading"],
        )
        return results[0] if results else []

    @property
    def is_ready(self) -> bool:
        return self._collection_ready or self.client.has_collection(COLLECTION)


# Singleton
vector_store = VectorStore()
