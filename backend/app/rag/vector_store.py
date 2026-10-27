"""Milvus 向量库。三种形态由配置决定，代码不区分：

- MILVUS_ENDPOINT 为空       → Milvus Lite，本地文件（默认 backend/data/milvus.db）
- MILVUS_LITE_PATH            → 显式指定 Lite 的数据文件位置（挂 eMMC/SSD，别放 TF 卡）
- MILVUS_ENDPOINT=http(s)://… → Milvus Server 或 Zilliz Cloud，配合 MILVUS_TOKEN / MILVUS_DB_NAME
"""

from __future__ import annotations

import os  # 本文件实际没有用到 os（路径全走 pathlib），属于遗留 import，不影响行为
from pathlib import Path
from typing import Any

# pymilvus 的 schema 描述接口：本项目只建集合，不用 ORM 模型，字段全靠 FieldSchema 手写。
from pymilvus import (
    DataType,
    FieldSchema,
    CollectionSchema,
    MilvusClient,
)

from app.config import settings

# 上溯三层得到 backend/ 目录：Lite 默认文件与相对路径都以它为基准，而不是进程 cwd。
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
# as_posix() 强制正斜杠：pymilvus 在 Windows 上把反斜杠本地路径当成非法 uri，这里在常量层就规避掉。
DEFAULT_LITE_FILE = (_BACKEND_DIR / "data" / "milvus.db").as_posix()

# 集合名在 import 时就定格：改 MILVUS_COLLECTION 需要重启进程才生效，运行期改 settings 不影响这里。
COLLECTION = settings.milvus_collection


def _resolve_target() -> tuple[str, str, str]:
    """返回 (uri, token, db_name)。uri 是 http(s):// 即远端，否则是本地文件。"""
    # 非空端点优先，空才落本地文件；不要把端点或 token 写进日志与提交物。
    endpoint = settings.milvus_endpoint.strip()
    if endpoint:
        # 远端形态才会带 token / db_name；token 是凭据字符串，值不进日志，这里只是原样透传给 SDK。
        return endpoint, settings.milvus_token, settings.milvus_db_name

    # pymilvus 在 Windows 上只接受正斜杠的本地文件路径，反斜杠会被当成非法 uri
    path = Path(settings.milvus_lite_path or DEFAULT_LITE_FILE)
    # 相对路径按 backend/ 解析，不按当前工作目录：从仓库根或从 backend/ 启动都落到同一个文件。
    if not path.is_absolute():
        path = _BACKEND_DIR / path
    # 构造单例时就建父目录（exist_ok），否则 MilvusClient 打开不存在的 .db 文件会失败。
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lite 形态固定返回空 token/空库名，避免把远端凭据误带进本地连接。
    return path.as_posix(), "", ""


class VectorStore:
    """Wraps Milvus for chunk storage and semantic search."""

    def __init__(self) -> None:
        # 导入模块底部就实例化单例，所以这一步在建目录、还没在建连接。
        self.uri, self.token, self.db_name = _resolve_target()
        # 只按前缀判断形态：其他 scheme（如自定义 file:// 之外的写法）会被当成本地文件处理。
        self.is_remote = self.uri.startswith(("http://", "https://"))
        self._client: MilvusClient | None = None
        # 只是进程级就绪缓存；远端集合被删除后该缓存不一定立即反映真实状态。
        self._collection_ready = False

    @property
    def client(self) -> MilvusClient:
        # 懒连接：第一次访问属性才真正创建客户端；没有锁，两个协程并发首访可能各建一个实例（后者覆盖前者）。
        if self._client is None:
            kwargs: dict[str, Any] = {"uri": self.uri}
            # 空 token / 空库名一律不传，交给 SDK 用默认值，避免远端配置缺项时报错。
            if self.token:
                kwargs["token"] = self.token
            if self.db_name:
                kwargs["db_name"] = self.db_name
            # 首次 client 访问才建连接；__init__ 本身不代表 Milvus 已连通。
            self._client = MilvusClient(**kwargs)
        return self._client

    def create_collection(self, drop_existing: bool = False) -> None:
        """Create the knowledge_chunks collection with schema."""
        # 这是破坏旧集合的入口；普通检索不应走 drop_existing。
        if drop_existing and self.client.has_collection(COLLECTION):
            self.client.drop_collection(COLLECTION)

        # 已存在就复用，不自动核验维度/模型兼容；换 embedding 模型需要重新入库。
        if self.client.has_collection(COLLECTION):
            self._collection_ready = True
            return

        schema = CollectionSchema(
            fields=[
                # 物理主键由 Milvus 自动生成；chunk_id 另作逻辑去重键，所以重复 insert 本身不防重复。
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                # 长度上限按字符算：来源名很长的嵌套目录 + "#chunkN" 可能撞 128；超长时 insert 直接报错。
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
                # 8192 是 Milvus 侧的 VARCHAR 上限（UTF-8 按字节计），中文切片实际约 2700 字就会写失败，
                # 而 SemanticSplitter 的 800 只是软上限、超长段落不会二次切分。
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="source_document", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(
                    # dim 来自 EMBEDDING_DIM，被固化进 schema：改维度只能重建集合，旧集合不接受新向量。
                    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim
                ),
            ],
            description="customer-service knowledge chunks",
        )

        # AUTOINDEX：Milvus Lite 实际退化成暴力扫描（FLAT），集合变大后 search 线性变慢；server 端才自动选索引类型。
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            # COSINE 的分数越大越相关；pymilvus 字段名虽叫 distance，也不能按越小越好来过滤。
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=COLLECTION,
            schema=schema,
            index_params=index_params,
        )
        # 建完立刻 load：Milvus 不 load 就搜不到；Lite 上这一步开销可忽略，远端会占内存。
        self.client.load_collection(COLLECTION)
        self._collection_ready = True

    def insert(self, rows: list[dict[str, Any]]) -> int:
        """Insert chunk records. Each row: chunk_id, content, source_document, heading, chunk_index, embedding."""
        # 空列表直接返回 0：不会触发网络请求，也就不会暴露"集合没建"的问题。
        if not rows:
            return 0
        # 同步网络调用；当前代码没有检查每条向量维度或跨存储提交的一致性。
        # 整个列表一次提交，不再分批：切片很多时是单次大请求；返回体里的部分失败项也没有逐条检查。
        result = self.client.insert(collection_name=COLLECTION, data=rows)
        return result["insert_count"]

    def delete_by_source(self, source_document: str) -> None:
        """Delete every old chunk for one logical source before replacement."""
        # 先 has_collection（一次额外请求）：集合不存在时静默跳过，不抛错——调用方以为"旧数据已清"。
        if not self.client.has_collection(COLLECTION):
            return
        # 只转义反斜杠和双引号，因为下面拼的是 source_document == "…" 这种带双引号的表达式。
        escaped = source_document.replace("\\", "\\\\").replace('"', '\\"')
        self.client.delete(
            collection_name=COLLECTION,
            # 按标量列过滤删除（不是按主键）：Milvus 走全表扫描 + 软删除标记，删除不回收空间也未强制 flush，
            # 紧接其后的 insert 是否立刻对搜索可见取决于集合一致性级别；这里没有校验被删条数。
            filter=f'source_document == "{escaped}"',
        )

    def load(self) -> None:
        """Load collection into memory (required before search)."""
        # 已存在就复用，不自动核验维度/模型兼容；换 embedding 模型需要重新入库。
        if self.client.has_collection(COLLECTION):
            # 每次调用都是 has_collection + load_collection 两个 RPC；集合已加载时 load 是幂等的但仍有往返。
            self.client.load_collection(COLLECTION)

    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Semantic search — returns top_k matched chunks with metadata."""
        # 每次查询都会先检查/加载集合，可能多一次远端请求；并非纯本地操作。
        self.load()  # ensure collection is in memory (idempotent)
        results = self.client.search(
            collection_name=COLLECTION,
            # Milvus 批量查询接口，这里只有一个 query；返回 results[0] 对应它的命中列表。
            data=[query_vector],
            # limit 由调用方给（top_k*2）；没有 filter 表达式，所以无法按来源或状态位圈定检索范围。
            limit=top_k,
            # 不返回 embedding，省带宽；hit["entity"] 只含这四列 + id/distance。
            output_fields=["chunk_id", "content", "source_document", "heading"],
        )
        # 空结果返回 []，上层据此走"只融合 BM25"；异常仍向上抛，由 retriever 冒泡到 Agent 层。
        return results[0] if results else []

    @property
    def is_ready(self) -> bool:
        # 短路左侧为真时不访问远端；左侧为假时可能网络报错，调用方需要处理异常。
        return self._collection_ready or self.client.has_collection(COLLECTION)


# Singleton
# 模块级单例：多 worker 进程各持一份客户端连接；Milvus Lite 的本地文件被多进程同时写并不安全。
vector_store = VectorStore()
