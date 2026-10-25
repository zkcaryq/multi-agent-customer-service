"""文档入库管线：分切 → 向量化 → 写 Milvus + BM25 → 同步 PostgreSQL。

【三套存储，各司其职】
- Milvus：chunk 向量 + 文本（语义召回）
- BM25：内存索引 + pickle 快照（关键词召回）
- PostgreSQL：文档/切片的**事实源**（后台展示、引用来源、按文档统计）

【幂等的核心：按 source 替换】
重复上传同一份文件时，正确行为是"替换"而不是"追加"。
如果只在 PG 里去重，Milvus 和 BM25 仍会累积重复向量/文档，
检索结果里会出现同一段话的多个副本——所以三处都要按 source 先删后写。

【关于同步 vs 异步】
Web 应用用 asyncpg，但入库是离线批处理，整条链路是同步的。
若在同步函数里反复 `asyncio.run` 包 async 引擎，会把连接池绑到短命的事件循环上，
从第二个文件开始就报 "another operation in progress"（Windows 上必现）。
所以这里单独建一个同步引擎/会话工厂。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.core.log import get_logger
from app.rag.splitter import auto_split, Chunk, extract_text
from app.rag.embedder import embedder
from app.rag.vector_store import vector_store
from app.rag.bm25_store import Bm25Doc, bm25_store

# 知识库根目录：<仓库根>/data（ingestion.py 在 backend/app/rag/ 下，往上四层）
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# BM25 索引持久化路径。注意：这是进程级的单文件，多进程部署时各自加载
BM25_INDEX_PATH = str(DATA_DIR / ".index" / "bm25.pkl")

# 向量化批大小：DashScope 单次请求最多 10 条文本
EMBED_BATCH = 10  # DashScope limit: max 10 texts per request

# 同步会话工厂的模块级缓存（懒初始化）
_sync_session_factory = None


def _get_sync_session_factory():
    """Return a synchronous SQLAlchemy session factory for this synchronous pipeline.

    The web application uses asyncpg, but ingestion itself is synchronous. Repeatedly
    wrapping the global async engine in ``asyncio.run`` binds pooled connections to
    short-lived event loops and breaks from the second file onward on Windows.
    """
    global _sync_session_factory
    if _sync_session_factory is None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.config import settings

        # 同一个数据库，换驱动：asyncpg → psycopg（同步）
        sync_url = settings.database_url.replace(
            "postgresql+asyncpg://", "postgresql+psycopg://", 1
        )
        sync_engine = create_engine(
            sync_url,
            pool_pre_ping=True,
            # 时区同样锁 UTC（见第 01 章：库里都是 naive 时间）
            connect_args={"options": "-c timezone=UTC"},
        )
        _sync_session_factory = sessionmaker(
            bind=sync_engine,
            expire_on_commit=False,
        )
    return _sync_session_factory


def _collect_files(root: Path) -> list[Path]:
    """Walk data/ and collect all supported file types."""
    files: list[Path] = []
    # sorted 保证多次运行顺序一致（chunk 顺序影响 BM25 的 doc 顺序，便于复现）
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".md", ".faq", ".txt", ".pdf", ".docx", ".xlsx"):
            files.append(path)
    return files


def _file_type(path: Path) -> str:
    """Map file extension to splitter type."""
    ext = path.suffix.lower()
    for t in ("md", "faq", "pdf", "docx", "xlsx"):
        if ext == f".{t}":
            return t
    return "txt"          # 未知后缀按纯文本处理，而不是跳过（宁可切得粗，也不要漏文件）


def _read_file_content(fp: Path) -> str:
    """Read file content, extracting text for binary formats."""
    ft = _file_type(fp)
    if ft in ("pdf", "docx", "xlsx"):
        return extract_text(str(fp), ft)     # 二进制格式用专用解析器抽文本
    return fp.read_text(encoding="utf-8")    # md/faq/txt 直接读


def _source_name(path: Path, root: Path) -> str:
    """Derive a human-readable source document name from the file path.

    例：data/after_sales/退换货政策.md → "after_sales / 退换货政策"
    这个字符串同时是 Milvus 的 source_document、BM25 的分组键、删除时的过滤条件，
    所以必须稳定（不能带绝对路径，否则换机器就删不掉旧数据）。
    """
    rel = path.relative_to(root)
    return str(rel.with_suffix("")).replace(os.sep, " / ")


def ingest(rebuild: bool = False) -> dict[str, int]:
    """Run the full ingestion pipeline.

    Returns a dict with stats: {files, chunks, embedded, milvus_count, bm25_count}
    """
    t0 = time.monotonic()
    root = DATA_DIR
    files = _collect_files(root)

    if not files:
        return {"files": 0, "chunks": 0, "embedded": 0, "milvus_count": 0, "bm25_count": 0}

    # Step 1: Split all files into chunks
    # 先全部切完再统一 embedding：批次更大、API 调用更少，也便于统计总量
    all_chunks: list[tuple[Path, Chunk]] = []
    for fp in files:
        content = _read_file_content(fp)
        ft = _file_type(fp)
        chunks = auto_split(content, ft)      # 按文件类型选分切策略（第 04 章前半）
        for c in chunks:
            all_chunks.append((fp, c))

    # Step 2: Embed chunks in batches
    texts = [c.content for _, c in all_chunks]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        vectors.extend(embedder.embed(batch))

    # Step 3: Prepare Milvus rows and BM25 docs
    source_root = root
    milvus_rows: list[dict] = []
    bm25_docs: list[Bm25Doc] = []

    for idx, ((fp, chunk), vec) in enumerate(zip(all_chunks, vectors)):
        # chunk_id 的构成："来源#chunkN"。既是主键，也是"同一段文本"的身份，
        # 两路检索融合时用它去重。
        chunk_id = f"{_source_name(fp, source_root)}#chunk{chunk.chunk_index}"
        src = _source_name(fp, source_root)
        heading = chunk.metadata.get("heading", "")

        milvus_rows.append({
            "chunk_id": chunk_id,
            "content": chunk.content,
            "source_document": src,
            "heading": heading,
            "chunk_index": chunk.chunk_index,
            "embedding": vec,
        })

        bm25_docs.append(Bm25Doc(
            chunk_id=chunk_id,
            content=chunk.content,
            source_document=src,
            heading=heading,
            chunk_index=chunk.chunk_index,
        ))

    # Step 4: Store in Milvus
    # rebuild=True：整库重来（换 embedding 模型后必须这么做，否则新旧向量空间混在一起）
    vector_store.create_collection(drop_existing=rebuild)
    if not rebuild:
        # rebuild=False：按 source 先删后写，重复上传不会产生重复向量
        for source in {row["source_document"] for row in milvus_rows}:
            vector_store.delete_by_source(source)
    insert_count = vector_store.insert(milvus_rows)
    vector_store.load()  # Ensure collection is loaded into memory

    # Step 5: Build and persist BM25 index
    # 全量重建：BM25 是内存结构，没有"增量插入"的说法
    bm25_store.build(bm25_docs)
    bm25_store.save(BM25_INDEX_PATH)

    elapsed = int((time.monotonic() - t0) * 1000)
    return {
        "files": len(files),
        "chunks": len(all_chunks),
        "embedded": len(vectors),
        "milvus_count": insert_count,
        "bm25_count": len(bm25_store),
        "elapsed_ms": elapsed,
    }


def _persist_to_db(
    file_path: str, chunks: list[Chunk], file_type: str, split_method: str,
) -> tuple[int | None, bool]:
    """Create knowledge_document + knowledge_chunk records in PostgreSQL.

    Returns ``(document_id, created)``. A repeated path replaces its PostgreSQL
    chunks and increments the document version instead of silently leaving stale
    relational data while Milvus receives another copy.
    """
    from pathlib import Path as _P

    try:
        from app.models.database import KnowledgeDocument, KnowledgeChunk
        from sqlalchemy import delete, select

        # 用绝对路径做唯一键：同一份文件不管从哪个 cwd 触发入库，都归到同一条文档
        fp = _P(file_path).resolve()
        filename = fp.name
        file_size = fp.stat().st_size if fp.exists() else 0
        session_factory = _get_sync_session_factory()
        with session_factory() as session:
            existing = session.execute(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.file_path == str(fp)
                ).limit(1)
            ).scalar_one_or_none()
            if existing:
                # 更新：先删旧切片再写新的，版本号 +1，便于追溯"这份文档被改过几次"
                doc = existing
                session.execute(
                    delete(KnowledgeChunk).where(KnowledgeChunk.document_id == doc.id)
                )
                doc.title = filename
                doc.file_type = file_type
                doc.split_method = split_method
                doc.original_filename = filename
                doc.file_size_bytes = file_size
                doc.version = (doc.version or 1) + 1
                doc.status = "active"
                created = False
            else:
                doc = KnowledgeDocument(
                    title=filename,
                    file_type=file_type,
                    split_method=split_method,
                    file_path=str(fp),
                    original_filename=filename,
                    file_size_bytes=file_size,
                    status="active",
                )
                session.add(doc)
                session.flush()          # flush 拿到自增 id，供下面切片外键使用
                created = True

            session.add_all([
                KnowledgeChunk(
                    document_id=doc.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                )
                for chunk in chunks
            ])
            session.commit()
            return doc.id, created
    except Exception as e:
        # 落库失败不影响向量检索（工具类和 RAG 仍可用），但必须在日志里可见
        get_logger(__name__).warning("ingestion DB persist failed: %s", e)
        return None, False


def ingest_file(file_path: str) -> dict:
    """Ingest a single file — split → embed → store in Milvus + BM25.

    Returns stats dict. Also persists BM25 index after ingestion.
    后台"上传文档"走的就是这条路径（单文件、即时生效）。
    """
    t0 = time.monotonic()
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = _read_file_content(fp)
    ft = _file_type(fp)
    chunks = auto_split(content, ft)

    if not chunks:
        return {"files": 0, "chunks": 0, "embedded": 0, "milvus_count": 0, "bm25_count": 0}

    # Embed
    texts = [c.content for c in chunks]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        vectors.extend(embedder.embed(batch))

    # Use same root-relative naming convention
    root = DATA_DIR
    src = _source_name(fp, root)

    # Milvus rows
    milvus_rows: list[dict] = []
    bm25_docs: list[Bm25Doc] = []
    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        chunk_id = f"{src}#chunk{chunk.chunk_index}"
        heading = chunk.metadata.get("heading", "")
        milvus_rows.append({
            "chunk_id": chunk_id,
            "content": chunk.content,
            "source_document": src,
            "heading": heading,
            "chunk_index": chunk.chunk_index,
            "embedding": vec,
        })
        bm25_docs.append(Bm25Doc(
            chunk_id=chunk_id, content=chunk.content,
            source_document=src, heading=heading, chunk_index=chunk.chunk_index,
        ))

    # Replace-by-source keeps repeated uploads idempotent in Milvus and BM25.
    if not vector_store.is_ready:
        vector_store.create_collection()
    vector_store.delete_by_source(src)      # 同一文件重复上传：先删掉旧的
    insert_count = vector_store.insert(milvus_rows)
    vector_store.load()

    # Merge into existing BM25 index and rebuild
    # BM25 只支持整体重建：把"除本文档外"的旧切片保留，再拼上新切片
    existing = [
        doc for doc in (bm25_store._docs[:] if bm25_store.is_ready else [])
        if doc.source_document != src
    ]
    existing.extend(bm25_docs)
    bm25_store.build(existing)
    bm25_store.save(BM25_INDEX_PATH)

    # Persist to PostgreSQL
    split_method = {"md": "structure", "faq": "qa_boundary", "txt": "fixed_size"}.get(ft, "fixed_size")
    doc_id, _created = _persist_to_db(str(fp), chunks, ft, split_method)

    return {
        "files": 1,
        "chunks": len(chunks),
        "embedded": len(vectors),
        "milvus_count": insert_count,
        "bm25_count": len(bm25_store),
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
        "source": src,
        "document_id": doc_id,
    }


def sync_db_all() -> dict:
    """Sync all existing data/ files to PostgreSQL without re-embedding.

    Reads each file, splits it (no embedding needed), and creates
    knowledge_documents + knowledge_chunks records if they don't exist.
    用于"向量库已经有数据、PG 却是空的"这种修复场景。
    """
    files = _collect_files(DATA_DIR)
    if not files:
        return {"files": 0, "documents": 0, "chunks": 0}

    docs_synced = 0
    docs_created = 0
    docs_existing = 0
    chunks_created = 0
    failed = 0
    for fp in files:
        content = _read_file_content(fp)
        ft = _file_type(fp)
        split_method = {"md": "structure", "faq": "qa_boundary", "txt": "fixed_size"}.get(ft, "fixed_size")
        chunks = auto_split(content, ft)
        doc_id, created = _persist_to_db(str(fp), chunks, ft, split_method)
        if doc_id:
            docs_synced += 1
            if created:
                docs_created += 1
                chunks_created += len(chunks)
            else:
                docs_existing += 1
        else:
            failed += 1

    return {
        "files": len(files),
        "documents": docs_synced,
        "created": docs_created,
        "existing": docs_existing,
        "chunks": chunks_created,
        "failed": failed,
    }
