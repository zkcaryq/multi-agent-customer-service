"""Document ingestion pipeline — split → embed → store in Milvus + BM25."""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.core.log import get_logger
from app.rag.splitter import auto_split, Chunk, extract_text
from app.rag.embedder import embedder
from app.rag.vector_store import vector_store
from app.rag.bm25_store import Bm25Doc, bm25_store

# Data directory relative to backend/
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# BM25 index persistence path
BM25_INDEX_PATH = str(DATA_DIR / ".index" / "bm25.pkl")

# Batch size for embedding API calls
EMBED_BATCH = 10  # DashScope limit: max 10 texts per request

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

        sync_url = settings.database_url.replace(
            "postgresql+asyncpg://", "postgresql+psycopg://", 1
        )
        sync_engine = create_engine(
            sync_url,
            pool_pre_ping=True,
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
    return "txt"


def _read_file_content(fp: Path) -> str:
    """Read file content, extracting text for binary formats."""
    ft = _file_type(fp)
    if ft in ("pdf", "docx", "xlsx"):
        return extract_text(str(fp), ft)
    return fp.read_text(encoding="utf-8")


def _source_name(path: Path, root: Path) -> str:
    """Derive a human-readable source document name from the file path."""
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
    all_chunks: list[tuple[Path, Chunk]] = []
    for fp in files:
        content = _read_file_content(fp)
        ft = _file_type(fp)
        chunks = auto_split(content, ft)
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
    vector_store.create_collection(drop_existing=rebuild)
    if not rebuild:
        for source in {row["source_document"] for row in milvus_rows}:
            vector_store.delete_by_source(source)
    insert_count = vector_store.insert(milvus_rows)
    vector_store.load()  # Ensure collection is loaded into memory

    # Step 5: Build and persist BM25 index
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
                session.flush()
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
        get_logger(__name__).warning("ingestion DB persist failed: %s", e)
        return None, False


def ingest_file(file_path: str) -> dict:
    """Ingest a single file — split → embed → store in Milvus + BM25.

    Returns stats dict. Also persists BM25 index after ingestion.
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
    vector_store.delete_by_source(src)
    insert_count = vector_store.insert(milvus_rows)
    vector_store.load()

    # Merge into existing BM25 index and rebuild
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
