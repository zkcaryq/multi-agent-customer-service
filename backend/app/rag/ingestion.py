"""文档入库管线：分切 → 向量化 → 写 Milvus + BM25 → 同步 PostgreSQL。

【三套存储，各司其职】
- Milvus：chunk 向量 + 文本（语义召回）
- BM25：内存索引 + pickle 快照（关键词召回）
- PostgreSQL：文档/切片的**事实源**（后台展示、引用来源、按文档统计）

【幂等的核心：按 source 替换】
重复上传同一份文件时，正确行为是"替换"而不是"追加"。
如果只在 PG 里去重，Milvus 和 BM25 仍会累积重复向量/文档，
检索结果里会出现同一段话的多个副本——所以单文件入库对检索按 source 替换，对 PG 按文件绝对路径替换；这不是跨存储原子事务。

【关于同步 vs 异步】
Web 应用用 asyncpg，但入库是离线批处理，整条链路是同步的。
若在同步函数里反复 `asyncio.run` 包 async 引擎，会把连接池绑到短命的事件循环上，
从第二个文件开始就报 "another operation in progress"（Windows 上必现）。
所以这里单独建一个同步引擎/会话工厂。
"""

from __future__ import annotations

import os  # 只用 os.sep：把平台路径分隔符统一换成 " / "，保证 source 键跨机器写法一致
import time  # time.monotonic 计时，不受系统时钟回拨影响；只用于 elapsed 统计，不参与任何判定
from pathlib import Path

from app.core.log import get_logger  # 项目封装的 logger（带 trace 字段），DB 降级只走这里
from app.rag.splitter import auto_split, Chunk, extract_text  # 分切路由 + 二进制文本抽取；Chunk 只用于类型标注
from app.rag.embedder import embedder  # 供应商无关的向量化入口（模块级单例，同步调用）
from app.rag.vector_store import vector_store  # Milvus 单例：建集合/按 source 删/插/load
from app.rag.bm25_store import Bm25Doc, bm25_store  # BM25 行结构 + 进程内索引单例

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
    # 复用同步 SQLAlchemy 连接池，不为每个文件创建新的事件循环和异步池。
    global _sync_session_factory
    # 只缓存工厂不缓存 session；进程内共享，多线程跑入库不会各自建引擎。
    if _sync_session_factory is None:
        # 依赖放在函数内：没装 SQLAlchemy 或 PG 不可用时，只影响 PG 落库，不拖累 Milvus/BM25 入库。
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # settings 也延迟取：本模块被 scripts 与 admin 上传接口共用，导入时不该触发 .env 解析。
        from app.config import settings

        # 同一个数据库，换驱动：asyncpg → psycopg（同步）
        # 只替换第一处前缀（count=1）；URL 其他位置出现同串不会被误改。
        sync_url = settings.database_url.replace(
            "postgresql+asyncpg://", "postgresql+psycopg://", 1
        )
        sync_engine = create_engine(
            sync_url,
            # 借用连接前探活，减少拿到失效连接；不提供跨 Milvus/BM25/PG 回滚。
            pool_pre_ping=True,
            # 时区同样锁 UTC（见第 01 章：库里都是 naive 时间）
            connect_args={"options": "-c timezone=UTC"},
        )
        # 连接池大小等参数用 SQLAlchemy 默认值；批量入库靠循环复用同一池。
        _sync_session_factory = sessionmaker(
            bind=sync_engine,
            # 提交后仍能读取 doc.id 等已加载属性，不触发额外加载。
            expire_on_commit=False,
        )
    return _sync_session_factory


def _collect_files(root: Path) -> list[Path]:
    """Walk data/ and collect all supported file types."""
    files: list[Path] = []
    # sorted 保证多次运行顺序一致（chunk 顺序影响 BM25 的 doc 顺序，便于复现）
    # rglob 递归全部子目录，也包含 .index/ 等隐藏目录：pkl 不在白名单里，靠后缀被过滤掉。
    for path in sorted(root.rglob("*")):
        # 后缀白名单决定后续 auto_split 的 file_type；不在名单内的文件静默忽略，不报错也不计入统计。
        if path.is_file() and path.suffix.lower() in (".md", ".faq", ".txt", ".pdf", ".docx", ".xlsx"):
            files.append(path)
    return files


def _file_type(path: Path) -> str:
    """Map file extension to splitter type."""
    ext = path.suffix.lower()
    # 五个已知后缀按元组顺序比较，命中即返回；大写后缀靠上面的 lower() 归一。
    for t in ("md", "faq", "pdf", "docx", "xlsx"):
        if ext == f".{t}":
            return t
    return "txt"          # 未知后缀按纯文本处理，而不是跳过（宁可切得粗，也不要漏文件）


def _read_file_content(fp: Path) -> str:
    """Read file content, extracting text for binary formats."""
    ft = _file_type(fp)
    # 三种二进制格式走各自解析库；解析失败会向上抛，整次入库中断（没有单文件级容错）。
    if ft in ("pdf", "docx", "xlsx"):
        return extract_text(str(fp), ft)     # 二进制格式用专用解析器抽文本
    # md/faq/txt 直接按 utf-8 读；非 utf-8 文件（如 gbk）在这里抛 UnicodeDecodeError。
    return fp.read_text(encoding="utf-8")    # md/faq/txt 直接读


def _source_name(path: Path, root: Path) -> str:
    """Derive a human-readable source document name from the file path.

    例：data/after_sales/退换货政策.md → "after_sales / 退换货政策"
    这个字符串同时是 Milvus 的 source_document、BM25 的分组键、删除时的过滤条件，
    所以必须稳定（不能带绝对路径，否则换机器就删不掉旧数据）。
    """
    # 文件不在 root 之下时 relative_to 直接抛 ValueError；上传接口若存到 data/ 外面就会走到这条。
    rel = path.relative_to(root)
    # 去掉扩展名后 md/faq 同名路径可能共用来源键；路径重命名也会改变 chunk 身份。
    return str(rel.with_suffix("")).replace(os.sep, " / ")


def ingest(rebuild: bool = False) -> dict[str, int]:
    """Run the full ingestion pipeline.

    返回切片、向量和 BM25 统计。本函数本身不写 PG；命令脚本会再调用 sync_db_all。
    """
    # 计时起点：monotonic 只用来算耗时，不参与任何业务判定；进程重启不影响取值意义。
    t0 = time.monotonic()
    # 全量入口的扫描根硬编码为 DATA_DIR，不接受外部目录参数，也无法只重跑某个子目录。
    root = DATA_DIR
    files = _collect_files(root)

    # 空库直接返回：不建集合、不动 BM25，磁盘上的旧 pkl 快照因此不会被清空。
    if not files:
        return {"files": 0, "chunks": 0, "embedded": 0, "milvus_count": 0, "bm25_count": 0}

    # Step 1: Split all files into chunks
    # 先全部切完再统一 embedding：批次更大、API 调用更少，也便于统计总量
    # 元素是 (文件路径, Chunk)：Chunk 自身不记来源，source 只能靠这个元组在第 3 步回推。
    all_chunks: list[tuple[Path, Chunk]] = []
    for fp in files:
        # 整份文件文本一次性进内存；全库扫描时峰值约等于所有文件文本之和，没有流式处理。
        content = _read_file_content(fp)
        ft = _file_type(fp)
        # 按文件类型选分切策略（第 04 章前半）
        # 这里不传 file_path：auto_split 只按扩展名路由，该形参在其函数体内从未被使用。
        chunks = auto_split(content, ft)
        # 平铺成一个大列表：跨文件的顺序靠 (fp, chunk.chunk_index) 表达，不是全局连续序号。
        for c in chunks:
            all_chunks.append((fp, c))

    # Step 2: Embed chunks in batches
    texts = [c.content for _, c in all_chunks]
    vectors: list[list[float]] = []
    # 每批 10 条（DashScope 单次上限）。embedder 内部还会再按同样的 10 条分批，两层批次重叠但值一致。
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        # 按文本顺序拼接批次向量；后面 zip 假设数量与顺序吻合，当前未严格核验全部向量数量。
        # 任一批次请求失败会直接向上传抛：已算好的向量留在局部变量里随之丢弃，Milvus 还没被写过。
        vectors.extend(embedder.embed(batch))

    # Step 3: Prepare Milvus rows and BM25 docs
    # 与 root 同值的别名，只为了让下面的调用读起来像"相对某个根目录"。
    source_root = root
    milvus_rows: list[dict] = []
    bm25_docs: list[Bm25Doc] = []

    # zip 取较短序列长度；供应商少返回向量时可能漏入库，不能把统计中的 chunks 当成功入库数。
    # idx 在本循环体内并未使用（chunk_id 用的是 chunk.chunk_index），只是 enumerate 的写法残留。
    for idx, ((fp, chunk), vec) in enumerate(zip(all_chunks, vectors)):
        # chunk_id 的构成："来源#chunkN"。既是主键，也是"同一段文本"的身份，
        # 两路检索融合时用它去重。
        chunk_id = f"{_source_name(fp, source_root)}#chunk{chunk.chunk_index}"
        # 同一个 source 名每片重算一次（relative_to + with_suffix）：结果稳定但有重复计算开销。
        src = _source_name(fp, source_root)
        # 只有 MarkdownSplitter/_extract_docx 路线会写 heading；FAQ/定长/Excel 片取到空串。
        heading = chunk.metadata.get("heading", "")

        # 与 BM25 行字段同名同值，但这里是 dict、那边是 dataclass 实例，两份存储各自独立。
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
    # rebuild=True 会删除旧集合；仅在明确要重建且有数据备份时运行，不是普通只读验证。
    vector_store.create_collection(drop_existing=rebuild)
    if not rebuild:
        # rebuild=False：按 source 先删后写，重复上传不会产生重复向量
        # 集合推导去重：一个文件只发一次 delete 请求；来源集合来自本次扫描，磁盘上已删掉的文件不会被清理。
        for source in {row["source_document"] for row in milvus_rows}:
            # 按当前存在文件的 source 删除；磁盘上已删除的文件来源不会在这一步自动清理。
            vector_store.delete_by_source(source)
    # 整个列表一次 insert：不在这里再切批，Milvus 客户端内部处理；返回的 insert_count 是成功写入数。
    insert_count = vector_store.insert(milvus_rows)
    vector_store.load()  # Ensure collection is loaded into memory

    # Step 5: Build and persist BM25 index
    # 全量重建：BM25 是内存结构，没有"增量插入"的说法
    # 全量 ingest 的 BM25 只用本次扫描到的文档；与未清理旧来源的向量库可能出现差异。
    bm25_store.build(bm25_docs)
    # 覆盖写 pkl：写完这份快照之前进程被杀，重启后 load 到的还是上一轮的旧索引。
    bm25_store.save(BM25_INDEX_PATH)

    elapsed = int((time.monotonic() - t0) * 1000)
    # 注意三个计数的口径差异：chunks=切出的片数，embedded=拿到的向量数，milvus_count=Milvus 回执写入数。
    # PG 不在这个函数里写，所以这里没有任何"PG 已同步"的信号。
    return {
        "files": len(files),                     # 命中后缀白名单的文件数，不含被忽略的其他文件
        "chunks": len(all_chunks),               # 切片总数，不代表入库成功数（zip 可能截短）
        "embedded": len(vectors),                # 实际拿到的向量数
        "milvus_count": insert_count,            # Milvus 回执的写入条数
        "bm25_count": len(bm25_store),           # 当前进程内存里的 BM25 文档数（等于本次全量结果）
        "elapsed_ms": elapsed,                   # 含 embedding 网络等待，不能当作纯算力耗时来对比
    }


def _persist_to_db(
    file_path: str, chunks: list[Chunk], file_type: str, split_method: str,
) -> tuple[int | None, bool]:
    """Create knowledge_document + knowledge_chunk records in PostgreSQL.

    Returns ``(document_id, created)``. A repeated path replaces its PostgreSQL
    chunks and increments the document version instead of silently leaving stale
    relational data while Milvus receives another copy.
    """
    # 局部别名：本模块的 Path 已用于 DATA_DIR/ingest_file，这里只为拿到 resolve()。
    from pathlib import Path as _P

    try:
        # ORM 模型与 SQL 构造器都在函数内导入：PG 依赖或表结构缺失时只影响降级分支，不会拖垮向量入库。
        from app.models.database import KnowledgeDocument, KnowledgeChunk
        from sqlalchemy import delete, select

        # 用绝对路径做唯一键：同一份文件不管从哪个 cwd 触发入库，都归到同一条文档
        # PG 文档身份是机器上的绝对路径；跨机器移动文件不等同于同一文档版本。
        # 这是第三套幂等键：向量/BM25 用 data/ 相对名，PG 用 resolve() 后的绝对路径，两边不能互相定位同一条记录。
        fp = _P(file_path).resolve()
        filename = fp.name
        # 文件已不存在时大小记 0（不抛）：路径写错或入库后被删掉，PG 仍会留下一条 0 字节文档记录。
        file_size = fp.stat().st_size if fp.exists() else 0
        session_factory = _get_sync_session_factory()
        # 整个函数一个事务：commit 只在最后；中途异常由 with 退出 + rollback 丢弃，检索存储不受影响。
        with session_factory() as session:
            # select-then-insert 没有行锁，也没有对 file_path 做唯一约束冲突处理：
            # 两个进程同时首次导入同一文件会各建一条文档，limit(1) 之后只认其中一条，另一条成为孤儿数据。
            existing = session.execute(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.file_path == str(fp)
                ).limit(1)
            ).scalar_one_or_none()
            if existing:
                # 更新：先删旧切片再写新的，版本号 +1，便于追溯"这份文档被改过几次"
                doc = existing
                # 按 document_id 手动删子表（不依赖数据库级联），与下面的插入同事务，PG 内部不会留下半套切片。
                session.execute(
                    delete(KnowledgeChunk).where(KnowledgeChunk.document_id == doc.id)
                )
                doc.title = filename
                doc.file_type = file_type
                doc.split_method = split_method
                # original_filename 也被写成磁盘上的最终文件名：后台看不到用户上传时的原始名（改名或去重后缀都会覆盖）。
                doc.original_filename = filename
                doc.file_size_bytes = file_size
                # 重复导入即使内容相同也增加版本，因此这里是替换式效果，非所有字段严格幂等。
                doc.version = (doc.version or 1) + 1
                # 无条件置 active：已被"下架"（status 改成非 active）的文档重新导入会被自动恢复可见。
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
                # flush 发送 INSERT 但未提交，后续插切片失败可在同一 PG 事务回滚。
                session.flush()          # flush 拿到自增 id，供下面切片外键使用
                created = True

            # 切片正文写 Text 列（无长度上限），但 Milvus content 是 VARCHAR(8192)：超长切片可能 PG 有、向量没有。
            session.add_all([
                KnowledgeChunk(
                    document_id=doc.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    # 只写这三列：retrieval_count/last_retrieved_at 走默认值，
                    # 且全项目没有任何自增点，后台按切片统计的"被召回次数"恒为 0。
                )
                for chunk in chunks
            ])
            session.commit()
            return doc.id, created
    except Exception as e:
        # 落库失败不影响向量检索（工具类和 RAG 仍可用），但必须在日志里可见
        get_logger(__name__).warning("ingestion DB persist failed: %s", e)
        # 检索存储可能已成功而 PG 失败；上层需看 document_id 和日志，不应只凭函数返回就认为三处一致。
        return None, False


def ingest_file(file_path: str) -> dict:
    """Ingest a single file — split → embed → store in Milvus + BM25.

    Returns stats dict. Also persists BM25 index after ingestion.
    后台"上传文档"走的就是这条路径（单文件、即时生效）。
    """
    # 计时起点：monotonic，覆盖分切、embedding、三份存储的全部等待。
    t0 = time.monotonic()
    fp = Path(file_path)
    # 这里只校验存在性；无权限或路径是目录时会在后面的读取阶段抛出别的错误。
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 写序固定：split → embed → 建集合 → 按 source 删 → 插 → 重建 BM25 → 最后写 PG。
    # 每一步成功都会保留下来，后续步骤失败不会回滚前面的，所以中途报错要整份重跑（重跑会覆盖，不叠加）。
    content = _read_file_content(fp)
    ft = _file_type(fp)
    chunks = auto_split(content, ft)

    # 切不出片（空文件、或 PDF 无文本层）就直接返回全 0：不动 Milvus/BM25/PG，旧数据原样留着。
    if not chunks:
        return {"files": 0, "chunks": 0, "embedded": 0, "milvus_count": 0, "bm25_count": 0}

    # Embed
    texts = [c.content for c in chunks]
    vectors: list[list[float]] = []
    # 单文件也按 10 条一批请求；embedding 失败向上抛，此时 Milvus/BM25/PG 都还没被动过。
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        # 按文本顺序拼接批次向量；后面 zip 假设数量与顺序吻合，当前未严格核验全部向量数量。
        vectors.extend(embedder.embed(batch))

    # Use same root-relative naming convention
    root = DATA_DIR
    # 要求文件位于 DATA_DIR 内；从其他目录直接传路径会因 relative_to 失败。
    src = _source_name(fp, root)

    # Milvus rows
    milvus_rows: list[dict] = []
    bm25_docs: list[Bm25Doc] = []
    # idx 同样未被使用；chunk_id 只由 src 和片内序号决定，因此同一片重导入后身份不变，可被两路融合认出。
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
    # 集合已存在时 create_collection 只把 _collection_ready 置真，不校验维度与当前 embedding_dim 是否一致。
    if not vector_store.is_ready:
        vector_store.create_collection()
    # 删后插之间存在空窗；插入失败时旧向量已删，不具备自动补偿。
    vector_store.delete_by_source(src)      # 同一文件重复上传：先删掉旧的
    insert_count = vector_store.insert(milvus_rows)
    vector_store.load()

    # Merge into existing BM25 index and rebuild
    # BM25 只支持整体重建：把"除本文档外"的旧切片保留，再拼上新切片
    existing = [
        # 保留当前进程加载过的其他文档；若未先 load 磁盘索引，可能丢失旧 BM25 语料。
        doc for doc in (bm25_store._docs[:] if bm25_store.is_ready else [])
        if doc.source_document != src
    ]
    # 直接读私有属性 _docs（拷贝一份避免边遍历边改），因为 Bm25Store 没有暴露只读的 docs 接口。
    existing.extend(bm25_docs)
    # 全量重建会按新语料重算 idf：加一个文件就可能改变所有文档的得分分布，与只更新本文件的语义不符。
    bm25_store.build(existing)
    # 落盘在这一步：若 build 成功而 save 失败，内存索引是新的、磁盘 pkl 仍是旧的，重启后回退。
    bm25_store.save(BM25_INDEX_PATH)

    # Persist to PostgreSQL
    # 这是 PG 展示标签；pdf/docx 实际走 SemanticSplitter，xlsx 走 ExcelSplitter，默认标签会失真。
    split_method = {"md": "structure", "faq": "qa_boundary", "txt": "fixed_size"}.get(ft, "fixed_size")
    # 最后同步 PG；Milvus、BM25、PG 三步无统一事务，失败后需对账/重新导入。
    # PG 用绝对 file_path 作键、且把 version+1，与上面两路存储的相对 source 不是同一套身份。
    doc_id, _created = _persist_to_db(str(fp), chunks, ft, split_method)

    return {
        "files": 1,
        "chunks": len(chunks),                    # 切出的片数
        "embedded": len(vectors),                 # 向量数，少于 chunks 说明供应商少返回，zip 已静默截短
        "milvus_count": insert_count,             # Milvus 回执写入数
        "bm25_count": len(bm25_store),            # 合并后的全量文档数，不只是本文件的片数
        "elapsed_ms": int((time.monotonic() - t0) * 1000),   # 含 embedding 与三次存储的阻塞时间
        "source": src,                            # 相对来源名：前端拿它展示，删除也只能按它匹配
        "document_id": doc_id,                    # PG 主键；落库失败时为 None，但本函数仍然"成功返回"
    }


def sync_db_all() -> dict:
    """Sync all existing data/ files to PostgreSQL without re-embedding.

    Reads each file, splits it (no embedding needed), and creates
    knowledge_documents + knowledge_chunks records, replacing existing chunks and incrementing version.
    用于"向量库已经有数据、PG 却是空的"这种修复场景。
    """
    # 只扫盘重建 PG，完全不碰 Milvus 与 BM25：因此 PG 里可能出现"向量库里没有对应向量"的切片。
    files = _collect_files(DATA_DIR)
    # 空目录返回的键比正常分支少（没有 failed/chunks 等），调用方按 dict.get 取值才安全。
    if not files:
        return {"files": 0, "documents": 0, "chunks": 0}

    # 以下计数器都只统计 PG 侧结果，与检索存储无关。
    docs_synced = 0
    docs_created = 0
    docs_existing = 0
    chunks_created = 0
    failed = 0
    for fp in files:
        content = _read_file_content(fp)
        ft = _file_type(fp)
        # 这是 PG 展示标签；pdf/docx 实际走 SemanticSplitter，xlsx 走 ExcelSplitter，默认标签会失真。
        split_method = {"md": "structure", "faq": "qa_boundary", "txt": "fixed_size"}.get(ft, "fixed_size")
        # 重新分切而不是复用已入库的切片：切分器一旦改过默认值或文件已变更，PG 文本就会和向量库版本分叉。
        chunks = auto_split(content, ft)
        doc_id, created = _persist_to_db(str(fp), chunks, ft, split_method)
        # 用真值判断成败：自增主键从 1 起，所以 0 不会误判；但 _persist_to_db 内部吞掉的异常在这里只表现为 failed+1。
        if doc_id:
            docs_synced += 1
            if created:
                docs_created += 1
                chunks_created += len(chunks)
            else:
                # 替换式更新不累加切片数，所以返回的 "chunks" 只反映新建文档，不代表知识库总切片数。
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
