"""Document splitters for multiple formats: Markdown, FAQ, PDF, Word, Excel, plain text."""

from __future__ import annotations

import re  # 三种切分策略都靠正则找边界（标题行 / Q: / 空行），没有引入分词或语义模型依赖
from dataclasses import dataclass, field


# 只是用 dataclass 省掉手写 __init__；Chunk 没有排序、哈希或校验语义，字段顺序即序列化顺序。
@dataclass
class Chunk:
    """A chunk unit returned by any splitter pipeline."""
    # 每个 Chunk 保留独立可读正文；后面 embedding 和 BM25 处理同一份内容。
    content: str
    # 源文件内顺序号；与 source 组合成逻辑 chunk_id，不是 Milvus 自动生成的数值主键。
    chunk_index: int
    # 每个切片独立元数据字典；当前主要保存标题和表格行范围。
    metadata: dict[str, str] = field(default_factory=dict)


# ── Text splitters ────────────────────────────────────────────────────────────

class MarkdownSplitter:
    """Split Markdown by H1/H2/H3 structure."""

    def split(self, text: str) -> list[Chunk]:
        # 按出现顺序追加；chunk_index 与列表下标一致，ingestion 用它拼 chunk_id 做两路融合去重。
        chunks: list[Chunk] = []
        # 零宽前瞻在标题前切分，保留标题本身；MULTILINE 让 ^ 匹配每一行开头。
        sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
        # 序号从 0 起；重复导入同一文件会重新从 0 编号，因此 chunk_id 不是全局唯一 ID。
        idx = 0
        for section in sections:
            # 移除块首尾空白，块内部换行保留，不重新排版知识内容。
            section = section.strip()
            # 纯空白段（例如首个标题之前的分隔行）直接丢，不占用 chunk_index，避免编号出现空洞。
            if not section:
                continue
            # 只在片段首行匹配标题：正文中间的 # 不会被当成新节；#{1,3} 与切分正则一致，H4 及更深标题留在父节内。
            heading_match = re.match(r"^(#{1,3})\s+(.+)", section)
            # 保存标题文字便于来源展示；标题之前的前言允许 heading 为空。
            heading = heading_match.group(2) if heading_match else ""
            # 一个标题节直接成为一片；没有长度二次切分，超长节可能超过 embedding 或 VARCHAR 限制。
            chunks.append(Chunk(content=section, chunk_index=idx, metadata={"heading": heading}))
            # 只增序号不做内容去重；同一段文字在两处出现会产生两条独立向量。
            idx += 1
        return chunks


class FAQSplitter:
    """Split by Q&A natural boundaries."""

    def split(self, text: str) -> list[Chunk]:
        # 以 Q:/问: 开头开始一片，把问题与其后答案一起保留；不是按每一个换行切片。
        # A:/答: 不作为切分点，答案只会带上下一个问题；冒号写半角或全角都能匹配。
        pairs = re.split(r"(?=^(?:Q|问)[：:])", text, flags=re.MULTILINE)
        chunks: list[Chunk] = []
        # 与 MarkdownSplitter 各自独立从 0 编号，跨文件的全局唯一性由 source 前缀保证。
        idx = 0
        for pair in pairs:
            # 去掉块前后空白与换行；QA 内部换行保留，向量与 BM25 看到的是同一段原文。
            pair = pair.strip()
            # 首行之前没有 Q:/问: 的说明文字会被切成空段跳过；一条 Q 都没有时整个文件产出 0 片。
            if not pair:
                continue
            # 不写 metadata：FAQ 片没有 heading，入库时 heading 字段落成空串，展示层只能回显正文。
            chunks.append(Chunk(content=pair, chunk_index=idx))
            idx += 1
        return chunks


class FixedSizeSplitter:
    """Fixed-size sliding window splitter for plain text."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        # 单位是 Python 字符数，不是 token；默认 512 字符。当前未检查 chunk_size>overlap。
        self.chunk_size = chunk_size
        # 相邻片共享默认 64 字符以保留边界上下文；overlap>=chunk_size 可能不前进。
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        # 空串或纯空白也走这个循环：range 条件不成立时返回空列表，不会造出一条空 Chunk。
        chunks: list[Chunk] = []
        # idx 负责逻辑顺序，start 是原文偏移，两者不要混同。
        idx, start = 0, 0
        while start < len(text):
            # 末片截到文本真实长度，避免为了凑长度填充无意义内容。
            end = min(start + self.chunk_size, len(text))
            # Python 切片左闭右开，例如 [0:512] 含 512 字符。
            chunks.append(Chunk(content=text[start:end], chunk_index=idx))
            idx += 1
            # 末片完成立即停止，否则 overlap 回退可能再产生重复尾片。
            if end >= len(text):
                break
            # 默认每次前进 448 字符；例：0..511 后一片从 448 开始。
            start = end - self.overlap
        return chunks


class SemanticSplitter:
    """Split by paragraph/section boundaries for documents with structure."""

    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 200) -> None:
        # 800/200 都是写死的默认值，没有对应的配置项；auto_split 也不会传参，改阈值只能改代码。
        self.max_chunk_size = max_chunk_size
        # min 只决定"要不要现在收口"，不决定"要不要输出"，尾块可以短于 200。
        self.min_chunk_size = min_chunk_size

    def split(self, text: str) -> list[Chunk]:
        # First, split by double newlines (paragraphs)
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        chunks: list[Chunk] = []
        idx = 0
        # current 是待合并的缓冲块；strip 后可能为空串，与"内容为空的段落"共用同一表示。
        current = ""
        for para in paras:
            # 按段落长度启发式合并，没有计算段落向量相似度；SemanticSplitter 的名字不代表语义模型。
            # 判据不含两段之间补回的 "\n\n"，且用严格小于，实际长度可以比 800 多出 2 个字符以上。
            if len(current) + len(para) < self.max_chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                # 已有块足够长才输出；否则继续并入下一段，因此 max_chunk_size 不是严格硬上限。
                if len(current) >= self.min_chunk_size:
                    chunks.append(Chunk(content=current, chunk_index=idx))
                    idx += 1
                    # 上一个块已输出，让当前段作为下一个块起点；超长单段也不会递归切开。
                    current = para
                else:
                    # 缓冲块还太短就继续吞下一段，块可能远大于 800；单段自身超限时同样不再细分。
                    current = (current + "\n\n" + para).strip()

        # 循环结束后输出尾块，允许不足 min_chunk_size，避免静默丢掉最后一段。
        if current.strip():
            chunks.append(Chunk(content=current, chunk_index=idx))

        return chunks


class ExcelSplitter:
    """Split Excel by rows, preserving header row in each chunk."""

    def __init__(self, rows_per_chunk: int = 10) -> None:
        # 默认 10 行；auto_split 的 xlsx 分支显式传同样的值，所以这里改默认值不会影响入库路径。
        self.rows_per_chunk = rows_per_chunk

    def split(self, rows: list[list[str]], headers: list[str] | None = None) -> list[Chunk]:
        # 空行列表返回空结果：该文件不产生 Chunk，也就不会写入任何一份存储（静默跳过）。
        if not rows:
            return []

        # headers 由调用方决定；xlsx 分支因下标错位通常传空列表，此时 header_line 为假值、不重复表头。
        header_line = "\t".join(headers) if headers else ""
        chunks: list[Chunk] = []
        idx = 0
        # 表格按十行一组，保留行内字段关联；不是每个单元格独立 embedding。
        for i in range(0, len(rows), self.rows_per_chunk):
            # 尾批可能不足 10 行；Python 切片自动截断，所以片数等于向上取整(len/rows_per_chunk)。
            batch = rows[i:i + self.rows_per_chunk]
            # 传入了表头才逐片重复，保证切片离开原表也能理解字段；调用方漏传就没有表头。
            lines = [header_line] if header_line else []
            for row in batch:
                # 每个单元格 str() 化：None 在提取阶段已变空串，这里不做数值/日期格式化。
                lines.append("\t".join(str(c) for c in row))
            chunks.append(Chunk(
                content="\n".join(lines),
                chunk_index=idx,
                # rows 记 [起始, 结束) 行区间，仅供来源展示；检索与入库都不解析这个值。
                metadata={"rows": f"{i}-{i + len(batch)}", "format": "table"},
            ))
            idx += 1
        return chunks


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text(file_path: str, file_type: str) -> str:
    """Extract plain text from PDF, Word, or Excel files."""
    # 只按 file_type 字符串分支（来自扩展名映射），不嗅探文件头；三个分支之外的类型一律返回空串。
    if file_type == "pdf":
        return _extract_pdf(file_path)
    elif file_type == "docx":
        return _extract_docx(file_path)
    elif file_type == "xlsx":
        return _extract_xlsx(file_path)
    # 未知二进制类型静默返回 ""，上层会切出 0 个 Chunk，看起来"入库成功但什么都没查到"。
    return ""


def _extract_pdf(path: str) -> str:
    """Extract text from PDF using PyPDF2 or pdfplumber."""
    try:
        # 解析库在函数内 import：只有真的遇到 PDF 才要求安装，缺库不影响其他文件入库。
        import pdfplumber
        texts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                # 只抽文本层，没有 OCR；扫描 PDF 图片可能得到空文本。
                t = page.extract_text()
                # 空页被跳过而不是保留占位，所以"第几页"这种位置信息在下游彻底丢失。
                if t:
                    texts.append(t)
        # 页之间补两个换行，正好让 SemanticSplitter 的 \n\s*\n 段落正则把跨页内容分开看。
        return "\n\n".join(texts)
    # 仅库缺失触发该分支；文件损坏或解析失败不自动换另一个解析器。
    except ImportError:
        pass

    try:
        # 兜底解析器：接口更简单但版面还原更差，同一份 PDF 换个库就可能切出不同内容。
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # 仅库缺失触发该分支；文件损坏或解析失败不自动换另一个解析器。
    except ImportError:
        raise ImportError("需要安装 pdfplumber 或 PyPDF2: pip install pdfplumber")


def _extract_docx(path: str) -> str:
    """Extract text from Word .docx using python-docx."""
    try:
        from docx import Document
        doc = Document(path)
        paras: list[str] = []
        # 当前只读取 Word 顶层段落，不遍历表格、文本框、图片及页眉页脚。
        for para in doc.paragraphs:
            # 靠样式名识别标题（依赖文档用了 Heading 样式）；手工加粗的大字号文字不算标题。
            if para.style.name.startswith("Heading"):
                # 取样式名最后一段当层级，"Heading 2" → "2"；中文样式"标题 2"不以 Heading 开头，整段按正文处理。
                level = para.style.name.split()[-1]
                try:
                    lv = int(level)
                except ValueError:
                    # 样式名不规范（如 "Heading One"）时退化成 1 级标题，而不是抛错中断整个文件。
                    lv = 1
                # 转成 Markdown 标题交给 SemanticSplitter；min(lv,3) 与 MarkdownSplitter 的 #{1,3} 对齐，H4 及更深按 H3 处理。
                paras.append(f"{'#' * min(lv, 3)} {para.text}")
            elif para.text.strip():
                # 空段落（仅用于排版的换行）直接丢弃，不产生额外切片边界。
                paras.append(para.text)
        # 段落之间补空行，作为后续 \n\s*\n 的唯一切分依据。
        return "\n\n".join(paras)
    # 仅库缺失触发该分支；文件损坏或解析失败不自动换另一个解析器。
    except ImportError:
        raise ImportError("需要安装 python-docx: pip install python-docx")


def _extract_xlsx(path: str) -> str:
    """Extract text from Excel, returning tab-separated rows."""
    try:
        from openpyxl import load_workbook
        # 只读降低内存，data_only 取缓存计算值而不执行公式；无缓存公式可能为空。
        wb = load_workbook(path, read_only=True, data_only=True)
        all_lines: list[str] = []
        # 遍历全部工作表：所有 sheet 拼成一段文本，切片之间没有 sheet 归属信息。
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            # "## 名称" 是 auto_split xlsx 分支识别 sheet 边界的唯一约定；单元格里以 "## " 开头的正文会被误当成标记。
            all_lines.append(f"## {sheet_name}")
            for row in ws.iter_rows(values_only=True):
                # 空单元格转空串而不是跳过，保留制表符位置，让列对齐靠位置而非字段名。
                values = [str(c) if c is not None else "" for c in row]
                all_lines.append("\t".join(values))
        # read_only 模式必须显式关闭，否则文件句柄留到进程结束（Windows 上会阻碍后续覆盖导入）。
        wb.close()
        # 这里只用单个换行连接（与 pdf/docx 不同），sheet 标记靠 startswith 判断，不依赖空行分段。
        return "\n".join(all_lines)
    # 仅库缺失触发该分支；文件损坏或解析失败不自动换另一个解析器。
    except ImportError:
        raise ImportError("需要安装 openpyxl: pip install openpyxl")


# ── Routing ───────────────────────────────────────────────────────────────────

def auto_split(text: str, file_type: str, file_path: str = "") -> list[Chunk]:
    """Route to the correct splitter based on file type."""
    # 路由只看 file_type（由 ingestion 按扩展名映射得到），不看内容、也不看 file_path；
    # 形参 file_path 在所有分支里都没被使用，调用方也都没有传（.md 内容存成 .txt 就会被当定长切）。
    if file_type == "md":
        return MarkdownSplitter().split(text)
    if file_type == "faq":
        return FAQSplitter().split(text)
    if file_type in ("pdf", "docx"):
        # 两种二进制格式都已在 extract_text 阶段压成"标题 + 空行分段"的纯文本，故共用段落启发式；
        # 切分器每调用一个新实例，无状态，可安全复用但当前没有缓存。
        return SemanticSplitter().split(text)
    if file_type == "xlsx":
        # Re-parse the tab-separated text back into rows for ExcelSplitter
        # 这里把 _extract_xlsx 的制表符文本再拆回行列（解析→序列化→再解析），只为一套接口同时支持两种调用来源。
        lines = text.strip().split("\n")
        rows: list[list[str]] = []
        headers: list[str] = []
        for i, line in enumerate(lines):
            # 跳过 sheet 标记后仍使用原始 enumerate 下标，下一行通常不是 i=0，导致表头未识别。
            if line.startswith("## "):
                continue  # skip sheet name markers
            # 只按 \t 拆列，不做引号转义与列数对齐；单元格内含制表符会错位。
            cells = line.split("\t")
            # 当前 xlsx 提取先写 sheet 标记，首个实际表头会被当数据；多 sheet 表头也不会分别维护。
            if i == 0:
                headers = cells
            else:
                rows.append(cells)
        # 表头识别失败时 headers 为空，切片就不带字段名；rows 为空则整个文件返回 0 片。
        return ExcelSplitter(rows_per_chunk=10).split(rows, headers)
    # 兜底：txt 以及所有未列出的类型都走 512/64 定长窗口，不看标点也不看段落。
    return FixedSizeSplitter().split(text)
