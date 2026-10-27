"""In-memory BM25 keyword index with pickle persistence."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class Bm25Doc:
    chunk_id: str
    content: str
    source_document: str
    heading: str
    chunk_index: int


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer — splits on whitespace and CJK character boundaries."""
    import re
    # Split on whitespace, and also separate CJK chars for partial matching
    tokens: list[str] = []
    # 先按空白分块；中文片段下一步按单字切分，没有引入中文分词模型。
    for word in text.split():
        # Split CJK runs into individual characters for recall
        if re.search(r'[一-鿿]', word):
            # 中文逐字召回可匹配局部，但也容易把不相关文本召回；混合片段的非中文部分未统一小写。
            tokens.extend(re.findall(r'[一-鿿]|[^一-鿿]+', word))
        else:
            tokens.append(word.lower())
    return [t for t in tokens if t.strip()]


class Bm25Store:
    """In-memory BM25 keyword search index."""

    def __init__(self) -> None:
        self._docs: list[Bm25Doc] = []
        self._index: BM25Okapi | None = None
        self._corpus: list[list[str]] = []

    def build(self, docs: list[Bm25Doc]) -> None:
        """Build BM25 index from documents."""
        self._docs = docs
        # 建库与 query 采用同一 tokenizer，否则词项不一致会降低召回。
        self._corpus = [_tokenize(d.content) for d in docs]
        if self._corpus:
            # 重新计算词频、文档频率和长度统计；当前应用接口采用全量重建。
            self._index = BM25Okapi(self._corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Bm25Doc, float]]:
        """Search and return top_k docs with normalized scores."""
        if not self._index or not self._docs:
            return []

        tokens = _tokenize(query)
        # BM25 按词项稀有程度、文档内频次和长度打分；原始分没有统一的概率含义。
        scores = self._index.get_scores(tokens)

        # Get top_k indices sorted by score descending
        # 保留下标是为了从 _docs 还原内容；下标和 corpus 必须始终同序。
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = indexed[:top_k]

        # Normalize scores to [0, 1]
        # 除以当前查询的最大正分，避免除零；不同 query 的归一化分仍不能直接比较置信度。
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        # 负分和零分直接过滤；得分 1 只表示本次最高正分，不表示绝对相关。
        return [(self._docs[i], score / max_score) for i, score in top if score > 0]

    def save(self, path: str) -> None:
        """Persist index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            # 保存语料与文档快照，load 后重建 BM25；这不是远程共享索引。
            pickle.dump({"docs": self._docs, "corpus": self._corpus}, f)

    def load(self, path: str) -> None:
        """Load index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with open(path, "rb") as f:
            # 只读取本项目可信索引文件；pickle 可执行反序列化代码，不能用它接收用户上传内容。
            data = pickle.load(f)
        self._docs = data["docs"]
        self._corpus = data["corpus"]
        if self._corpus:
            # 重新计算词频、文档频率和长度统计；当前应用接口采用全量重建。
            self._index = BM25Okapi(self._corpus)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._docs) > 0

    def __len__(self) -> int:
        return len(self._docs)


# Singleton
bm25_store = Bm25Store()
