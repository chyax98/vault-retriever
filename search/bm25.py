"""BM25S 关键词搜索（内存优化版）"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass

import bm25s
import jieba

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    path: str
    score: float
    snippet: str


class BM25Search:
    """BM25S 搜索引擎（基于稀疏矩阵，支持 mmap）"""

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path
        self.index_dir = storage_path / "bm25_index" if storage_path else None

        self.retriever: bm25s.BM25 | None = None
        self.paths: list[str] = []  # 文档路径列表
        self._indexed = False

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        """中文分词"""
        result = []
        for text in texts:
            # 清理特殊字符
            text = re.sub(r'[#\[\](){}]', ' ', text)
            tokens = list(jieba.cut(text))
            result.append(tokens)
        return result

    def _tokenize_single(self, text: str) -> list[str]:
        """单条文本分词"""
        text = re.sub(r'[#\[\](){}]', ' ', text)
        return list(jieba.cut(text))

    @staticmethod
    def get_snippet(content: str, query_terms: set[str], max_len: int = 150) -> str:
        """提取包含查询词的片段"""
        lines = content.split('\n')

        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in query_terms if len(term) > 1):
                if len(line) > max_len:
                    return line[:max_len] + "..."
                return line

        return content[:max_len] + "..." if len(content) > max_len else content

    def index(self, documents: dict[str, str]):
        """建立索引"""
        self.paths = list(documents.keys())

        if not self.paths:
            self.retriever = None
            self._indexed = False
            return

        # 分词
        contents = [documents[p] for p in self.paths]
        corpus_tokens = self._tokenize(contents)

        # 创建 BM25 索引
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(contents, stemmer=None, stopwords=None, show_progress=False))
        self._indexed = True

        # 保存索引（如果有存储路径）
        if self.index_dir:
            self._save_index()

    def _save_index(self):
        """保存索引到磁盘"""
        if not self.retriever or not self.index_dir:
            return

        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.retriever.save(str(self.index_dir))

            # 单独保存路径列表
            paths_file = self.index_dir / "paths.txt"
            paths_file.write_text("\n".join(self.paths), encoding="utf-8")

            logger.info(f"BM25 索引已保存: {self.index_dir}")
        except Exception as e:
            logger.warning(f"保存 BM25 索引失败: {e}")

    def load_index(self, use_mmap: bool = True) -> bool:
        """从磁盘加载索引（支持 mmap）"""
        if not self.index_dir or not self.index_dir.exists():
            return False

        try:
            # 加载路径列表
            paths_file = self.index_dir / "paths.txt"
            if not paths_file.exists():
                return False
            self.paths = paths_file.read_text(encoding="utf-8").strip().split("\n")

            # 加载 BM25 索引（使用 mmap 节省内存）
            self.retriever = bm25s.BM25.load(str(self.index_dir), mmap=use_mmap)
            self._indexed = True

            logger.info(f"BM25 索引已加载 (mmap={use_mmap}): {len(self.paths)} 文档")
            return True
        except Exception as e:
            logger.warning(f"加载 BM25 索引失败: {e}")
            return False

    def is_indexed(self) -> bool:
        """检查是否已索引"""
        return self._indexed and self.retriever is not None

    def search_paths(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """搜索，只返回路径和分数（不需要文档内容）"""
        if not self.retriever or not self.paths:
            return []

        query_tokens = bm25s.tokenize([query], stemmer=None, stopwords=None, show_progress=False)
        results_ids, scores = self.retriever.retrieve(query_tokens, k=min(limit, len(self.paths)))

        results = []
        for i in range(results_ids.shape[1]):
            doc_idx = results_ids[0, i]
            score = scores[0, i]
            if score > 0 and doc_idx < len(self.paths):
                results.append((self.paths[doc_idx], float(score)))

        return results

    def search(
        self,
        query: str,
        doc_contents: dict[str, str],
        limit: int = 10
    ) -> list[SearchResult]:
        """搜索"""
        if not self.retriever or not self.paths:
            return []

        # 分词查询
        query_tokens = bm25s.tokenize([query], stemmer=None, stopwords=None, show_progress=False)
        query_terms = set(t.lower() for t in self._tokenize_single(query))

        # 检索
        results_ids, scores = self.retriever.retrieve(query_tokens, k=min(limit, len(self.paths)))

        # 构建结果
        results = []
        for i in range(results_ids.shape[1]):
            doc_idx = results_ids[0, i]
            score = scores[0, i]

            if score > 0 and doc_idx < len(self.paths):
                path = self.paths[doc_idx]
                content = doc_contents.get(path, "")
                results.append(SearchResult(
                    path=path,
                    score=float(score),
                    snippet=self.get_snippet(content, query_terms),
                ))

        return results

    @property
    def documents(self) -> dict[str, str]:
        """兼容旧接口（返回空字典，不存储原始文档）"""
        return {}
