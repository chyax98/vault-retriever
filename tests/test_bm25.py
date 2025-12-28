"""BM25Search 单元测试"""

import pytest
from pathlib import Path
import tempfile
import shutil

from search.bm25 import BM25Search, SearchResult


class TestBM25Search:
    """BM25Search 测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def bm25(self, temp_dir):
        """创建 BM25Search 实例"""
        return BM25Search(storage_path=temp_dir)

    @pytest.fixture
    def sample_docs(self):
        """示例文档"""
        return {
            "note1.md": "Python 是一种编程语言，广泛用于数据科学和机器学习",
            "note2.md": "JavaScript 是 Web 开发的核心语言",
            "note3.md": "Rust 是一种系统编程语言，注重安全性和性能",
            "note4.md": "Python 的 NumPy 库用于科学计算",
        }

    def test_index_empty(self, bm25):
        """测试空文档索引"""
        bm25.index({})
        assert not bm25.is_indexed()
        assert bm25.paths == []

    def test_index_documents(self, bm25, sample_docs):
        """测试文档索引"""
        bm25.index(sample_docs)
        assert bm25.is_indexed()
        assert len(bm25.paths) == 4
        assert set(bm25.paths) == set(sample_docs.keys())

    def test_search_paths(self, bm25, sample_docs):
        """测试路径搜索"""
        bm25.index(sample_docs)
        results = bm25.search_paths("Python", limit=10)
        assert len(results) > 0
        paths = [p for p, _ in results]
        assert "note1.md" in paths or "note4.md" in paths

    def test_search_paths_empty_index(self, bm25):
        """测试空索引搜索"""
        results = bm25.search_paths("Python", limit=10)
        assert results == []

    def test_search_with_content(self, bm25, sample_docs):
        """测试带内容的搜索"""
        bm25.index(sample_docs)
        results = bm25.search("Python", sample_docs, limit=10)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_remove_existing_path(self, bm25, sample_docs):
        """测试删除存在的路径"""
        bm25.index(sample_docs)
        assert "note1.md" in bm25.paths
        bm25.remove("note1.md")
        assert "note1.md" not in bm25.paths
        assert len(bm25.paths) == 3

    def test_remove_nonexistent_path(self, bm25, sample_docs):
        """测试删除不存在的路径"""
        bm25.index(sample_docs)
        original_len = len(bm25.paths)
        bm25.remove("nonexistent.md")
        assert len(bm25.paths) == original_len

    def test_reindex_after_remove(self, bm25, sample_docs):
        """测试删除后重新索引"""
        bm25.index(sample_docs)
        bm25.remove("note1.md")
        # 用新的文档集重新索引
        new_docs = {k: v for k, v in sample_docs.items() if k != "note1.md"}
        bm25.index(new_docs)
        assert len(bm25.paths) == 3
        assert "note1.md" not in bm25.paths

    def test_save_and_load_index(self, bm25, sample_docs):
        """测试索引保存和加载"""
        bm25.index(sample_docs)

        # 创建新实例加载索引
        bm25_new = BM25Search(storage_path=bm25.storage_path)
        loaded = bm25_new.load_index(use_mmap=False)

        assert loaded
        assert bm25_new.is_indexed()
        assert set(bm25_new.paths) == set(sample_docs.keys())

    def test_get_snippet(self):
        """测试摘要提取"""
        content = "第一行内容\n这里包含 Python 关键词\n第三行内容"
        query_terms = {"python"}
        snippet = BM25Search.get_snippet(content, query_terms)
        assert "Python" in snippet or "python" in snippet.lower()

    def test_get_snippet_no_match(self):
        """测试无匹配时的摘要"""
        content = "这是一段没有关键词的内容"
        query_terms = {"python"}
        snippet = BM25Search.get_snippet(content, query_terms)
        assert snippet == content

    def test_search_limit(self, bm25, sample_docs):
        """测试搜索结果限制"""
        bm25.index(sample_docs)
        results = bm25.search_paths("编程语言", limit=2)
        assert len(results) <= 2
