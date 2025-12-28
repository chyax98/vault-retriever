"""VectorSearch 单元测试"""

import pytest
from pathlib import Path
import tempfile
import shutil

from search.vector import VectorSearch, SearchResult


class TestVectorSearch:
    """VectorSearch 测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def vector(self, temp_dir):
        """创建 VectorSearch 实例"""
        return VectorSearch(storage_path=temp_dir)

    @pytest.fixture
    def sample_docs(self):
        """示例文档"""
        return {
            "note1.md": "Python 是一种编程语言，广泛用于数据科学和机器学习",
            "note2.md": "JavaScript 是 Web 开发的核心语言",
            "note3.md": "Rust 是一种系统编程语言，注重安全性和性能",
        }

    def test_index_documents(self, vector, sample_docs):
        """测试文档索引"""
        vector.index(sample_docs)
        assert vector.is_indexed()

    def test_search(self, vector, sample_docs):
        """测试语义搜索"""
        vector.index(sample_docs)
        results = vector.search("数据科学", limit=10)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_add_single_document(self, vector, sample_docs):
        """测试添加单个文档"""
        vector.index(sample_docs)
        vector.add("note4.md", "新增的文档内容")
        stats = vector.get_stats()
        assert stats["indexed"] == 4

    def test_remove_document(self, vector, sample_docs):
        """测试删除文档"""
        vector.index(sample_docs)
        initial_count = vector.get_stats()["indexed"]
        vector.remove("note1.md")
        final_count = vector.get_stats()["indexed"]
        assert final_count == initial_count - 1

    def test_remove_nonexistent(self, vector, sample_docs):
        """测试删除不存在的文档"""
        vector.index(sample_docs)
        initial_count = vector.get_stats()["indexed"]
        vector.remove("nonexistent.md")
        final_count = vector.get_stats()["indexed"]
        assert final_count == initial_count

    def test_search_empty_index(self, vector):
        """测试空索引搜索"""
        results = vector.search("Python", limit=10)
        assert results == []

    def test_get_stats_empty(self, vector):
        """测试空索引统计"""
        stats = vector.get_stats()
        assert stats["indexed"] == 0

    def test_index_empty_documents(self, vector):
        """测试空文档索引"""
        vector.index({})
        assert not vector.is_indexed()
