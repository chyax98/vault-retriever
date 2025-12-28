"""Indexer 单元测试"""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, MagicMock

from search.indexer import Indexer, FileState
from search.bm25 import BM25Search
from search.vector import VectorSearch


class TestIndexer:
    """Indexer 测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def mock_bm25(self):
        """模拟 BM25Search"""
        mock = Mock(spec=BM25Search)
        mock.index = Mock()
        mock.remove = Mock()
        return mock

    @pytest.fixture
    def mock_vector(self):
        """模拟 VectorSearch"""
        mock = Mock(spec=VectorSearch)
        mock.index = Mock()
        mock.add = Mock()
        mock.remove = Mock()
        return mock

    @pytest.fixture
    def indexer(self, temp_dir, mock_bm25, mock_vector):
        """创建 Indexer 实例"""
        return Indexer(
            storage_path=temp_dir,
            bm25=mock_bm25,
            vector=mock_vector,
            interval=300,
            vector_ready_fn=lambda: True,
        )

    @pytest.fixture
    def sample_docs(self):
        """示例文档"""
        return {
            "note1.md": "Python 内容",
            "note2.md": "JavaScript 内容",
        }

    @pytest.fixture
    def sample_stats(self):
        """示例文件统计"""
        return {
            "note1.md": (1000.0, 100),
            "note2.md": (1001.0, 120),
        }

    def test_index_full(self, indexer, mock_bm25, sample_docs, sample_stats):
        """测试全量索引"""
        indexer.index_full(sample_docs, sample_stats)

        mock_bm25.index.assert_called_once_with(sample_docs)
        assert indexer.cache_file.exists()

    def test_index_incremental_no_change(self, indexer, sample_docs, sample_stats):
        """测试无变化的增量索引"""
        # 先做一次全量索引
        indexer.index_full(sample_docs, sample_stats)

        # 再做增量索引，应该返回 unchanged
        result = indexer.index_incremental(sample_docs, sample_stats)
        assert result["status"] == "unchanged"

    def test_index_incremental_with_additions(
        self, indexer, mock_bm25, mock_vector, sample_docs, sample_stats
    ):
        """测试有新增文件的增量索引"""
        # 先做一次全量索引
        indexer.index_full(sample_docs, sample_stats)

        # 添加新文件
        new_docs = {**sample_docs, "note3.md": "新文档"}
        new_stats = {**sample_stats, "note3.md": (1002.0, 50)}

        result = indexer.index_incremental(new_docs, new_stats)

        assert result["status"] == "updated"
        assert result["added"] == 1
        assert result["modified"] == 0
        assert result["deleted"] == 0
        mock_bm25.index.assert_called_with(new_docs)

    def test_index_incremental_with_deletions(
        self, indexer, mock_bm25, mock_vector, sample_docs, sample_stats
    ):
        """测试有删除文件的增量索引"""
        # 先做一次全量索引
        indexer.index_full(sample_docs, sample_stats)

        # 删除一个文件
        new_docs = {"note1.md": sample_docs["note1.md"]}
        new_stats = {"note1.md": sample_stats["note1.md"]}

        result = indexer.index_incremental(new_docs, new_stats)

        assert result["status"] == "updated"
        assert result["deleted"] == 1
        mock_vector.remove.assert_called_with("note2.md")
        mock_bm25.index.assert_called_with(new_docs)

    def test_index_incremental_with_modifications(
        self, indexer, mock_bm25, mock_vector, sample_docs, sample_stats
    ):
        """测试有修改文件的增量索引"""
        # 先做一次全量索引
        indexer.index_full(sample_docs, sample_stats)

        # 修改一个文件（改变 mtime）
        new_docs = sample_docs.copy()
        new_stats = {**sample_stats, "note1.md": (2000.0, 100)}

        result = indexer.index_incremental(new_docs, new_stats)

        assert result["status"] == "updated"
        assert result["modified"] == 1
        mock_bm25.index.assert_called_with(new_docs)

    def test_cache_persistence(self, indexer, sample_docs, sample_stats, temp_dir):
        """测试缓存持久化"""
        indexer.index_full(sample_docs, sample_stats)

        # 创建新的 indexer 实例，使用相同的存储路径
        mock_bm25 = Mock(spec=BM25Search)
        mock_vector = Mock(spec=VectorSearch)
        new_indexer = Indexer(
            storage_path=temp_dir,
            bm25=mock_bm25,
            vector=mock_vector,
            interval=300,
        )

        # 增量索引应该检测到无变化
        result = new_indexer.index_incremental(sample_docs, sample_stats)
        assert result["status"] == "unchanged"

    def test_many_additions_triggers_full_reindex(
        self, indexer, mock_bm25, sample_docs, sample_stats
    ):
        """测试大量新增触发全量重建"""
        # 先做一次空的全量索引
        indexer.index_full({}, {})
        mock_bm25.reset_mock()

        # 添加超过 100 个文件
        many_docs = {f"note{i}.md": f"内容 {i}" for i in range(150)}
        many_stats = {f"note{i}.md": (float(i), i) for i in range(150)}

        result = indexer.index_incremental(many_docs, many_stats)

        assert result["status"] == "updated"
        assert result["added"] == 150


class TestFileState:
    """FileState 数据类测试"""

    def test_file_state_creation(self):
        """测试 FileState 创建"""
        state = FileState(mtime=1000.0, size=500)
        assert state.mtime == 1000.0
        assert state.size == 500

    def test_file_state_equality(self):
        """测试 FileState 相等性"""
        state1 = FileState(mtime=1000.0, size=500)
        state2 = FileState(mtime=1000.0, size=500)
        assert state1 == state2
