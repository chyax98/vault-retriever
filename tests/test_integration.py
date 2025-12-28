"""集成测试 - 测试组件间的协作"""

import pytest
from pathlib import Path
import tempfile
import shutil

from search.bm25 import BM25Search
from search.vector import VectorSearch
from search.indexer import Indexer
from vault.reader import VaultReader


class TestSearchIntegration:
    """搜索模块集成测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def storage_dir(self, temp_dir):
        """创建存储目录"""
        storage = temp_dir / "storage"
        storage.mkdir()
        return storage

    @pytest.fixture
    def sample_docs(self):
        """示例文档"""
        return {
            "python/basics.md": "Python 是一种解释型编程语言，以简洁著称",
            "python/advanced.md": "Python 高级特性包括装饰器、生成器和元类",
            "javascript/intro.md": "JavaScript 是 Web 开发的核心语言，运行在浏览器中",
            "rust/memory.md": "Rust 的所有权系统确保内存安全，无需垃圾回收",
        }

    @pytest.fixture
    def sample_stats(self, sample_docs):
        """示例文件统计"""
        return {path: (1000.0 + i, len(content)) for i, (path, content) in enumerate(sample_docs.items())}

    def test_bm25_indexer_integration(self, storage_dir, sample_docs, sample_stats):
        """测试 BM25 与 Indexer 的集成"""
        bm25 = BM25Search(storage_path=storage_dir)
        vector = VectorSearch(storage_path=storage_dir)
        indexer = Indexer(
            storage_path=storage_dir,
            bm25=bm25,
            vector=vector,
            interval=300,
            vector_ready_fn=lambda: False,  # 禁用向量索引
        )

        # 全量索引
        indexer.index_full(sample_docs, sample_stats)
        assert bm25.is_indexed()

        # 搜索测试
        results = bm25.search_paths("Python", limit=10)
        paths = [p for p, _ in results]
        assert any("python" in p for p in paths)

    def test_incremental_index_workflow(self, storage_dir, sample_docs, sample_stats):
        """测试增量索引工作流"""
        bm25 = BM25Search(storage_path=storage_dir)
        vector = VectorSearch(storage_path=storage_dir)
        indexer = Indexer(
            storage_path=storage_dir,
            bm25=bm25,
            vector=vector,
            interval=300,
            vector_ready_fn=lambda: False,
        )

        # 初始索引
        indexer.index_full(sample_docs, sample_stats)
        initial_paths = set(bm25.paths)

        # 添加新文档
        new_docs = {**sample_docs, "go/intro.md": "Go 语言由 Google 开发"}
        new_stats = {**sample_stats, "go/intro.md": (2000.0, 50)}

        result = indexer.index_incremental(new_docs, new_stats)
        assert result["status"] == "updated"
        assert result["added"] == 1

        # 验证新文档可搜索
        results = bm25.search_paths("Google", limit=10)
        paths = [p for p, _ in results]
        assert "go/intro.md" in paths

    def test_delete_workflow(self, storage_dir, sample_docs, sample_stats):
        """测试删除工作流"""
        bm25 = BM25Search(storage_path=storage_dir)
        vector = VectorSearch(storage_path=storage_dir)
        indexer = Indexer(
            storage_path=storage_dir,
            bm25=bm25,
            vector=vector,
            interval=300,
            vector_ready_fn=lambda: False,
        )

        # 初始索引
        indexer.index_full(sample_docs, sample_stats)

        # 删除文档
        del_path = "python/basics.md"
        new_docs = {k: v for k, v in sample_docs.items() if k != del_path}
        new_stats = {k: v for k, v in sample_stats.items() if k != del_path}

        result = indexer.index_incremental(new_docs, new_stats)
        assert result["status"] == "updated"
        assert result["deleted"] == 1

        # 验证删除的文档不再出现在索引中
        assert del_path not in bm25.paths

    def test_modify_workflow(self, storage_dir, sample_docs, sample_stats):
        """测试修改工作流"""
        bm25 = BM25Search(storage_path=storage_dir)
        vector = VectorSearch(storage_path=storage_dir)
        indexer = Indexer(
            storage_path=storage_dir,
            bm25=bm25,
            vector=vector,
            interval=300,
            vector_ready_fn=lambda: False,
        )

        # 初始索引
        indexer.index_full(sample_docs, sample_stats)

        # 修改文档（更新 mtime 和内容）
        modified_docs = sample_docs.copy()
        modified_docs["python/basics.md"] = "Python 是最流行的编程语言之一，适合初学者"
        modified_stats = sample_stats.copy()
        modified_stats["python/basics.md"] = (3000.0, len(modified_docs["python/basics.md"]))

        result = indexer.index_incremental(modified_docs, modified_stats)
        assert result["status"] == "updated"
        assert result["modified"] == 1

    def test_bm25_persistence(self, storage_dir, sample_docs, sample_stats):
        """测试 BM25 索引持久化"""
        # 创建并索引
        bm25 = BM25Search(storage_path=storage_dir)
        bm25.index(sample_docs)

        # 创建新实例并加载
        bm25_new = BM25Search(storage_path=storage_dir)
        loaded = bm25_new.load_index(use_mmap=False)

        assert loaded
        assert set(bm25_new.paths) == set(bm25.paths)

        # 验证搜索功能
        results = bm25_new.search_paths("Python", limit=10)
        assert len(results) > 0


class TestVaultReaderIntegration:
    """VaultReader 集成测试"""

    @pytest.fixture
    def temp_vault(self):
        """创建临时 vault"""
        d = tempfile.mkdtemp()
        vault_path = Path(d)

        # 创建 .obsidian 目录
        (vault_path / ".obsidian").mkdir()

        # 创建测试笔记
        notes = {
            "note1.md": "# 笔记1\n\n这是第一篇笔记，链接到 [[note2]]。\n\n#python #学习",
            "note2.md": "# 笔记2\n\n这是第二篇笔记，链接到 [[note3]]。\n\n#javascript",
            "folder/note3.md": "# 笔记3\n\n这是第三篇笔记。\n\n#python",
        }

        for path, content in notes.items():
            file_path = vault_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        yield vault_path
        shutil.rmtree(d)

    def test_list_notes(self, temp_vault):
        """测试列出笔记"""
        reader = VaultReader(temp_vault)
        notes = reader.list_notes()
        assert len(notes) == 3
        assert "note1.md" in notes
        assert "folder/note3.md" in notes

    def test_read_note(self, temp_vault):
        """测试读取笔记"""
        reader = VaultReader(temp_vault)
        content = reader.read_note("note1.md")
        assert "# 笔记1" in content

    def test_get_links(self, temp_vault):
        """测试获取链接"""
        reader = VaultReader(temp_vault)
        links = reader.get_links("note1.md")
        assert "note2" in links.outgoing

    def test_get_all_tags(self, temp_vault):
        """测试获取所有标签"""
        reader = VaultReader(temp_vault)
        tags = reader.get_all_tags()
        assert "#python" in tags
        assert tags["#python"] == 2  # note1 和 note3

    def test_find_by_tag(self, temp_vault):
        """测试按标签查找"""
        reader = VaultReader(temp_vault)
        notes = reader.find_by_tag("python")
        assert len(notes) == 2

    def test_full_workflow(self, temp_vault):
        """测试完整工作流：读取 -> 索引 -> 搜索"""
        reader = VaultReader(temp_vault)
        storage = temp_vault / ".obsidian" / "test-storage"
        storage.mkdir(parents=True)

        # 加载所有文档
        docs = reader.load_all_documents()
        doc_contents = {d.path: d.content for d in docs}
        file_stats = {d.path: (d.mtime, len(d.content)) for d in docs}

        # 创建索引
        bm25 = BM25Search(storage_path=storage)
        vector = VectorSearch(storage_path=storage)
        indexer = Indexer(
            storage_path=storage,
            bm25=bm25,
            vector=vector,
            interval=300,
            vector_ready_fn=lambda: False,
        )

        indexer.index_full(doc_contents, file_stats)

        # 搜索（使用实际存在于文档中的关键词）
        results = bm25.search_paths("python", limit=10)
        assert len(results) > 0
