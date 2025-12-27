"""索引管理器 - 缓存和增量更新"""

import json
import logging
import threading
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from .bm25 import BM25Search
from .vector import VectorSearch

logger = logging.getLogger(__name__)


@dataclass
class FileState:
    """文件状态"""
    mtime: float
    size: int


class Indexer:
    """索引管理器"""

    def __init__(
        self,
        storage_path: Path,
        bm25: BM25Search,
        vector: VectorSearch,
        interval: int = 300,
        vector_ready_fn: callable = None,
    ):
        self.storage_path = storage_path
        self.bm25 = bm25
        self.vector = vector
        self.interval = interval
        self._vector_ready_fn = vector_ready_fn or (lambda: True)

        self.cache_file = storage_path / "index_cache.json"
        self._file_states: dict[str, FileState] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    def _load_cache(self) -> dict[str, FileState]:
        """加载缓存"""
        if not self.cache_file.exists():
            return {}

        try:
            data = json.loads(self.cache_file.read_text())
            return {k: FileState(**v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return {}

    def _save_cache(self):
        """保存缓存"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {k: asdict(v) for k, v in self._file_states.items()}
            self.cache_file.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def _get_changed_files(
        self,
        current_docs: dict[str, tuple[str, float, int]],
    ) -> tuple[list[str], list[str], list[str]]:
        """检测文件变化

        Returns:
            (added, modified, deleted)
        """
        cached = self._load_cache()
        current_paths = set(current_docs.keys())
        cached_paths = set(cached.keys())

        added = list(current_paths - cached_paths)
        deleted = list(cached_paths - current_paths)

        modified = []
        for path in current_paths & cached_paths:
            content, mtime, size = current_docs[path]
            old = cached[path]
            if mtime != old.mtime or size != old.size:
                modified.append(path)

        return added, modified, deleted

    def index_full(self, documents: dict[str, str], file_stats: dict[str, tuple[float, int]]):
        """全量索引（只做 BM25，向量索引单独处理）"""
        logger.info(f"BM25 全量索引 {len(documents)} 个文档")

        # 只做 BM25 索引（快速）
        self.bm25.index(documents)

        # 更新缓存
        self._file_states = {
            path: FileState(mtime=stats[0], size=stats[1])
            for path, stats in file_stats.items()
        }
        self._save_cache()

    def index_incremental(
        self,
        all_docs: dict[str, str],
        file_stats: dict[str, tuple[float, int]],
    ) -> dict:
        """增量索引"""
        # 构建用于比较的数据
        current = {
            path: (content, file_stats[path][0], file_stats[path][1])
            for path, content in all_docs.items()
        }

        added, modified, deleted = self._get_changed_files(current)

        if not added and not modified and not deleted:
            return {"status": "unchanged"}

        logger.info(f"增量更新: +{len(added)}, ~{len(modified)}, -{len(deleted)}")

        # 如果新增文件过多，直接全量索引（避免逐个添加重建）
        if len(added) > 100:
            logger.info("新增文件过多，执行全量索引")
            self.index_full(all_docs, file_stats)
            return {
                "status": "updated",
                "added": len(added),
                "modified": len(modified),
                "deleted": len(deleted),
            }

        # 删除
        for path in deleted:
            self.bm25.remove(path)
            self.vector.remove(path)
            if path in self._file_states:
                del self._file_states[path]

        # 添加和修改 - 批量处理
        to_update = added + modified
        if to_update:
            # 更新 BM25（直接使用完整文档集重建）
            self.bm25.index(all_docs)

            # 更新向量索引（仅当向量索引已就绪时）
            if self._vector_ready_fn():
                for path in modified:
                    self.vector.remove(path)
                for path in to_update:
                    self.vector.add(path, all_docs[path])

            # 更新缓存状态
            for path in to_update:
                mtime, size = file_stats[path]
                self._file_states[path] = FileState(mtime=mtime, size=size)

        self._save_cache()

        return {
            "status": "updated",
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
        }

    def start_background(self, get_docs_fn):
        """启动后台定时更新"""
        if self._running:
            return

        self._running = True

        def worker():
            while self._running:
                time.sleep(self.interval)
                if not self._running:
                    break
                try:
                    docs, stats = get_docs_fn()
                    result = self.index_incremental(docs, stats)
                    if result["status"] == "updated":
                        logger.info(f"后台索引更新: {result}")
                except Exception as e:
                    logger.error(f"后台索引错误: {e}")

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        logger.info(f"后台索引已启动，间隔 {self.interval} 秒")

    def stop_background(self):
        """停止后台更新"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
