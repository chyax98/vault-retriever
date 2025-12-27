"""Obsidian Vault MCP Server - 极简版（低内存占用）"""

import json
import logging
import threading
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from config import Config, load_config
from vault import VaultReader
from search import BM25Search, VectorSearch, Indexer
from memory import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== PageRank（持久化版） ==========

@dataclass
class GraphState:
    """图谱状态（持久化到磁盘）"""
    nodes: int = 0
    edges: int = 0
    scores: dict = None

    def __post_init__(self):
        if self.scores is None:
            self.scores = {}


class GraphRanker:
    """基于链接结构的 PageRank 排序（支持持久化）"""

    def __init__(self, storage_path: Path):
        self.cache_file = storage_path / "pagerank.json"
        self._scores: dict[str, float] = {}
        self._stats = {"nodes": 0, "edges": 0}
        self._lock = threading.Lock()
        self._ready = False

    def load(self) -> bool:
        """从磁盘加载"""
        if not self.cache_file.exists():
            return False
        try:
            data = json.loads(self.cache_file.read_text())
            with self._lock:
                self._scores = data.get("scores", {})
                self._stats = {"nodes": data.get("nodes", 0), "edges": data.get("edges", 0)}
                self._ready = True
            logger.info(f"PageRank 从缓存加载 ({self._stats['nodes']} 节点)")
            return True
        except Exception as e:
            logger.warning(f"加载 PageRank 失败: {e}")
            return False

    def build(self, links_map: dict[str, list[str]]) -> float:
        """构建图谱并计算 PageRank"""
        import networkx as nx

        start = time.time()
        G = nx.DiGraph()

        for source, targets in links_map.items():
            G.add_node(source)
            for target in targets:
                G.add_edge(source, target)

        try:
            scores = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            scores = {}

        with self._lock:
            self._scores = scores
            self._stats = {"nodes": len(G.nodes), "edges": len(G.edges)}
            self._ready = True

        # 持久化
        self._save()

        elapsed = (time.time() - start) * 1000
        logger.info(f"PageRank 计算完成 ({len(G.nodes)} 节点, {len(G.edges)} 边, {elapsed:.0f}ms)")
        return elapsed

    def _save(self):
        """保存到磁盘"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "scores": self._scores,
                "nodes": self._stats["nodes"],
                "edges": self._stats["edges"],
            }
            self.cache_file.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"保存 PageRank 失败: {e}")

    def get_score(self, path: str) -> float:
        with self._lock:
            return self._scores.get(path, 0.0)

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    def get_stats(self) -> dict:
        with self._lock:
            return self._stats.copy()


# ========== RRF 融合 ==========

def rrf_fusion(
    bm25_results: list[tuple[str, float]],
    vector_results: list,
    k: int = 60
) -> list[tuple[str, float]]:
    """RRF 融合算法（不含 snippet）"""
    scores: dict[str, float] = {}

    for rank, (path, _) in enumerate(bm25_results, 1):
        scores[path] = scores.get(path, 0) + 1 / (rank + k)

    for rank, r in enumerate(vector_results, 1):
        scores[r.path] = scores.get(r.path, 0) + 1 / (rank + k)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ========== 索引状态（极简版） ==========

class IndexState:
    """索引状态（不存储文档内容）"""

    def __init__(self):
        self.bm25_ready = False
        self.vector_ready = False
        self.total_docs = 0
        self._lock = threading.Lock()
        self.metrics = {
            "bm25_index_time": 0.0,
            "vector_index_time": 0.0,
            "pagerank_time": 0.0,
            "last_update": "",
        }

    def set_bm25_ready(self, doc_count: int):
        with self._lock:
            self.bm25_ready = True
            self.total_docs = doc_count

    def set_vector_ready(self):
        with self._lock:
            self.vector_ready = True

    def is_bm25_ready(self) -> bool:
        with self._lock:
            return self.bm25_ready

    def is_vector_ready(self) -> bool:
        with self._lock:
            return self.vector_ready

    def update_metrics(self, **kwargs):
        with self._lock:
            self.metrics.update(kwargs)

    def get_metrics(self) -> dict:
        with self._lock:
            return self.metrics.copy()


def create_server(vault_path: Path, config: Config | None = None) -> FastMCP:
    """创建 MCP 服务器（极简版）"""

    config = config or load_config(vault_path)
    storage = config.storage_path

    # 初始化组件
    vault = VaultReader(vault_path)
    bm25 = BM25Search(storage)
    vector = VectorSearch(storage, model_name=config.embedding_model)
    memory = MemoryStore(storage)
    state = IndexState()
    graph_ranker = GraphRanker(storage)

    indexer = Indexer(
        storage, bm25, vector,
        interval=config.index_interval,
        vector_ready_fn=state.is_vector_ready,
    )

    # 后台初始化 BM25
    def init_bm25():
        logger.info("后台初始化 BM25...")
        start = time.time()

        docs = vault.load_all_documents()
        doc_contents = {d.path: d.content for d in docs}
        file_stats = {d.path: (d.mtime, len(d.content)) for d in docs}

        if bm25.load_index(use_mmap=True):
            result = indexer.index_incremental(doc_contents, file_stats)
            if result["status"] == "updated":
                bm25.index(doc_contents)
        else:
            bm25.index(doc_contents)

        # 注意：不存储 doc_contents，只记录数量
        bm25_time = time.time() - start
        state.set_bm25_ready(len(doc_contents))
        state.update_metrics(bm25_index_time=bm25_time)
        logger.info(f"BM25 就绪 ({bm25_time:.2f}s, {len(doc_contents)} 文档)")

    # 后台初始化向量索引
    def init_vector():
        while not state.is_bm25_ready():
            time.sleep(0.5)

        if vector.is_indexed():
            state.set_vector_ready()
            state.update_metrics(last_update=time.strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("向量索引从缓存加载")
            return

        # 需要重新加载文档构建向量索引
        docs = vault.load_all_documents()
        doc_contents = {d.path: d.content for d in docs}

        logger.info(f"后台构建向量索引 ({len(doc_contents)} 文档)...")
        start = time.time()
        vector.index(doc_contents)
        vector_time = time.time() - start

        state.set_vector_ready()
        state.update_metrics(
            vector_index_time=vector_time,
            last_update=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        logger.info(f"向量索引就绪 ({vector_time:.2f}s)")

    # 后台初始化 PageRank
    def init_graph():
        # 先尝试从缓存加载
        if graph_ranker.load():
            return

        while not state.is_bm25_ready():
            time.sleep(0.5)

        logger.info("后台构建知识图谱...")
        links = vault.get_all_outgoing_links()
        elapsed = graph_ranker.build(links)
        state.update_metrics(pagerank_time=elapsed)

    # 启动后台线程
    threading.Thread(target=init_bm25, daemon=True).start()
    threading.Thread(target=init_vector, daemon=True).start()
    threading.Thread(target=init_graph, daemon=True).start()

    # 定时更新（简化版：只更新索引，不存储内容）
    def get_docs():
        docs = vault.load_all_documents()
        contents = {d.path: d.content for d in docs}
        stats = {d.path: (d.mtime, len(d.content)) for d in docs}
        state.set_bm25_ready(len(contents))
        # 更新图谱
        links = vault.get_all_outgoing_links()
        graph_ranker.build(links)
        return contents, stats

    indexer.start_background(get_docs)

    # 创建 MCP 服务器
    mcp = FastMCP(
        name="obsidian-vault-mcp",
        instructions="""Obsidian Vault 搜索和记忆服务（轻量版）。

功能：
- search: 搜索笔记（BM25 + 向量混合）
- get_backlinks: 获取反向链接
- get_tags: 获取标签
- find_orphans: 查找孤立笔记
- memory_*: 存储和获取记忆""",
    )

    # ========== 辅助函数 ==========

    def get_snippet(content: str, query: str, max_len: int = 150) -> str:
        """提取包含查询词的片段"""
        query_terms = set(query.lower().split())
        for line in content.split('\n'):
            line_lower = line.lower()
            if any(term in line_lower for term in query_terms if len(term) > 1):
                return line[:max_len] + "..." if len(line) > max_len else line
        return content[:max_len] + "..." if len(content) > max_len else content

    def read_files_for_results(paths: list[str], query: str) -> list[dict]:
        """按需读取文件并生成结果"""
        results = []
        for path in paths:
            try:
                content = vault.read_note(path)
                results.append({
                    "path": path,
                    "snippet": get_snippet(content, query),
                })
            except Exception:
                results.append({"path": path, "snippet": ""})
        return results

    # ========== 搜索 ==========

    @mcp.tool(name="search", description="搜索笔记")
    def search(
        query: Annotated[str, Field(description="搜索内容")],
        mode: Annotated[
            Literal["bm25", "semantic", "hybrid"],
            Field(default="hybrid", description="搜索模式")
        ] = "hybrid",
        limit: Annotated[int, Field(default=10, ge=1, le=50)] = 10,
    ) -> dict:
        start = time.time()

        # BM25 搜索
        if mode == "bm25":
            if not state.is_bm25_ready():
                return {"error": "索引初始化中", "results": [], "count": 0}

            # BM25 只返回路径和分数
            raw_results = bm25.search_paths(query, limit)
            paths = [p for p, _ in raw_results]

            # 按需读取文件
            results = read_files_for_results(paths, query)
            for i, (_, score) in enumerate(raw_results):
                results[i]["score"] = float(score)

            return {
                "results": results,
                "count": len(results),
                "time_ms": int((time.time() - start) * 1000),
            }

        # 语义搜索
        if mode == "semantic":
            if not state.is_vector_ready():
                return {"error": "向量索引初始化中", "results": [], "count": 0}

            results = vector.search(query, limit)
            return {
                "results": [asdict(r) for r in results],
                "count": len(results),
                "time_ms": int((time.time() - start) * 1000),
            }

        # ========== 混合搜索 ==========
        if not state.is_bm25_ready():
            return {"error": "索引初始化中", "results": [], "count": 0}

        # 1. BM25 召回（只要路径）
        bm25_results = bm25.search_paths(query, limit * 3)

        # 降级为纯 BM25
        if not state.is_vector_ready():
            paths = [p for p, _ in bm25_results[:limit]]
            results = read_files_for_results(paths, query)
            for i, (_, score) in enumerate(bm25_results[:limit]):
                results[i]["score"] = float(score)
            return {
                "results": results,
                "count": len(results),
                "time_ms": int((time.time() - start) * 1000),
                "note": "向量索引未就绪",
            }

        # 2. 向量召回
        vector_results = vector.search(query, limit * 3)

        # 3. RRF 融合
        fused = rrf_fusion(bm25_results, vector_results, k=60)

        # 4. PageRank 加权
        if graph_ranker.is_ready():
            max_pr = max((graph_ranker.get_score(p) for p, _ in fused[:30]), default=0.001) or 0.001
            weighted = []
            for path, score in fused:
                pr_boost = 1 + (graph_ranker.get_score(path) / max_pr) * 0.1
                weighted.append((path, score * pr_boost))
            fused = sorted(weighted, key=lambda x: x[1], reverse=True)

        # 5. 取 top-K，按需读取文件
        top_paths = [p for p, _ in fused[:limit]]
        results = read_files_for_results(top_paths, query)
        for i, (_, score) in enumerate(fused[:limit]):
            results[i]["score"] = float(score)

        return {
            "results": results,
            "count": len(results),
            "time_ms": int((time.time() - start) * 1000),
            "features": {"rrf": True, "pagerank": graph_ranker.is_ready()},
        }

    # ========== 链接分析 ==========

    @mcp.tool(name="get_backlinks", description="获取笔记的反向链接")
    def get_backlinks(path: Annotated[str, Field(description="笔记路径")]) -> dict:
        links = vault.get_links(path)
        return {"path": path, "backlinks": links.backlinks, "outgoing": links.outgoing}

    # ========== 标签 ==========

    @mcp.tool(name="get_tags", description="获取标签或按标签查找")
    def get_tags(
        tag: Annotated[str | None, Field(default=None)] = None,
    ) -> dict:
        if tag:
            notes = vault.find_by_tag(tag)
            return {"tag": tag, "notes": notes, "count": len(notes)}
        return {"tags": vault.get_all_tags(), "count": len(vault.get_all_tags())}

    # ========== Vault 分析 ==========

    @mcp.tool(name="find_orphans", description="查找孤立笔记")
    def find_orphans() -> dict:
        orphans = vault.find_orphans()
        return {"orphans": orphans, "count": len(orphans)}

    @mcp.tool(name="recent_notes", description="最近修改的笔记")
    def recent_notes(
        days: Annotated[int, Field(default=7, ge=1)] = 7,
        limit: Annotated[int, Field(default=20, ge=1, le=100)] = 20,
    ) -> dict:
        return {"notes": vault.get_recent_notes(days, limit)}

    # ========== Memory ==========

    @mcp.tool(name="memory_set", description="存储记忆")
    def memory_set(
        key: Annotated[str, Field(description="键")],
        value: Annotated[str, Field(description="值")],
        category: Annotated[str, Field(default="general")] = "general",
    ) -> dict:
        return asdict(memory.set(key, value, category))

    @mcp.tool(name="memory_get", description="获取记忆")
    def memory_get(key: Annotated[str, Field(description="键")]) -> dict:
        mem = memory.get(key)
        return asdict(mem) if mem else {"error": "not found", "key": key}

    @mcp.tool(name="memory_list", description="列出记忆")
    def memory_list(category: Annotated[str, Field(default="general")] = "general") -> dict:
        memories = memory.list_by_category(category)
        return {"memories": [asdict(m) for m in memories], "count": len(memories)}

    @mcp.tool(name="memory_delete", description="删除记忆")
    def memory_delete(key: Annotated[str, Field(description="键")]) -> dict:
        return {"deleted": memory.delete(key), "key": key}

    # ========== 统计 ==========

    @mcp.tool(name="stats", description="统计信息")
    def stats() -> dict:
        metrics = state.get_metrics()
        return {
            "vault": {"notes": len(vault.list_notes())},
            "index": {
                "bm25_ready": state.is_bm25_ready(),
                "vector_ready": state.is_vector_ready(),
                "graph_ready": graph_ranker.is_ready(),
                "bm25_docs": len(bm25.paths),
                "vector": vector.get_stats(),
                "graph": graph_ranker.get_stats(),
            },
            "performance": {
                "bm25_index_time_s": round(metrics["bm25_index_time"], 2),
                "vector_index_time_s": round(metrics["vector_index_time"], 2),
                "pagerank_time_ms": round(metrics["pagerank_time"], 2),
                "last_update": metrics["last_update"],
            },
            "memory": memory.get_stats(),
        }

    return mcp


def main():
    """入口"""
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Obsidian Vault MCP Server")
    parser.add_argument("--vault", type=str, default=None, help="Vault 路径")
    args = parser.parse_args()

    vault_str = args.vault or os.environ.get("OBSIDIAN_VAULT_PATH") or "."
    vault_path = Path(vault_str).resolve()

    if not vault_path.exists():
        print(f"错误: 路径不存在 {vault_path}")
        exit(1)

    obsidian_dir = vault_path / ".obsidian"
    if not obsidian_dir.exists():
        print(f"警告: {vault_path} 不是 Obsidian vault（缺少 .obsidian 目录）")

    logger.info(f"启动服务: {vault_path}")
    mcp = create_server(vault_path)
    mcp.run()


if __name__ == "__main__":
    main()
