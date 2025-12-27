"""Obsidian Vault MCP Server"""

import logging
import threading
import time
from pathlib import Path
from dataclasses import asdict
from typing import Annotated, Literal

import networkx as nx
from fastmcp import FastMCP
from pydantic import Field
from flashrank import Ranker, RerankRequest

from config import Config, load_config
from vault import VaultReader
from search import BM25Search, VectorSearch, Indexer
from memory import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== 知识图谱 ==========

class GraphRanker:
    """基于链接结构的 PageRank 排序"""

    def __init__(self):
        self._graph: nx.DiGraph | None = None
        self._pagerank: dict[str, float] = {}
        self._lock = threading.Lock()

    def build(self, links_map: dict[str, list[str]]):
        """构建图谱并计算 PageRank"""
        start = time.time()
        G = nx.DiGraph()

        for source, targets in links_map.items():
            G.add_node(source)
            for target in targets:
                G.add_edge(source, target)

        # 计算 PageRank
        try:
            scores = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            scores = {}

        with self._lock:
            self._graph = G
            self._pagerank = scores

        elapsed = (time.time() - start) * 1000
        logger.info(f"PageRank 计算完成 ({len(G.nodes)} 节点, {len(G.edges)} 边, {elapsed:.0f}ms)")
        return elapsed

    def get_score(self, path: str) -> float:
        """获取节点的 PageRank 分数"""
        with self._lock:
            return self._pagerank.get(path, 0.0)

    def get_stats(self) -> dict:
        """获取图谱统计"""
        with self._lock:
            if self._graph is None:
                return {"nodes": 0, "edges": 0}
            return {
                "nodes": len(self._graph.nodes),
                "edges": len(self._graph.edges),
            }


# ========== RRF 融合 ==========

def rrf_fusion(
    bm25_results: list,
    vector_results: list,
    k: int = 60
) -> list[tuple[str, float, str]]:
    """
    Reciprocal Rank Fusion 融合算法
    score = sum(1 / (rank + k))
    """
    scores: dict[str, float] = {}
    snippets: dict[str, str] = {}

    # BM25 排名贡献
    for rank, r in enumerate(bm25_results, 1):
        scores[r.path] = scores.get(r.path, 0) + 1 / (rank + k)
        if r.path not in snippets:
            snippets[r.path] = r.snippet

    # 向量排名贡献
    for rank, r in enumerate(vector_results, 1):
        scores[r.path] = scores.get(r.path, 0) + 1 / (rank + k)
        if r.path not in snippets:
            snippets[r.path] = r.snippet

    # 按分数排序
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(path, score, snippets[path]) for path, score in sorted_items]


class IndexState:
    """索引状态管理"""
    def __init__(self):
        self.bm25_ready = False
        self.vector_ready = False
        self.reranker_ready = False
        self.graph_ready = False
        self.doc_contents: dict[str, str] = {}
        self._lock = threading.Lock()
        # 性能指标
        self.metrics = {
            "bm25_index_time": 0.0,
            "vector_index_time": 0.0,
            "reranker_load_time": 0.0,
            "pagerank_time": 0.0,
            "total_docs": 0,
            "last_update": "",
        }

    def set_bm25_ready(self, contents: dict[str, str]):
        with self._lock:
            self.doc_contents = contents
            self.bm25_ready = True
            self.metrics["total_docs"] = len(contents)

    def set_vector_ready(self):
        with self._lock:
            self.vector_ready = True

    def set_reranker_ready(self):
        with self._lock:
            self.reranker_ready = True

    def set_graph_ready(self):
        with self._lock:
            self.graph_ready = True

    def is_bm25_ready(self) -> bool:
        with self._lock:
            return self.bm25_ready

    def is_vector_ready(self) -> bool:
        with self._lock:
            return self.vector_ready

    def is_reranker_ready(self) -> bool:
        with self._lock:
            return self.reranker_ready

    def is_graph_ready(self) -> bool:
        with self._lock:
            return self.graph_ready

    def get_contents(self) -> dict[str, str]:
        with self._lock:
            return self.doc_contents.copy()

    def update_metrics(self, **kwargs):
        with self._lock:
            self.metrics.update(kwargs)

    def get_metrics(self) -> dict:
        with self._lock:
            return self.metrics.copy()


def create_server(vault_path: Path, config: Config | None = None) -> FastMCP:
    """创建 MCP 服务器"""

    config = config or load_config(vault_path)
    storage = config.storage_path

    # 初始化组件
    vault = VaultReader(vault_path)
    bm25 = BM25Search(storage)  # 支持持久化和 mmap
    vector = VectorSearch(storage, model_name=config.embedding_model)
    memory = MemoryStore(storage)
    state = IndexState()
    graph_ranker = GraphRanker()
    reranker: Ranker | None = None  # 延迟加载

    indexer = Indexer(
        storage, bm25, vector,
        interval=config.index_interval,
        vector_ready_fn=state.is_vector_ready,
    )

    # 后台初始化 BM25（支持缓存加载）
    def init_bm25():
        logger.info("后台初始化 BM25...")
        start = time.time()

        docs = vault.load_all_documents()
        doc_contents = {d.path: d.content for d in docs}
        file_stats = {d.path: (d.mtime, len(d.content)) for d in docs}

        # 尝试从缓存加载（使用 mmap 节省内存）
        if bm25.load_index(use_mmap=True):
            # 检查增量更新
            result = indexer.index_incremental(doc_contents, file_stats)
            if result["status"] == "updated":
                # 有变化，重建索引
                bm25.index(doc_contents)
        else:
            # 没有缓存，全量索引
            bm25.index(doc_contents)

        bm25_time = time.time() - start
        state.set_bm25_ready(doc_contents)
        state.update_metrics(bm25_index_time=bm25_time)
        logger.info(f"BM25 就绪 ({bm25_time:.2f}s, {len(doc_contents)} 文档)")

    # 后台初始化向量索引（慢速，可能几分钟）
    def init_vector():
        # 等 BM25 先完成
        while not state.is_bm25_ready():
            time.sleep(0.5)

        doc_contents = state.get_contents()
        if not doc_contents:
            return

        # 检查是否已有向量索引
        if vector.is_indexed():
            state.set_vector_ready()
            state.update_metrics(
                vector_index_time=0,
                last_update=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            logger.info("向量索引从缓存加载")
            return

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

    # 后台初始化 Reranker（轻量，几秒）
    def init_reranker():
        nonlocal reranker
        logger.info("后台加载 Reranker...")
        start = time.time()
        try:
            reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            load_time = time.time() - start
            state.set_reranker_ready()
            state.update_metrics(reranker_load_time=load_time)
            logger.info(f"Reranker 就绪 ({load_time:.2f}s)")
        except Exception as e:
            logger.warning(f"Reranker 加载失败: {e}")

    # 后台初始化知识图谱（快速，<1s）
    def init_graph():
        # 等 BM25 先完成
        while not state.is_bm25_ready():
            time.sleep(0.5)

        logger.info("后台构建知识图谱...")
        links = vault.get_all_outgoing_links()
        elapsed = graph_ranker.build(links)
        state.set_graph_ready()
        state.update_metrics(pagerank_time=elapsed)

    # 启动后台线程
    threading.Thread(target=init_bm25, daemon=True).start()
    threading.Thread(target=init_vector, daemon=True).start()
    threading.Thread(target=init_reranker, daemon=True).start()
    threading.Thread(target=init_graph, daemon=True).start()

    # 启动定时更新
    def get_docs():
        docs = vault.load_all_documents()
        contents = {d.path: d.content for d in docs}
        stats = {d.path: (d.mtime, len(d.content)) for d in docs}
        state.set_bm25_ready(contents)
        # 同时更新图谱
        links = vault.get_all_outgoing_links()
        graph_ranker.build(links)
        return contents, stats

    indexer.start_background(get_docs)

    # 创建 MCP 服务器
    mcp = FastMCP(
        name="obsidian-vault-mcp",
        instructions="""Obsidian Vault 搜索和记忆服务。

功能：
- search: 搜索笔记（支持关键词/语义/混合模式）
- get_backlinks: 获取反向链接
- get_tags: 获取标签
- find_orphans: 查找孤立笔记
- memory_*: 存储和获取记忆""",
    )

    # ========== 搜索（核心） ==========

    @mcp.tool(name="search", description="搜索笔记")
    def search(
        query: Annotated[str, Field(description="搜索内容")],
        mode: Annotated[
            Literal["bm25", "semantic", "hybrid"],
            Field(default="hybrid", description="搜索模式")
        ] = "hybrid",
        limit: Annotated[int, Field(default=10, ge=1, le=50)] = 10,
        use_rerank: Annotated[bool, Field(default=True, description="是否使用 Reranker")] = True,
    ) -> dict:
        start = time.time()

        # BM25 搜索
        if mode == "bm25":
            if not state.is_bm25_ready():
                return {"error": "BM25 索引初始化中", "results": [], "count": 0}
            doc_contents = state.get_contents()
            results = bm25.search(query, doc_contents, limit)
            return {
                "results": [asdict(r) for r in results],
                "count": len(results),
                "time_ms": int((time.time() - start) * 1000),
            }

        # 语义搜索
        if mode == "semantic":
            if not state.is_vector_ready():
                return {"error": "向量索引初始化中，请使用 bm25 模式", "results": [], "count": 0}
            results = vector.search(query, limit)
            return {
                "results": [asdict(r) for r in results],
                "count": len(results),
                "time_ms": int((time.time() - start) * 1000),
            }

        # ========== 混合搜索（优化版） ==========
        if not state.is_bm25_ready():
            return {"error": "索引初始化中", "results": [], "count": 0}

        # 获取文档内容（用于 BM25 snippet）
        doc_contents = state.get_contents()

        # 1. 多路召回
        bm25_results = bm25.search(query, doc_contents, limit * 5)  # 扩大召回量

        # 如果向量索引未就绪，降级为纯 BM25
        if not state.is_vector_ready():
            return {
                "results": [asdict(r) for r in bm25_results[:limit]],
                "count": min(len(bm25_results), limit),
                "time_ms": int((time.time() - start) * 1000),
                "note": "向量索引未就绪，已降级为 BM25",
            }

        vector_results = vector.search(query, limit * 5)

        # 2. RRF 融合（替代简单加权）
        fused = rrf_fusion(bm25_results, vector_results, k=60)

        # 3. PageRank 加权（如果图谱就绪）
        if state.is_graph_ready():
            # 对 RRF 分数应用 PageRank 权重
            max_pr = max((graph_ranker.get_score(p) for p, _, _ in fused[:50]), default=0.001) or 0.001
            weighted = []
            for path, score, snippet in fused:
                pr_score = graph_ranker.get_score(path)
                # PageRank 贡献 10% 权重
                pr_boost = 1 + (pr_score / max_pr) * 0.1
                weighted.append((path, score * pr_boost, snippet))
            fused = sorted(weighted, key=lambda x: x[1], reverse=True)

        # 4. Reranker 精排（如果可用且启用）
        if use_rerank and state.is_reranker_ready() and reranker and len(fused) > 0:
            # 取 Top 50 进行 rerank
            candidates = fused[:50]
            passages = [{"id": p, "text": s} for p, _, s in candidates]

            try:
                rerank_request = RerankRequest(query=query, passages=passages)
                reranked = reranker.rerank(rerank_request)

                # 重新排序
                path_to_snippet = {p: s for p, _, s in candidates}
                results = [
                    {
                        "path": r["id"],
                        "score": float(r["score"]),
                        "snippet": path_to_snippet.get(r["id"], ""),
                    }
                    for r in reranked[:limit]
                ]
            except Exception as e:
                logger.warning(f"Rerank 失败: {e}")
                results = [
                    {"path": p, "score": s, "snippet": sn}
                    for p, s, sn in fused[:limit]
                ]
        else:
            results = [
                {"path": p, "score": s, "snippet": sn}
                for p, s, sn in fused[:limit]
            ]

        return {
            "results": results,
            "count": len(results),
            "time_ms": int((time.time() - start) * 1000),
            "features": {
                "rrf": True,
                "rerank": use_rerank and state.is_reranker_ready(),
                "pagerank": state.is_graph_ready(),
            },
        }

    # ========== 链接分析 ==========

    @mcp.tool(name="get_backlinks", description="获取笔记的反向链接")
    def get_backlinks(
        path: Annotated[str, Field(description="笔记路径")]
    ) -> dict:
        links = vault.get_links(path)
        return {
            "path": path,
            "backlinks": links.backlinks,
            "outgoing": links.outgoing,
        }

    # ========== 标签 ==========

    @mcp.tool(name="get_tags", description="获取标签或按标签查找")
    def get_tags(
        tag: Annotated[str | None, Field(default=None, description="指定标签则返回该标签的笔记")] = None,
    ) -> dict:
        if tag:
            notes = vault.find_by_tag(tag)
            return {"tag": tag, "notes": notes, "count": len(notes)}
        else:
            tags = vault.get_all_tags()
            return {"tags": tags, "count": len(tags)}

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
        notes = vault.get_recent_notes(days, limit)
        return {"notes": notes, "count": len(notes)}

    # ========== Memory ==========

    @mcp.tool(name="memory_set", description="存储记忆")
    def memory_set(
        key: Annotated[str, Field(description="键")],
        value: Annotated[str, Field(description="值")],
        category: Annotated[str, Field(default="general")] = "general",
    ) -> dict:
        mem = memory.set(key, value, category)
        return asdict(mem)

    @mcp.tool(name="memory_get", description="获取记忆")
    def memory_get(key: Annotated[str, Field(description="键")]) -> dict:
        mem = memory.get(key)
        if mem:
            return asdict(mem)
        return {"error": "not found", "key": key}

    @mcp.tool(name="memory_list", description="列出记忆")
    def memory_list(
        category: Annotated[str, Field(default="general")] = "general",
    ) -> dict:
        memories = memory.list_by_category(category)
        return {"memories": [asdict(m) for m in memories], "count": len(memories)}

    @mcp.tool(name="memory_delete", description="删除记忆")
    def memory_delete(key: Annotated[str, Field(description="键")]) -> dict:
        deleted = memory.delete(key)
        return {"deleted": deleted, "key": key}

    # ========== 统计 ==========

    @mcp.tool(name="stats", description="统计信息")
    def stats() -> dict:
        metrics = state.get_metrics()
        return {
            "vault": {"notes": len(vault.list_notes())},
            "index": {
                "bm25_ready": state.is_bm25_ready(),
                "vector_ready": state.is_vector_ready(),
                "reranker_ready": state.is_reranker_ready(),
                "graph_ready": state.is_graph_ready(),
                "bm25_docs": len(bm25.paths),
                "vector": vector.get_stats(),
                "graph": graph_ranker.get_stats(),
            },
            "performance": {
                "bm25_index_time_s": round(metrics["bm25_index_time"], 2),
                "vector_index_time_s": round(metrics["vector_index_time"], 2),
                "reranker_load_time_s": round(metrics["reranker_load_time"], 2),
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

    # 优先级：命令行参数 > 环境变量 > 当前目录
    vault_str = args.vault or os.environ.get("OBSIDIAN_VAULT_PATH") or "."
    vault_path = Path(vault_str).resolve()

    if not vault_path.exists():
        print(f"错误: 路径不存在 {vault_path}")
        exit(1)

    # 验证是否是 Obsidian vault
    obsidian_dir = vault_path / ".obsidian"
    if not obsidian_dir.exists():
        print(f"警告: {vault_path} 不是 Obsidian vault（缺少 .obsidian 目录）")

    logger.info(f"启动服务: {vault_path}")
    mcp = create_server(vault_path)
    mcp.run()


if __name__ == "__main__":
    main()
