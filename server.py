"""Obsidian Vault MCP Server - 极简版"""

import json
import logging
import threading
import time
from pathlib import Path
from dataclasses import asdict
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from config import Config, load_config
from vault import VaultReader
from search import BM25Search, VectorSearch, Indexer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== PageRank ==========

class GraphRanker:
    """PageRank 排序（持久化）"""

    def __init__(self, storage_path: Path):
        self.cache_file = storage_path / "pagerank.json"
        self._scores: dict[str, float] = {}
        self._stats = {"nodes": 0, "edges": 0}
        self._lock = threading.Lock()
        self._ready = False

    def load(self) -> bool:
        if not self.cache_file.exists():
            return False
        try:
            data = json.loads(self.cache_file.read_text())
            with self._lock:
                self._scores = data.get("scores", {})
                self._stats = {"nodes": data.get("nodes", 0), "edges": data.get("edges", 0)}
                self._ready = True
            logger.info(f"PageRank 加载完成 ({self._stats['nodes']} 节点)")
            return True
        except Exception as e:
            logger.warning(f"PageRank 加载失败: {e}")
            return False

    def build(self, links_map: dict[str, list[str]]) -> float:
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
        self._save()
        elapsed = (time.time() - start) * 1000
        logger.info(f"PageRank 构建完成 ({len(G.nodes)} 节点, {elapsed:.0f}ms)")
        return elapsed

    def _save(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps({
                "scores": self._scores,
                "nodes": self._stats["nodes"],
                "edges": self._stats["edges"],
            }))
        except Exception:
            pass

    def get_score(self, path: str) -> float:
        with self._lock:
            return self._scores.get(path, 0.0)

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready


# ========== RRF 融合 ==========

def rrf_fusion(bm25_results: list, vector_results: list, k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (path, _) in enumerate(bm25_results, 1):
        scores[path] = scores.get(path, 0) + 1 / (rank + k)
    for rank, r in enumerate(vector_results, 1):
        scores[r.path] = scores.get(r.path, 0) + 1 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ========== 索引状态 ==========

class IndexState:
    def __init__(self):
        self.bm25_ready = False
        self.vector_ready = False
        self.total_docs = 0
        self._lock = threading.Lock()

    def set_bm25_ready(self, count: int):
        with self._lock:
            self.bm25_ready = True
            self.total_docs = count

    def set_vector_ready(self):
        with self._lock:
            self.vector_ready = True

    def is_bm25_ready(self) -> bool:
        with self._lock:
            return self.bm25_ready

    def is_vector_ready(self) -> bool:
        with self._lock:
            return self.vector_ready


# ========== 服务器 ==========

def create_server(vault_path: Path, config: Config | None = None) -> FastMCP:
    config = config or load_config(vault_path)
    storage = config.storage_path

    vault = VaultReader(vault_path)
    bm25 = BM25Search(storage)
    vector = VectorSearch(storage, model_name=config.embedding_model)
    state = IndexState()
    graph = GraphRanker(storage)

    indexer = Indexer(
        storage, bm25, vector,
        interval=config.index_interval,
        vector_ready_fn=state.is_vector_ready,
    )

    # 后台初始化
    def init_bm25():
        logger.info("初始化 BM25...")
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
        state.set_bm25_ready(len(doc_contents))
        logger.info(f"BM25 就绪 ({time.time()-start:.1f}s, {len(doc_contents)} 文档)")

    def init_vector():
        while not state.is_bm25_ready():
            time.sleep(0.5)
        if vector.is_indexed():
            state.set_vector_ready()
            logger.info("向量索引从缓存加载")
            return
        docs = vault.load_all_documents()
        doc_contents = {d.path: d.content for d in docs}
        logger.info(f"构建向量索引 ({len(doc_contents)} 文档)...")
        start = time.time()
        vector.index(doc_contents)
        state.set_vector_ready()
        logger.info(f"向量索引就绪 ({time.time()-start:.1f}s)")

    def init_graph():
        if graph.load():
            return
        while not state.is_bm25_ready():
            time.sleep(0.5)
        links = vault.get_all_outgoing_links()
        graph.build(links)

    threading.Thread(target=init_bm25, daemon=True).start()
    threading.Thread(target=init_vector, daemon=True).start()
    threading.Thread(target=init_graph, daemon=True).start()

    def get_docs():
        docs = vault.load_all_documents()
        contents = {d.path: d.content for d in docs}
        stats = {d.path: (d.mtime, len(d.content)) for d in docs}
        state.set_bm25_ready(len(contents))
        links = vault.get_all_outgoing_links()
        graph.build(links)
        return contents, stats

    indexer.start_background(get_docs)

    # MCP 服务器
    mcp = FastMCP(
        name="obsidian-vault-mcp",
        instructions="Obsidian 知识库搜索服务，提供语义搜索、笔记读取、链接分析等功能。",
    )

    # ========== 工具 ==========

    @mcp.tool(
        name="vault_search",
        description="搜索 Obsidian 笔记。支持关键词(bm25)、语义(semantic)、混合(hybrid)三种模式，默认混合模式效果最佳。"
    )
    def vault_search(
        query: Annotated[str, Field(description="搜索关键词或问题")],
        mode: Annotated[
            Literal["bm25", "semantic", "hybrid"],
            Field(default="hybrid", description="搜索模式：bm25=关键词匹配，semantic=语义理解，hybrid=混合")
        ] = "hybrid",
        limit: Annotated[int, Field(default=10, ge=1, le=50, description="返回结果数量")] = 10,
    ) -> dict:
        """搜索笔记，返回匹配的笔记路径、相关度分数和内容摘要"""
        start = time.time()

        def get_snippet(content: str, q: str) -> str:
            terms = set(q.lower().split())
            for line in content.split('\n'):
                if any(t in line.lower() for t in terms if len(t) > 1):
                    return line[:150] + "..." if len(line) > 150 else line
            return content[:150] + "..." if len(content) > 150 else content

        def read_results(paths: list[str]) -> list[dict]:
            results = []
            for p in paths:
                try:
                    content = vault.read_note(p)
                    results.append({"path": p, "snippet": get_snippet(content, query)})
                except:
                    results.append({"path": p, "snippet": ""})
            return results

        if not state.is_bm25_ready():
            return {"error": "索引初始化中，请稍后重试", "results": []}

        if mode == "bm25":
            raw = bm25.search_paths(query, limit)
            results = read_results([p for p, _ in raw])
            for i, (_, score) in enumerate(raw):
                results[i]["score"] = round(score, 3)
            return {"results": results, "count": len(results), "time_ms": int((time.time()-start)*1000)}

        if mode == "semantic":
            if not state.is_vector_ready():
                return {"error": "向量索引未就绪，请使用 bm25 模式", "results": []}
            raw = vector.search(query, limit)
            return {"results": [asdict(r) for r in raw], "count": len(raw), "time_ms": int((time.time()-start)*1000)}

        # hybrid
        bm25_results = bm25.search_paths(query, limit * 3)
        if not state.is_vector_ready():
            results = read_results([p for p, _ in bm25_results[:limit]])
            for i, (_, score) in enumerate(bm25_results[:limit]):
                results[i]["score"] = round(score, 3)
            return {"results": results, "count": len(results), "time_ms": int((time.time()-start)*1000), "note": "向量索引未就绪"}

        vector_results = vector.search(query, limit * 3)
        fused = rrf_fusion(bm25_results, vector_results, k=60)

        if graph.is_ready():
            max_pr = max((graph.get_score(p) for p, _ in fused[:30]), default=0.001) or 0.001
            fused = sorted(
                [(p, s * (1 + graph.get_score(p) / max_pr * 0.1)) for p, s in fused],
                key=lambda x: x[1], reverse=True
            )

        results = read_results([p for p, _ in fused[:limit]])
        for i, (_, score) in enumerate(fused[:limit]):
            results[i]["score"] = round(score, 4)

        return {"results": results, "count": len(results), "time_ms": int((time.time()-start)*1000)}

    @mcp.tool(
        name="vault_read",
        description="读取 Obsidian 笔记的完整内容。用于获取搜索结果的详细内容。"
    )
    def vault_read(
        path: Annotated[str, Field(description="笔记路径，如 'folder/note.md'")],
    ) -> dict:
        """读取笔记全文"""
        try:
            content = vault.read_note(path)
            return {"path": path, "content": content, "length": len(content)}
        except FileNotFoundError:
            return {"error": f"笔记不存在: {path}"}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool(
        name="vault_list",
        description="列出 Obsidian 知识库中的笔记。可按目录过滤，支持查看最近修改的笔记。"
    )
    def vault_list(
        folder: Annotated[str | None, Field(default=None, description="目录路径过滤，如 'projects/'")] = None,
        recent_days: Annotated[int | None, Field(default=None, ge=1, description="只返回最近 N 天修改的笔记")] = None,
        limit: Annotated[int, Field(default=50, ge=1, le=200, description="返回数量限制")] = 50,
    ) -> dict:
        """列出笔记"""
        notes = vault.list_notes()

        if folder:
            folder = folder.rstrip('/')
            notes = [n for n in notes if n.startswith(folder + '/') or n.startswith(folder)]

        if recent_days:
            recent = vault.get_recent_notes(days=recent_days, limit=1000)
            recent_paths = {n["path"] for n in recent}
            notes = [n for n in notes if n in recent_paths]

        notes = notes[:limit]
        return {"notes": notes, "count": len(notes), "total": len(vault.list_notes())}

    @mcp.tool(
        name="vault_links",
        description="获取笔记的链接关系。返回指向该笔记的反向链接(backlinks)和该笔记引用的出链(outgoing)。"
    )
    def vault_links(
        path: Annotated[str, Field(description="笔记路径")],
    ) -> dict:
        """获取链接关系"""
        try:
            links = vault.get_links(path)
            return {
                "path": path,
                "backlinks": links.backlinks,
                "backlinks_count": len(links.backlinks),
                "outgoing": links.outgoing,
                "outgoing_count": len(links.outgoing),
            }
        except FileNotFoundError:
            return {"error": f"笔记不存在: {path}"}

    @mcp.tool(
        name="vault_tags",
        description="获取标签信息。不传参数返回所有标签统计，传入标签名返回使用该标签的笔记列表。"
    )
    def vault_tags(
        tag: Annotated[str | None, Field(default=None, description="标签名(带或不带#)，不传则返回所有标签")] = None,
    ) -> dict:
        """标签查询"""
        if tag:
            notes = vault.find_by_tag(tag)
            return {"tag": tag, "notes": notes, "count": len(notes)}
        tags = vault.get_all_tags()
        return {"tags": tags, "count": len(tags)}

    @mcp.tool(
        name="vault_related",
        description="查找与指定笔记语义相似的其他笔记。基于向量相似度，适合发现相关内容。"
    )
    def vault_related(
        path: Annotated[str, Field(description="笔记路径")],
        limit: Annotated[int, Field(default=10, ge=1, le=30, description="返回数量")] = 10,
    ) -> dict:
        """查找相似笔记"""
        if not state.is_vector_ready():
            return {"error": "向量索引未就绪"}
        try:
            content = vault.read_note(path)
            # 用笔记内容作为查询
            results = vector.search(content[:2000], limit + 1)  # +1 因为可能包含自身
            # 过滤掉自身
            filtered = [r for r in results if r.path != path][:limit]
            return {
                "path": path,
                "related": [{"path": r.path, "score": round(r.score, 3), "snippet": r.snippet} for r in filtered],
                "count": len(filtered),
            }
        except FileNotFoundError:
            return {"error": f"笔记不存在: {path}"}

    return mcp


def main():
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

    if not (vault_path / ".obsidian").exists():
        print(f"警告: {vault_path} 不是 Obsidian vault")

    logger.info(f"启动服务: {vault_path}")
    mcp = create_server(vault_path)
    mcp.run()


if __name__ == "__main__":
    main()
