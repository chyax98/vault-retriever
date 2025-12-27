"""Vault 文件读取和分析"""

import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Document:
    """文档"""
    path: str
    content: str
    mtime: float


@dataclass
class LinkInfo:
    """链接信息"""
    path: str
    backlinks: list[str]
    outgoing: list[str]


class VaultReader:
    """Vault 读取器"""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self._link_pattern = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
        self._tag_pattern = re.compile(r'(?:^|\s)#([a-zA-Z\u4e00-\u9fff][\w\u4e00-\u9fff/-]*)')

    def list_notes(self) -> list[str]:
        """列出所有笔记路径"""
        notes = []
        for f in self.vault_path.rglob("*.md"):
            rel = f.relative_to(self.vault_path)
            # 跳过隐藏目录
            if any(p.startswith('.') for p in rel.parts):
                continue
            notes.append(str(rel))
        return notes

    def read_note(self, path: str) -> str:
        """读取笔记内容"""
        file_path = self.vault_path / path
        if not file_path.exists():
            raise FileNotFoundError(f"笔记不存在: {path}")
        return file_path.read_text(encoding='utf-8')

    def load_all_documents(self) -> list[Document]:
        """加载所有文档"""
        docs = []
        for path in self.list_notes():
            try:
                file_path = self.vault_path / path
                content = file_path.read_text(encoding='utf-8')
                mtime = file_path.stat().st_mtime
                docs.append(Document(path=path, content=content, mtime=mtime))
            except Exception:
                continue
        return docs

    def get_file_hash(self, path: str) -> tuple[float, int]:
        """获取文件 mtime 和 size 用于缓存验证"""
        file_path = self.vault_path / path
        if not file_path.exists():
            return (0, 0)
        stat = file_path.stat()
        return (stat.st_mtime, stat.st_size)

    def get_links(self, path: str) -> LinkInfo:
        """获取笔记的链接信息"""
        content = self.read_note(path)

        # 提取出链
        outgoing = list(set(self._link_pattern.findall(content)))

        # 查找反向链接
        backlinks = []
        target_name = Path(path).stem
        for note_path in self.list_notes():
            if note_path == path:
                continue
            try:
                note_content = self.read_note(note_path)
                links = self._link_pattern.findall(note_content)
                if target_name in links or path in links:
                    backlinks.append(note_path)
            except Exception:
                continue

        return LinkInfo(path=path, backlinks=backlinks, outgoing=outgoing)

    def get_all_tags(self) -> dict[str, int]:
        """获取所有标签及其使用次数"""
        tags: dict[str, int] = {}
        for path in self.list_notes():
            try:
                content = self.read_note(path)
                for tag in self._tag_pattern.findall(content):
                    tag = f"#{tag}"
                    tags[tag] = tags.get(tag, 0) + 1
            except Exception:
                continue
        return tags

    def find_by_tag(self, tag: str) -> list[str]:
        """按标签查找笔记"""
        if not tag.startswith('#'):
            tag = f"#{tag}"

        results = []
        for path in self.list_notes():
            try:
                content = self.read_note(path)
                found_tags = [f"#{t}" for t in self._tag_pattern.findall(content)]
                if tag in found_tags:
                    results.append(path)
            except Exception:
                continue
        return results

    def find_orphans(self) -> list[str]:
        """查找孤立笔记（没有被任何其他笔记链接）"""
        all_notes = set(self.list_notes())
        linked_notes = set()

        for path in all_notes:
            try:
                content = self.read_note(path)
                links = self._link_pattern.findall(content)
                for link in links:
                    # 尝试匹配完整路径或文件名
                    for note in all_notes:
                        if Path(note).stem == link or note == link or note == f"{link}.md":
                            linked_notes.add(note)
            except Exception:
                continue

        return list(all_notes - linked_notes)

    def get_recent_notes(self, days: int = 7, limit: int = 20) -> list[dict]:
        """获取最近修改的笔记"""
        cutoff = datetime.now() - timedelta(days=days)
        notes = []

        for path in self.list_notes():
            try:
                file_path = self.vault_path / path
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime >= cutoff:
                    notes.append({
                        "path": path,
                        "modified": mtime.isoformat(),
                    })
            except Exception:
                continue

        # 按时间排序
        notes.sort(key=lambda x: x["modified"], reverse=True)
        return notes[:limit]

    def get_all_outgoing_links(self) -> dict[str, list[str]]:
        """获取所有笔记的出链（用于构建知识图谱）"""
        all_notes = set(self.list_notes())
        note_stems = {Path(n).stem: n for n in all_notes}  # 文件名 -> 完整路径
        links_map: dict[str, list[str]] = {}

        for path in all_notes:
            try:
                content = self.read_note(path)
                raw_links = self._link_pattern.findall(content)

                # 解析链接目标，转换为实际路径
                resolved = []
                for link in raw_links:
                    # 优先匹配完整路径，其次匹配文件名
                    if link in all_notes:
                        resolved.append(link)
                    elif f"{link}.md" in all_notes:
                        resolved.append(f"{link}.md")
                    elif link in note_stems:
                        resolved.append(note_stems[link])

                links_map[path] = list(set(resolved))
            except Exception:
                links_map[path] = []

        return links_map
