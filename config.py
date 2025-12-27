"""配置管理"""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Config:
    """服务器配置"""
    vault_path: Path
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    index_interval: int = 600  # 索引更新间隔（秒），大型知识库建议 10-15 分钟

    @property
    def storage_path(self) -> Path:
        """数据存储路径"""
        path = self.vault_path / ".obsidian" / "plugins" / "vault-mcp-data"
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(vault_path: Path) -> Config:
    """加载配置"""
    config_file = vault_path / ".obsidian" / "vault-mcp.json"

    if config_file.exists():
        data = json.loads(config_file.read_text())
        return Config(
            vault_path=vault_path,
            embedding_model=data.get("embedding_model", Config.embedding_model),
            index_interval=data.get("index_interval", Config.index_interval),
        )

    return Config(vault_path=vault_path)
