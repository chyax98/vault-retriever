# Obsidian Vault MCP

[![PyPI](https://img.shields.io/pypi/v/obsidian-vault-mcp)](https://pypi.org/project/obsidian-vault-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/obsidian-vault-mcp)](https://pypi.org/project/obsidian-vault-mcp/)
[![License](https://img.shields.io/github/license/chyax98/obsidian-mcp)](https://github.com/chyax98/obsidian-mcp/blob/main/LICENSE)

Obsidian 知识库的 MCP 服务，提供语义搜索和知识图谱分析能力。

## 特性

- **混合搜索**：BM25 关键词 + 向量语义 + RRF 融合 + PageRank 加权
- **轻量高效**：mmap 索引加载，内存占用 ~300MB
- **后台索引**：启动不阻塞，增量更新
- **零配置**：开箱即用

## 安装

```bash
uv tool install obsidian-vault-mcp --python 3.12
```

## 工具列表

| 工具 | 说明 |
|------|------|
| `vault_search` | 搜索笔记，支持 bm25/semantic/hybrid 三种模式 |
| `vault_read` | 读取笔记完整内容 |
| `vault_list` | 列出笔记，支持目录过滤和最近修改筛选 |
| `vault_links` | 获取笔记的反向链接和出链关系 |
| `vault_tags` | 获取标签统计或按标签查找笔记 |
| `vault_related` | 查找语义相似的相关笔记 |

### 工具详情

**vault_search** - 搜索笔记
- `query`: 搜索关键词或问题
- `mode`: 搜索模式（bm25=关键词匹配，semantic=语义理解，hybrid=混合，默认 hybrid）
- `limit`: 返回数量（1-50，默认 10）

**vault_read** - 读取笔记
- `path`: 笔记路径，如 `folder/note.md`

**vault_list** - 列出笔记
- `folder`: 目录过滤，如 `projects/`
- `recent_days`: 只返回最近 N 天修改的笔记
- `limit`: 返回数量（1-200，默认 50）

**vault_links** - 链接关系
- `path`: 笔记路径
- 返回：backlinks（反向链接）和 outgoing（出链）

**vault_tags** - 标签查询
- `tag`: 标签名（可选），不传则返回所有标签统计

**vault_related** - 相关笔记
- `path`: 笔记路径
- `limit`: 返回数量（1-30，默认 10）

## 使用

### Vault 路径配置

优先级：`--vault` 参数 > `OBSIDIAN_VAULT_PATH` 环境变量 > 当前目录

```bash
# 命令行参数
obsidian-vault-mcp --vault /path/to/vault

# 环境变量
export OBSIDIAN_VAULT_PATH=/path/to/vault
obsidian-vault-mcp
```

### Claude Code 配置

**命令行添加：**

```bash
claude mcp add obsidian-vault \
  -e OBSIDIAN_VAULT_PATH=/path/to/vault \
  -- obsidian-vault-mcp
```

**项目 `.mcp.json`（放在 vault 目录下）：**

```json
{
  "mcpServers": {
    "obsidian-vault": {
      "type": "stdio",
      "command": "obsidian-vault-mcp",
      "env": {
        "OBSIDIAN_VAULT_PATH": "${PWD}"
      }
    }
  }
}
```

## 索引机制

- 启动时后台初始化（不阻塞主线程）
- BM25 索引使用 mmap 加载，内存高效
- 向量索引使用 LanceDB 持久化
- PageRank 图谱缓存到磁盘
- 每 10 分钟检查文件变动，增量更新

## 配置

在 vault 目录下创建 `.obsidian/vault-mcp.json`：

```json
{
  "embedding_model": "BAAI/bge-small-zh-v1.5",
  "index_interval": 600
}
```

- `embedding_model`: 向量模型（默认 bge-small-zh-v1.5）
- `index_interval`: 索引更新间隔秒数（默认 600）

## 开发

```bash
git clone https://github.com/chyax98/obsidian-mcp.git
cd obsidian-mcp
uv sync
uv run obsidian-vault-mcp --vault /path/to/vault
```

## License

MIT
