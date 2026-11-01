# MCP server：把工具层暴露成标准协议

MCP（Model Context Protocol）是"AI 客户端 ↔ 你的能力"的标准接口。写一次 server，
Claude Desktop / Cursor / 任何 MCP 客户端都能调用这里的工具，不需要为每个客户端写适配。

## 装了它之后有什么变化

| 之前 | 现在 |
| --- | --- |
| 工具只在 `/agent/ask` 里给**一个**模型用 | 同一个工具层可被任意 MCP 客户端复用 |
| 每个客户端要自己写调用代码 | 客户端读 `tools/list` 后自动知道怎么调 |

**关键点：这不是把代码又写了一遍。** 工具实现仍然是 `tools.py` 里那套
（权限闭包、只读校验、同文案防泄露），`mcp_server/server.py` 只做协议适配。

## 运行

```bash
pip install "mcp>=2.0"          # 可选依赖，未装时主测试会跳过相关用例

# stdio（推荐，由客户端拉起进程）
python -m mcp_server.server

# streamable-http（远程/容器内）
python -m mcp_server.server --http 8001
```

## 接到客户端

**Claude Desktop**（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "rag-kb": {
      "command": "C:/path/to/rag-kb/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/path/to/rag-kb",
      "env": { "RAG_KB_USER": "alice", "RAG_KB_GROUPS": "财务" }
    }
  }
}
```

**Cursor / 其他客户端**：同样是 `command + args + env` 三件事。

## 暴露了什么

| 类型 | 名称 | 说明 |
| --- | --- | --- |
| Tool | `search_knowledge_base(question, top_k=4)` | 两段式检索，返回带 citation 编号的片段 |
| Tool | `list_tables()` | 当前身份可见的表格与列名 |
| Tool | `query_table(table, sql)` | 只读 SELECT（表名要带引号写完整 key） |
| Resource | `kb://index/metadata` | 索引元数据（chunk 数、embedding 模型、维度） |
| Resource | `kb://tables` | 与 `list_tables` 同源 |
| Prompt | `grounded_answer(question)` | 要求"仅依据资料 + 标注引用"的模板 |

实测（`RAG_KB_GROUPS=财务`）：

```
query_table('财务/报销标准', 'SELECT 项目, 标准 FROM "财务/报销标准" WHERE 项目 LIKE "%住宿%"')
→ 表 财务/报销标准 命中 2 行：
  | 住宿-一线城市 | 500元/晚 |
  | 住宿-其他城市 | 350元/晚 |

query_table('人事/工资表', ...)     → 表不存在。          （与无权同文案）
query_table(..., 'DELETE FROM ...')  → 仅支持只读 SELECT 查询该表。
```

## 三个设计决定

### ① 不暴露写操作

`commit_expense_claim` **刻意没有暴露**。报销的提交依赖"用户点确认"这个交互，而 MCP 的
tool call 是模型单方面发起的，没有确认通道。暴露它等于让模型可以自己把申请递上去。

写操作仍留在 HTTP 接口里，由确认卡驱动——**协议适配不能把原有的安全属性弄丢**。

### ② 身份来自环境变量，不是工具参数

MCP 协议里没有"当前用户"这个概念。所以身份从 `RAG_KB_USER` / `RAG_KB_GROUPS` 读，
**一个 server 实例代表一个身份**。

这不是偷懒，而是这类部署的常见形态：多用户场景下每个用户起一个实例，
或由外层网关完成认证后注入这两个变量。**绝不能让模型通过参数传 user_id**——
那等于把权限判定交给模型。

### ③ 只读三层照旧生效

`query_table` 仍然走 `query_table_impl` 的白名单 + SELECT-only + sqlite authorizer。
上面那三个实测就是这三层在起作用（第二个是白名单，第三个是语法校验）。

## 边界

- **本 server 不是新的鉴权层**：它复用主应用的 `Principal` 语义，但身份由部署者通过
  环境变量给定。如果要在多租户环境暴露 HTTP 传输，需要在网关层做认证。
- **索引与模型的加载成本**：首次调用会加载 embedding 与 reranker（数秒到数十秒），
  客户端应预期首调慢。
- **测试**：`tests/test_mcp_server.py` 验证注册与 schema，未安装 `mcp` 时自动跳过。
