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

## 一个最小客户端：看清楚协议到底传了什么

```bash
python -m mcp_server.client_demo
```

它做的事和 Claude Desktop 完全一样，只是把每一步打印出来：

```
① 握手完成：server 自我介绍
  name: rag-kb
  protocol_version: 2026-07-28
  instructions: 企业知识库检索服务。回答必须基于工具返回的资料并标注引用编号…

② tools/list：客户端看到的工具清单
  · search_knowledge_base   参数: ['question', 'top_k']  必填: ['question']
  · list_tables             参数: []
  · query_table             参数: ['table', 'sql']       必填: ['table', 'sql']

③ tools/call：search_knowledge_base
  <document citation="1" source="报销流程.md" section="报销步骤">
  员工在费用发生后三十天内，在 OA 系统填写报销单…
  </document>

④ tools/call：query_table（合法查询）
  表 财务/报销标准 命中 2 行：| 住宿-一线城市 | 500元/晚 |

④b tools/call：query_table（DELETE 被拒）
  仅支持只读 SELECT 查询该表。

⑤ resources/read：kb://index/metadata
  chunk 数: 126 / embedding: BAAI/bge-small-zh-v1.5 / 维度: 512

⑥ prompts/get：grounded_answer
  请仅依据工具返回的资料回答问题，并在句末用 [1][2] 标注引用编号…
```

**关键认识：MCP 不过就是几次 JSON-RPC 往返。** 没有魔法，也没有"AI 自动发现"——
客户端拿到 `tools/list` 的 schema 交给模型，模型决定调哪个，客户端再发 `tools/call`。

客户端的三个要素：

```python
StdioServerParameters(
    command=sys.executable,           # 用什么程序启动
    args=["-m", "mcp_server.server"], # 启动参数
    cwd=str(ROOT),                    # 工作目录
    env={"RAG_KB_USER": "...", "RAG_KB_GROUPS": "财务"},   # ← 身份在这里
)
```

**注意 `env` 那一行**：客户端通过环境变量决定"这个 server 实例代表谁"。这就是
"一个 server 实例 = 一个身份"在客户端侧的样子。

## LLM 驱动的客户端：让模型自己决定调什么

```bash
python -m mcp_server.agent_client "一线城市出差住一晚，最多能报多少住宿费？"
```

和上面那个 demo 的区别：demo 是**人**写死调哪个工具，这个是**模型**决定。
实测轨迹（三轮，模型全程自主）：

```
工具清单已获取：['search_knowledge_base', 'list_tables', 'query_table']

--- 第 1 轮：模型决定调用 2 个工具（并行）
    → search_knowledge_base({'question': '一线城市出差住宿费报销标准上限是多少'})
    → list_tables({})
--- 第 2 轮：模型决定调用 1 个工具
    → query_table({'table': '财务/报销标准',
                   'sql': 'SELECT 项目, 标准, 说明 FROM "财务/报销标准" WHERE 项目 LIKE \'%住宿%\' ...'})
--- 第 3 轮：模型决定作答
    → 一线城市上限 500 元/晚 [1]；其他城市 350 元/晚 [1]；超 5000 元需双签 [2]
    → "超出部分能否报销，知识库里没有查到相关规定，建议向财务确认"
```

三个值得注意的细节：

1. **SQL 是模型自己写的，而且写对了带引号的表名**——因为上一轮 `list_tables` 告诉了它
   `财务/报销标准`。这就是"先 list 再 query"这个设计的意义。
2. **它并行调了两个工具**（检索 + 列清单），模型自己判断这两件事可以同时做。
3. **它承认了知识库的边界**（"超出部分……没有查到相关规定"），而不是编一个答案。

## 这个循环有多短

```python
messages = [SystemMessage(系统提示), HumanMessage(问题)]
for _ in range(MAX_ROUNDS):
    response = await bound.ainvoke(messages)      # 模型决定
    if not response.tool_calls:                   # 不调工具 -> 收工
        return response.content
    for call in response.tool_calls:              # 调工具
        result = await mcp.call_tool(call["name"], call["args"])
        messages.append(ToolMessage(content=结果, tool_call_id=call["id"]))
```

**LangGraph 做的就是这件事的工程化版本**——加了轮数上限、证据门槛、状态持久化、
可恢复。手写一遍才看得清"框架帮你做了什么"。

**MCP 工具转 function calling 几乎零转换**：MCP 的 `input_schema` 就是 JSON Schema，
OpenAI 风格的工具定义要的也是 JSON Schema，只需把字段名换成 `parameters`。

## 三个设计决定

### ① 不暴露写操作（理由要写准确）

`commit_expense_claim` **刻意没有暴露**。但理由**不是**"MCP 没有确认通道"——那句话是错的：

> MCP 2.x 有 **elicitation**：工具内部可以通过 `Context.elicit()` 在调用中途向用户提问。
> 所以"服务端无法向用户确认"这个说法不成立。

真实理由有三条：

1. **客户端支持不一致。** elicitation 要求客户端声明能力，Claude Desktop 支持，
   很多客户端不支持——把写操作的正确性押在"客户端支不支持"上是不可接受的。
2. **我们要的确认卡不是一次问答。** 卡片里的金额必须**由服务端从数据库渲染**
   （模型说的金额不可信），而 elicitation 只是把一段文本问出去、拿一个回答回来。
3. **要可追溯。** 提案有独立的 `proposal_id`、`proposals` 表与 `claim_audit` 表，
   事后能回答"用户到底确认过哪一笔、什么时候确认的"。单次调用内的确认做不到这一点。

写操作仍留在 HTTP 接口里，由确认卡驱动——**协议适配不能把原有的安全属性弄丢**。
将来若要暴露，正确形态是**两阶段提案 + elicitation 二次确认**，而不是单次调用直接提交。

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
