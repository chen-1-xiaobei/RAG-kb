"""把 rag-kb 的工具层暴露成 MCP server。

MCP（Model Context Protocol）是"AI 客户端 ↔ 你的能力"的标准接口：写一次 server，
Claude Desktop / Cursor / 任意 MCP 客户端都能调用你的工具。

本 server 暴露三类东西：

- **Tools**：`search_knowledge_base` / `list_tables` / `query_table`（只读）
- **Resources**：`kb://index/metadata`（索引元数据）、`kb://tables`（当前身份可见的表）
- **Prompt**：`grounded_answer`（带引用要求的回答模板）

## 两个刻意的设计决定

**一、身份不在 MCP 协议里。**
MCP 没有"当前用户"这个概念——它只负责传参。所以身份从**环境变量**读，一个 server
实例代表一个身份。这不是偷懒，而是这类系统的常见做法：多用户场景下每个用户起一个
server 实例（或由网关做 OAuth 后注入），而不是让模型传 user_id。

**二、不暴露写操作（commit_expense_claim）。**
理由**不是**"MCP 没有确认通道"——2.x 的 elicitation（`Context.elicit()`）允许工具在
调用中途向用户提问，所以那句话是错的。真实理由是三条：

1. 客户端对 elicitation 的支持不一致，写操作的正确性不能押在这上面；
2. 确认卡里的金额必须由**服务端从数据库渲染**（不能信模型给的数字），
   而 elicitation 只是一次问答；
3. 需要独立 `proposal_id` 与审计记录，事后能回答"用户确认过哪一笔"。

所以这里只暴露只读工具，写操作仍留在 HTTP 接口由确认卡驱动；将来若要暴露，
正确形态是"两阶段提案 + elicitation 二次确认"。

运行：

    python -m mcp_server.server                # stdio（给客户端拉起）
    python -m mcp_server.server --http 8001    # streamable-http（远程）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 与主应用一样，必须先修 NO_PROXY 再碰 huggingface_hub，否则启动即失败
from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

from mcp.server.mcpserver import MCPServer  # noqa: E402

from auth import Principal  # noqa: E402
from config import settings  # noqa: E402
from retrieval import IndexConfigError, index_manager, retrieve_ranked  # noqa: E402
from tools import TableRegistry, query_table_impl  # noqa: E402

DEFAULT_USER = "mcp-user"
DEFAULT_GROUPS = "public"

server = MCPServer(
    name="rag-kb",
    instructions=(
        "企业知识库检索服务。回答必须基于工具返回的资料并标注引用编号；"
        "工具没有返回内容时，明确说明无法从知识库确认，不要编造。"
    ),
)

# 表注册表进程内只加载一次
_registry: TableRegistry | None = None


def current_principal() -> Principal:
    """从环境变量取身份。

    一个 server 实例 = 一个身份。多用户场景应为每个用户起一个实例，
    或在外层网关完成认证后注入这两个变量。
    """
    groups = os.getenv("RAG_KB_GROUPS", DEFAULT_GROUPS)
    return Principal(
        user_id=os.getenv("RAG_KB_USER", DEFAULT_USER),
        groups=frozenset(item.strip() for item in groups.split(",") if item.strip()),
    )


def registry() -> TableRegistry:
    global _registry
    if _registry is None:
        _registry = TableRegistry()
        _registry.load()
    return _registry


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@server.tool()
def search_knowledge_base(question: str, top_k: int = 4) -> str:
    """在企业知识库里检索资料。

    Args:
        question: 检索问题。多轮对话中请改写成完整句子（例如把「那住宿标准呢」
            写成「出差住宿报销标准是多少」），否则检索质量会下降。
        top_k: 返回几条资料，1-8，默认 4。

    Returns:
        带引用编号的资料片段；没有命中时返回提示语。
    """
    question = (question or "").strip()
    if not question:
        return "检索问题为空。"
    top_k = max(1, min(int(top_k or 4), 8))

    try:
        hits = retrieve_ranked(question, top_k, current_principal())
    except FileNotFoundError:
        return "索引不存在：请先在服务端运行 python ingest.py。"
    except IndexConfigError as exc:
        return f"索引与当前配置不兼容：{exc}"

    if not hits:
        return "知识库没有检索到相关资料。"

    blocks = [
        f'<document citation="{index}" source="{doc.metadata.get("source", "")}" '
        f'section="{doc.metadata.get("section", "")}">\n{doc.page_content}\n</document>'
        for index, (doc, _score) in enumerate(hits, 1)
    ]
    return "\n\n".join(blocks)


@server.tool()
def list_tables() -> str:
    """列出当前身份可访问的结构化表格（CSV）及其列名。

    表格用于金额、标准、清单这类精确查询——它们是向量检索的短板。
    """
    try:
        tables = registry().list_for(current_principal())
    except Exception as exc:
        return f"表格加载失败：{exc}"
    if not tables:
        return "当前身份没有可访问的表格。"
    lines = [f"{table.key}（来源 {table.source}）列：{', '.join(table.columns)}" for table in tables]
    return "\n".join(lines)


@server.tool()
def query_table(table: str, sql: str) -> str:
    """对一张已授权的表格执行只读 SQL（仅支持 SELECT）。

    Args:
        table: 表名，必须来自 list_tables 的返回。
        sql: SELECT 语句，只能查询这一张表。

    Returns:
        markdown 表格；越权或不存在的表返回同一句提示（不泄露表是否存在）。
    """
    try:
        text, _rows = query_table_impl(
            table, sql, current_principal(), registry(), settings.table_max_rows
        )
    except Exception as exc:
        return f"查询失败：{exc}"
    return text


# ---------------------------------------------------------------------------
# Resources：客户端可以直接读的数据
# ---------------------------------------------------------------------------

@server.resource("kb://index/metadata")
def index_metadata() -> str:
    """索引元数据：chunk 数、embedding 模型、向量维度、创建时间。"""
    try:
        store = index_manager.get()
    except Exception as exc:
        return f"索引不可用：{exc}"
    import json

    return json.dumps(store.index_metadata, ensure_ascii=False, indent=2)


@server.resource("kb://tables")
def tables_resource() -> str:
    """当前身份可见的表格清单（与 list_tables 工具同一份数据）。"""
    return list_tables()


# ---------------------------------------------------------------------------
# Prompt：给客户端复用的提示词模板
# ---------------------------------------------------------------------------

@server.prompt()
def grounded_answer(question: str) -> str:
    """生成一个"要求引用来源"的回答模板。

    Args:
        question: 用户的问题。
    """
    return (
        "请仅依据工具返回的资料回答问题，并在句末用 [1][2] 标注引用编号。\n"
        "如果资料不足以回答，请直接说明无法从现有知识库确认，不要推测。\n\n"
        f"问题：{question}"
    )


def main() -> None:
    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"
        index = sys.argv.index("--http")
        port = sys.argv[index + 1] if len(sys.argv) > index + 1 else "8001"
        os.environ.setdefault("FASTMCP_PORT", port)
    server.run(transport=transport)


if __name__ == "__main__":
    main()
