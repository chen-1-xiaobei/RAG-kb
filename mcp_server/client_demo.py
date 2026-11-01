"""一个最小 MCP 客户端：连上自己的 server，把协议层每一步打出来。

    python -m mcp_server.client_demo

它做的事和 Claude Desktop 完全一样：启动 server 进程 → 握手 → 读工具清单 →
调用工具 → 读资源 → 取提示词模板。

看懂这个文件，你就知道 MCP 的"协议"到底是什么——它不过是几次 JSON-RPC 往返。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

from mcp import Client, StdioServerParameters  # noqa: E402


def server_params() -> StdioServerParameters:
    """告诉客户端"怎么启动 server"。

    command + args 就是启动命令；env 决定这个 server 实例代表哪个身份。
    """
    return StdioServerParameters(
        command=sys.executable,                       # 用当前这个 python
        args=["-m", "mcp_server.server"],             # 启动我们的 server
        cwd=str(ROOT),
        env={
            **{k: v for k, v in os.environ.items() if k in {"PATH", "SYSTEMROOT", "TEMP", "TMP"}},
            "RAG_KB_USER": "mcp-client-demo",
            "RAG_KB_GROUPS": "财务",
            "PYTHONIOENCODING": "utf-8",
        },
    )


async def main() -> int:
    async with Client(server_params(), raise_exceptions=False) as client:
        # ① 握手之后，客户端能拿到 server 的自我介绍
        info = client.server_info
        print("=" * 74)
        print("① 握手完成：server 自我介绍")
        print("=" * 74)
        print(f"  name: {getattr(info, 'name', info)}")
        print(f"  protocol_version: {client.protocol_version}")
        print(f"  instructions: {client.instructions}")

        # ② 读工具清单——模型看到的就是这份
        print("\n" + "=" * 74)
        print("② tools/list：客户端看到的工具清单")
        print("=" * 74)
        tools = await client.list_tools()
        for tool in tools.tools:
            params = list(tool.input_schema.get("properties", {}))
            required = tool.input_schema.get("required", [])
            print(f"\n  · {tool.name}")
            print(f"      说明: {tool.description.splitlines()[0]}")
            print(f"      参数: {params}  必填: {required}")

        # ③ 调用检索工具
        print("\n" + "=" * 74)
        print("③ tools/call：调用 search_knowledge_base")
        print("=" * 74)
        result = await client.call_tool(
            "search_knowledge_base",
            {"question": "报销单要在费用发生后多久内提交", "top_k": 2},
        )
        for block in result.content:
            text = getattr(block, "text", str(block))
            print(text[:500])

        # ④ 调用表格工具（两次：合法查询 + 越权查询）
        print("\n" + "=" * 74)
        print("④ tools/call：query_table（合法查询）")
        print("=" * 74)
        result = await client.call_tool(
            "query_table",
            {
                "table": "财务/报销标准",
                "sql": 'SELECT 项目, 标准 FROM "财务/报销标准" WHERE 项目 LIKE "%住宿%"',
            },
        )
        for block in result.content:
            print(getattr(block, "text", str(block)))

        print("\n" + "=" * 74)
        print("④b tools/call：query_table（写操作会被拒绝）")
        print("=" * 74)
        result = await client.call_tool(
            "query_table", {"table": "财务/报销标准", "sql": 'DELETE FROM "财务/报销标准"'}
        )
        for block in result.content:
            print(getattr(block, "text", str(block)))

        # ⑤ 读资源（不需要模型决定，客户端直接读）
        print("\n" + "=" * 74)
        print("⑤ resources/read：kb://index/metadata")
        print("=" * 74)
        resource = await client.read_resource("kb://index/metadata")
        for block in resource.contents:
            text = getattr(block, "text", "")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                print(text)
            else:
                print(f"  chunk 数: {payload.get('record_count')}")
                print(f"  embedding: {payload.get('embedding_model')}")
                print(f"  维度: {payload.get('vector_dimension')}")

        # ⑥ 取提示词模板
        print("\n" + "=" * 74)
        print("⑥ prompts/get：grounded_answer")
        print("=" * 74)
        prompt = await client.get_prompt("grounded_answer", {"question": "报销多久内提交"})
        for message in prompt.messages:
            print(getattr(message.content, "text", message.content))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
