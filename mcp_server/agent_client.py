"""一个真正的 LLM 驱动 MCP 客户端：让模型自己决定调哪个工具。

    python -m mcp_server.agent_client "一线城市住一晚能报多少"

和 `client_demo.py` 的区别：
- demo 是**人**决定调哪个工具（写死在代码里）
- 这个是**模型**决定：它只拿到 `tools/list` 的 schema，自己决定调什么、传什么参数

## 这个文件想说明三件事

1. **MCP 的 input_schema 就是 JSON Schema**，而 function calling 要的也是 JSON Schema——
   所以"把 MCP 工具交给模型"几乎是零转换，改个字段名就行。
2. **Agent 循环本身很短**：模型 → 有 tool_calls 就执行 → 结果塞回去 → 再问模型。
   你项目里 LangGraph 做的就是这件事的工程化版本（加了轮数上限、证据门槛、状态持久化）。
3. **工具返回的字符串就是模型的全部事实来源**。没有证据时它会说"查不到"，
   因为工具返回的就是"知识库没有检索到相关资料"。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

from langchain_core.messages import (  # noqa: E402
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_deepseek import ChatDeepSeek  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402

from config import settings  # noqa: E402

MAX_ROUNDS = 4

SYSTEM_PROMPT = """你是企业知识库助手。涉及公司制度、流程、标准、金额时，必须调用工具获取资料，
不要凭常识回答。

拿到资料后：用 [1][2] 标注引用编号；如果工具返回"没有检索到"，直接说明无法从知识库确认。"""


def server_params(groups: str) -> StdioServerParameters:
    """启动 server 进程，并注入身份。"""
    keep = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "PYTHONIOENCODING"}
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(ROOT),
        env={
            **{key: value for key, value in os.environ.items() if key in keep},
            "RAG_KB_USER": "agent-client",
            "RAG_KB_GROUPS": groups,
            "PYTHONIOENCODING": "utf-8",
        },
    )


def to_function_schemas(tools) -> list[dict]:
    """MCP 工具 → function calling 的 schema。

    几乎是直接搬：MCP 的 `input_schema` 就是 JSON Schema，OpenAI 风格的工具定义
    要的也是 JSON Schema。唯一要做的是把字段名换成 `parameters`。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in tools.tools
    ]


def content_to_text(result) -> str:
    """把 MCP 的调用结果拍平成字符串（模型的上下文只认文本）。"""
    blocks = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        blocks.append(text if text is not None else str(block))
    return "\n".join(blocks)


async def run(question: str, groups: str) -> int:
    llm = ChatDeepSeek(model=settings.chat_model, temperature=0)

    async with Client(server_params(groups), raise_exceptions=False) as mcp:
        tools = await mcp.list_tools()
        schemas = to_function_schemas(tools)
        print(f"工具清单已获取：{[item['function']['name'] for item in schemas]}\n")

        bound = llm.bind_tools(schemas)
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]

        for round_no in range(1, MAX_ROUNDS + 1):
            started = time.perf_counter()
            response = await bound.ainvoke(messages)
            elapsed = (time.perf_counter() - started) * 1000
            messages.append(response)

            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                print(f"--- 第 {round_no} 轮：模型决定作答（{elapsed:.0f}ms）")
                print(f"\n回答：\n{response.content}")
                return 0

            print(f"--- 第 {round_no} 轮：模型决定调用 {len(calls)} 个工具（{elapsed:.0f}ms）")
            for call in calls:
                print(f"    → {call['name']}({call['args']})")
                started = time.perf_counter()
                result = await mcp.call_tool(call["name"], call["args"])
                text = content_to_text(result)
                cost = (time.perf_counter() - started) * 1000
                preview = text.replace("\n", " ")[:110]
                print(f"      ← {cost:.0f}ms，返回 {len(text)} 字：{preview}…")
                messages.append(
                    ToolMessage(content=text, tool_call_id=call["id"])
                )

        print(f"\n达到最大轮数（{MAX_ROUNDS}），强制收口。")
        return 1


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or "一线城市出差住一晚，最多能报多少住宿费？"
    groups = os.getenv("DEMO_GROUPS", "财务")
    print(f"问题：{question}\n身份：{groups}\n")
    return asyncio.run(run(question, groups))


if __name__ == "__main__":
    raise SystemExit(main())
