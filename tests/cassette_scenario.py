"""一个多轮场景的确定性定义，录制与回放共用同一份代码。

为什么要单独成模块：如果录制脚本和回放测试各写一遍流程，两边迟早漂移——
录的时候问 A、回放的时候问 B，键永远对不上。共用同一份 `run_turn` 才不会。

场景是：
    第 1 轮「报销要几天内提交？」      -> 模型检索
    第 2 轮「那住宿标准呢？」          -> 模型应把指代替换成完整查询后检索

这正是项目 README「已知边界」里承认没覆盖的那条：多轮改写质量无法自动评测。
有了 cassette，第 2 轮模型写了什么查询被完整录下来了，于是可以断言。
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from agent_flow import build_agent_graph
from tools import SEARCH_TOOL_NAME, execute_tools

MODEL_NAME = "deepseek-chat"
AGENT_PROMPT = (
    "你是企业知识库助手。问题涉及企业制度、流程、手册时调用 search_knowledge_base 检索；"
    "多轮对话中，检索查询必须是完整的、不依赖上下文也能看懂的句子。"
)
# 第 2 轮把上一轮对话塞进 system prompt，模拟 app.build_rewrite_context 的做法
HISTORY_TURN_1 = (
    "<conversation_history>\n"
    "用户: 报销要几天内提交？\n"
    "助手: 员工需在费用发生后三十天内提交报销单。\n"
    "</conversation_history>"
)


class FakeDoc:
    metadata = {"id": "d1", "source": "报销流程.md"}
    page_content = "员工需在费用发生后三十天内提交报销单。"


class StubSearch:
    """只记录模型给了什么查询，不真的检索——测的是"模型写了什么"，不是检索质量。"""

    name = SEARCH_TOOL_NAME

    def __init__(self, log: list[str]):
        self._log = log

    def invoke(self, args):
        self._log.append(args.get("question", ""))
        return "命中 1 条资料：员工需在费用发生后三十天内提交报销单。", [(FakeDoc(), 5.0)]


@dataclass
class TurnResult:
    queries: list[str]
    response: object
    message_count: int


def build(model, queries: list[str]):
    """用给定的（可能是录制或回放的）模型组装一张真实的 agent 图。"""
    tool = StubSearch(queries)

    def agent_node(state):
        # 与 app._agent_agent 的做法一致：历史拼接在 system 里，不混进消息列表
        system = AGENT_PROMPT
        history = state.get("history") or ""
        if history:
            system = f"{system}\n{history}"
        bound = model.bind_tools([tool])
        response = bound.invoke([SystemMessage(system), *state["messages"]])
        return {"messages": [response]}

    def tools_node(state):
        return execute_tools(state)

    def answer_node(state):
        return {"response": "最终答案"}

    graph = build_agent_graph(agent_node, tools_node, answer_node)
    return graph, tool


def run_turn(model, question: str, history: str = "") -> TurnResult:
    """跑一轮对话，返回模型实际给出的检索查询。"""
    queries: list[str] = []
    graph, tool = build(model, queries)
    result = graph.invoke({
        "question": question,
        "messages": [HumanMessage(question)],
        "tools": {SEARCH_TOOL_NAME: tool},
        "hits": [],
        "table_rows": [],
        "search_memo": {},
        "tool_rounds": 0,
        "history": history,
        "conversation_id": "conv-cassette",
    })
    return TurnResult(queries, result.get("response"), len(result.get("messages", [])))


def run_scenario(model) -> tuple[TurnResult, TurnResult]:
    """完整的两轮场景：录制与回放都调用它，保证键一致。"""
    first = run_turn(model, "报销要几天内提交？")
    second = run_turn(model, "那住宿标准呢？", history=HISTORY_TURN_1)
    return first, second
