"""Agentic RAG 图单测（stub 节点 + 构造消息，不 import app、不调用 LLM）。"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_flow import build_agent_graph, build_evidence, make_route_after_agent
from tools import execute_tools


class FakeDoc:
    def __init__(self, doc_id):
        self.metadata = {"id": doc_id}
        self.page_content = doc_id


class StubTool:
    def __init__(self, result):
        self._result = result

    def invoke(self, args):
        return self._result


def _state(**overrides):
    state = {
        "question": "q", "top_k": 4, "messages": [HumanMessage("q")],
        "tools": {}, "hits": [], "table_rows": [], "tool_rounds": 0,
    }
    state.update(overrides)
    return state


def _ai_with_call(call_id="c1", name="search_knowledge_base"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"question": "q"}, "id": call_id}],
    )


def test_graph_topology_has_agent_tools_answer():
    graph = build_agent_graph(
        lambda state: {}, lambda state: {}, lambda state: {"response": "x"}
    )
    assert {"agent", "tools", "answer"} <= set(graph.get_graph().nodes)


def test_route_tool_calls_go_to_tools():
    route = make_route_after_agent(3)
    assert route(_state(messages=[_ai_with_call()])) == "tools"


def test_route_no_tool_calls_goes_to_answer():
    route = make_route_after_agent(3)
    assert route(_state(messages=[AIMessage(content="直接作答")])) == "answer"
    # ToolMessage 没有 tool_calls 属性：getattr 容错，同样路由到 answer
    assert route(_state(messages=[ToolMessage(content="材料", tool_call_id="c1")])) == "answer"


def test_route_round_limit_forces_answer():
    route = make_route_after_agent(2)
    assert route(_state(messages=[_ai_with_call()], tool_rounds=1)) == "tools"
    assert route(_state(messages=[_ai_with_call()], tool_rounds=2)) == "answer"


def test_build_evidence_empty_and_merged():
    assert build_evidence(_state()) == []
    doc, row = FakeDoc("d1"), FakeDoc("t1")
    assert build_evidence(_state(hits=[(doc, 0.9)], table_rows=[row])) == [
        (doc, 0.9), (row, 0.0),  # 表格行没有检索分数，score 以 0.0 占位
    ]


def test_graph_runs_tool_loop_offline():
    """离线端到端：stub agent 第一轮调工具、第二轮收手，工具节点用真 execute_tools。"""
    rounds = {"n": 0}

    def agent_node(state):
        rounds["n"] += 1
        if rounds["n"] == 1:
            return {"messages": [_ai_with_call()]}
        return {"messages": [AIMessage(content="资料已足够")]}

    seen = {}

    def answer_node(state):
        seen["evidence"] = build_evidence(state)
        return {"response": "最终答案"}

    doc = FakeDoc("d1")
    graph = build_agent_graph(agent_node, execute_tools, answer_node, max_tool_rounds=3)
    result = graph.invoke(_state(tools={
        "search_knowledge_base": StubTool(("材料文本", [(doc, 0.8)])),
    }))

    assert rounds["n"] == 2              # agent 被调用两次：调工具 -> 收手
    assert result["tool_rounds"] == 1
    assert result["response"] == "最终答案"
    assert seen["evidence"] == [(doc, 0.8)]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].tool_call_id == "c1"  # 与 AIMessage.tool_calls 的 id 对齐


def test_graph_round_limit_terminates_loop():
    """agent 永远返回 tool_calls 时，轮数上限强制收敛到 answer。"""
    def agent_node(state):
        return {"messages": [_ai_with_call(call_id=f"c{state.get('tool_rounds', 0)}")]}

    graph = build_agent_graph(
        agent_node, execute_tools, lambda state: {"response": "收敛"}, max_tool_rounds=2
    )
    result = graph.invoke(_state(tools={
        "search_knowledge_base": StubTool(("材料", [])),
    }))
    assert result["tool_rounds"] == 2
    assert result["response"] == "收敛"
