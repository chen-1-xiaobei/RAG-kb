"""checkpointer 的离线单测：状态存档、按轮次分线程、工具不进 state。

不加载模型、不碰网络：用 stub 节点组装图，验证的是"存档机制"本身。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent_flow import build_agent_graph  # noqa: E402
from checkpointing import trace_steps, turn_thread_id  # noqa: E402
from tools import resolve_tools  # noqa: E402


def make_graph():
    """三步图：agent -> tools -> agent -> answer（用 stub 节点，不碰模型）。"""
    calls = {"tools": 0}

    def agent_node(state, config=None):
        if state.get("tool_rounds", 0) == 0:
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": "search", "args": {"q": "住宿"}, "id": "c1"},
            ])]}
        return {"messages": [AIMessage(content="最终答案")]}

    def tools_node(state, config=None):
        calls["tools"] += 1
        return {
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "hits": [f"doc-{calls['tools']}"],
        }

    def answer_node(state, config=None):
        return {"response": "答案"}

    return build_agent_graph(
        agent_node, tools_node, answer_node, checkpointer=InMemorySaver(),
    )


def initial_state() -> dict:
    return {
        "question": "住宿标准",
        "messages": [HumanMessage("住宿标准")],
        "hits": [],
        "table_rows": [],
        "search_memo": {},
        "tool_rounds": 0,
        "conversation_id": "conv-1",
    }


def test_thread_id_is_per_turn_not_per_conversation():
    """按轮次分线程：同一会话的相邻两轮必须落在不同线程。

    否则 LangGraph 会把上一轮的 hits 合并进本轮，绕过"每轮重新检索"的规则。
    """
    assert turn_thread_id("conv-1", 1) != turn_thread_id("conv-1", 2)
    assert turn_thread_id("conv-1", 1).startswith("conv-1#")


def test_execution_creates_checkpoints_and_trace_summary():
    graph = make_graph()
    config = {"configurable": {"thread_id": turn_thread_id("conv-1", 1)}}

    result = graph.invoke(initial_state(), config)
    assert result["response"] == "答案"

    steps = trace_steps(graph, config["configurable"]["thread_id"])
    assert len(steps) >= 2, "多步执行应当留下多个 checkpoint"
    # 摘要只给结构信息，不给完整 state
    assert {"checkpoint_id", "next", "message_count", "tool_calls"} <= set(steps[0])
    assert "question" not in steps[0], "摘要不应带出提问原文"


def test_two_threads_do_not_share_state():
    graph = make_graph()
    first = {"configurable": {"thread_id": turn_thread_id("conv-1", 1)}}
    second = {"configurable": {"thread_id": turn_thread_id("conv-1", 2)}}

    graph.invoke(initial_state(), first)
    reuse = initial_state()
    reuse["hits"] = []
    graph.invoke(reuse, second)

    # 第二个线程的第一条 checkpoint 不应继承第一个线程的 hits
    history = list(graph.get_state_history(second))
    earliest = history[-1].values
    assert earliest.get("hits") in ([], None), "新线程不得继承上一轮的检索证据"


def test_trace_steps_is_empty_for_unknown_thread():
    graph = make_graph()
    assert trace_steps(graph, "never-used-thread") == []


def test_resolve_tools_prefers_config_over_state():
    """工具闭包从 config 取：state 里的工具表只是兼容用。"""
    from_state = {"search": object()}
    from_config = {"search": object()}

    assert resolve_tools({"tools": from_state}) is from_state
    resolved = resolve_tools(
        {"tools": from_state}, {"configurable": {"tools": from_config}}
    )
    assert resolved is from_config


def test_state_after_run_contains_no_callables():
    """state 必须可序列化——这是接 checkpointer 的前提。

    踩过的坑：工具是按请求构造的闭包，原先放在 state 里；一旦挂上 checkpointer，
    每个 super-step 都要把 state 序列化落盘，闭包会直接让执行失败。
    """
    graph = make_graph()
    config = {"configurable": {"thread_id": turn_thread_id("conv-2", 1)}}
    graph.invoke(initial_state(), config)

    for state in graph.get_state_history(config):
        for key, value in (state.values or {}).items():
            assert not callable(value), f"state 字段 {key} 是不可序列化的 callable"
