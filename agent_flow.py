"""Agentic RAG 显式工作流：agent（bind_tools 决策）-> [tools -> agent]*N -> answer。

    START -> agent -> (route) -> tools -> agent -> ... -> answer -> END

route 由 make_route_after_agent 构造：最后一条 AIMessage 有 tool_calls 且
工具轮数未超上限 -> tools；否则（模型直接作答或轮数耗尽）-> answer。
无证据作答不会漏出去：答案节点的证据门槛会拦截并拒答。
"""
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    user_id: str
    groups: list[str]
    # 会话标识（无会话时为空串）。只用于回传响应与落库，不参与权限判定。
    conversation_id: str
    # 仅供查询改写的对话上下文（最近 N 轮问答，有界、无原文、已标注不可信）。
    history: str
    # 对话消息：AIMessage.tool_calls 与 ToolMessage.tool_call_id 一一对应，
    # 模型据此把工具结果与调用对上号
    messages: Annotated[list[AnyMessage], add_messages]
    # per-request 工具闭包表：工具名 -> 已绑定 Principal 的工具实例（模型不可见）。
    # 注意：callable 放 state 只在无 checkpointer 的纯内存 invoke 下安全；
    # 若将来接 checkpoint 持久化，需改为 state 存工具名、节点从注册表取实例。
    tools: dict[str, Any]
    hits: list[Any]        # 跨轮累积的检索证据 [(Document, score)]，按 id 去重
    table_rows: list[Any]  # 跨轮累积的表格伪 Document
    # 单请求内的检索去重表：归一化调用键 -> (材料文本, 证据)。
    # 只在一次请求的 state 里存活，不跨用户共享，因此不引入权限泄露面。
    search_memo: dict[str, Any]
    # 本轮是否调用过反问工具。答案节点据此区分"该问"与"该拒"：
    # 无证据 + 调过反问 -> 反问；无证据 + 没调 -> 拒答（硬门槛）。
    clarification_requested: bool
    # 本轮的写操作事件（提案 / 提交）。答案节点据此渲染确认卡，且**不调用 LLM**：
    # 金额与提案号由服务端渲染，模型无法在确认文案里改数字。
    claim_events: list[Any]
    tool_rounds: int       # 工具节点执行轮数，路由上限用
    response: Any


def make_route_after_agent(max_tool_rounds: int = 3) -> Callable[[AgentState], str]:
    """agent 节点后置路由：有工具调用且轮数未超 -> tools；否则 -> answer。

    模型第一轮就不调工具直接作答也会进 answer，由答案节点的证据门槛拦截，
    因此这里不必区分"作答"与"收手"，统一放行进 answer 即可。
    """
    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("tool_rounds", 0) < max_tool_rounds:
            return "tools"
        return "answer"
    return route_after_agent


def build_evidence(state: AgentState) -> list:
    """合并检索 hits 与表格伪文档为最终证据；空 = 无证据（答案节点据此拒答）。

    表格行没有检索分数，score 以 0.0 占位——仅影响 Source.score 展示值，
    citation 编号/校验/拒答逻辑与检索文档完全一致。
    """
    hits = state.get("hits") or []
    rows = state.get("table_rows") or []
    return list(hits) + [(doc, 0.0) for doc in rows]


def build_agent_graph(
    agent_node: Callable[[AgentState], dict],
    tools_node: Callable[[AgentState], dict],
    answer_node: Callable[[AgentState], dict],
    max_tool_rounds: int = 3,
    route_after_agent: Callable[[AgentState], str] | None = None,
):
    """组装 Agentic RAG 图：agent -> [tools -> agent]* -> answer。

    三个节点与可选路由全部注入式传入（与旧版 retrieve/rerank/answer 同风格），
    单测可注入 stub 节点与 stub 路由，不触碰任何模型。
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "agent")
    route = route_after_agent or make_route_after_agent(max_tool_rounds)
    graph.add_conditional_edges("agent", route, {"tools": "tools", "answer": "answer"})
    graph.add_edge("tools", "agent")
    graph.add_edge("answer", END)
    return graph.compile()
