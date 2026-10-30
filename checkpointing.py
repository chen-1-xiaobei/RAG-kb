"""LangGraph checkpointer：把每次 Agent 执行的状态存下来，支持回看与恢复。

两条设计决定都来自本项目的既有约束。

**一、thread_id 按轮次，不按会话。**

`{conversation_id}#{turn_no}`。原因是 `AgentState` 里有 `messages`（带 add_messages 归约器）
与 `hits` / `table_rows`：如果同一会话共享一个 thread，LangGraph 会把上一轮的 state
**合并**进本轮——上一轮的检索证据就"活"到了这一轮，直接绕过规则 A（每轮重新检索、
不复用跨轮证据，否则权限判定会退化成"首次检索时判定一次"）。

按轮次分线程之后，checkpoint 只承担两件事：
- **回看**：这一轮模型走了几步、每步调了哪些工具；
- **恢复**：这一轮中途失败，可以从最后一个 checkpoint 续跑。

跨轮记忆仍然由 `conversations.db` 负责——那里带着创建者的 principal 与权限组指纹。

**二、SQLite 优先，内存兜底。**

装了 `langgraph-checkpoint-sqlite` 就持久化到 `checkpoints.db`（重启后仍能回看）；
没装则退化为 `InMemorySaver` 并打一条告警，功能不中断，只是轨迹不跨进程。
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("rag-kb")


def turn_thread_id(conversation_id: str, turn_no: int) -> str:
    """一轮一个线程。`turn_no` 用「轮次序号」，从 1 开始。"""
    return f"{conversation_id or 'stateless'}#{turn_no}"


def build_checkpointer(path: Path):
    """优先用 SQLite 持久化，失败则退化为内存。"""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), check_same_thread=False)
        logger.info("Checkpointer: SQLite (%s)", path.name)
        return SqliteSaver(connection)
    except Exception as exc:                      # 未安装或初始化失败都不阻断服务
        from langgraph.checkpoint.memory import InMemorySaver

        logger.warning(
            "Checkpointer: 退化为内存（%s）。执行轨迹不会跨进程保留；"
            "安装 langgraph-checkpoint-sqlite 可持久化到 %s。",
            exc, path.name,
        )
        return InMemorySaver()


def thread_exists(graph, thread_id: str) -> bool:
    """该线程是否有存档。

    用 `get_tuple` 而不是 `list()`：`list()` 的内部约定随版本变化
    （实测在 langgraph-checkpoint 4.3 上对空 configurable 直接 KeyError: 'thread_id'），
    而 `get_tuple` 是稳定的单点查询。
    """
    try:
        return graph.checkpointer.get_tuple({"configurable": {"thread_id": thread_id}}) is not None
    except Exception as exc:
        logger.warning("thread_probe_failed thread=%s err=%s", thread_id, exc)
        return False


def latest_thread(graph, conversation_id: str, turn_count: int) -> str | None:
    """该会话最近一次执行的线程。

    不靠枚举，也不靠单一算术：`conversation.turn_count` 记的是**行数**
    （一问一答算 2 行），而线程名用的是**执行前**的 turn_count，所以最近一次执行
    对应的后缀要么是 `turn_count`（本轮落库失败、计数没推进），要么是
    `turn_count - 2`（正常落库）。从大到小探测，第一个存在的就是它。
    """
    candidates = list(range(turn_count if turn_count % 2 == 0 else turn_count - 1, -1, -2))
    for suffix in candidates:
        thread_id = turn_thread_id(conversation_id, suffix)
        if thread_exists(graph, thread_id):
            return thread_id
    return None


def trace_steps(graph, thread_id: str) -> list[dict]:
    """把某个线程的 checkpoint 历史压成可读的步骤摘要。

    刻意**不返回完整 state**：里面有提问原文、检索证据与模型回复，属于会话内容，
    应当由会话鉴权控制访问。这里只给"走了几步、每步进哪个节点、调了哪些工具"。
    """
    config = {"configurable": {"thread_id": thread_id}}
    steps: list[dict] = []
    try:
        history = list(graph.get_state_history(config))
    except Exception as exc:                      # 没有该线程的 checkpoint
        logger.warning("trace_unavailable thread=%s err=%s", thread_id, exc)
        return steps

    for state in history:
        values = state.values or {}
        messages = values.get("messages") or []
        last = messages[-1] if messages else None
        steps.append({
            "checkpoint_id": (state.config.get("configurable") or {}).get("checkpoint_id"),
            "next": list(state.next),
            "message_count": len(messages),
            "tool_calls": [call.get("name") for call in (getattr(last, "tool_calls", None) or [])],
            "hits": len(values.get("hits") or []),
            "table_rows": len(values.get("table_rows") or []),
            "tool_rounds": values.get("tool_rounds", 0),
            "has_response": values.get("response") is not None,
        })
    return steps
