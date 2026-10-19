"""会话状态与鉴权的单测（纯 SQLite，不加载模型、不调用 LLM）。"""
from pathlib import Path

import pytest

from conversation import (
    ConversationStore,
    SessionDenied,
    authorize_session,
    build_rewrite_context,
    build_turn_state,
    fingerprint,
    turn_pair,
)

FINANCE = {"财务", "public"}
HR = {"人事", "public"}


@pytest.fixture()
def store(tmp_path):
    return ConversationStore(tmp_path / "conv.db")


def _append(store, conv, question="q", answer="a", sources=(), **kwargs):
    ok = store.append(
        conv.conversation_id,
        turn_pair(question, answer, list(sources), **kwargs),
        expected_turn_count=store.get(conv.conversation_id).turn_count,
    )
    assert ok


# ---------- 基础存取 ----------

def test_create_and_get(store):
    conv = store.create("alice", FINANCE)
    assert conv.principal_id == "alice"
    assert conv.turn_count == 0
    assert store.get(conv.conversation_id) == conv
    assert store.get("不存在") is None


def test_conversation_id_is_unguessable(store):
    """128 位随机：会话 ID 客户端可见，不能靠猜。"""
    ids = {store.create("alice", FINANCE).conversation_id for _ in range(50)}
    assert len(ids) == 50
    assert all(len(value) >= 20 for value in ids)


def test_append_assigns_increasing_seq(store):
    conv = store.create("alice", FINANCE)
    _append(store, conv, "第一问", "第一答", ["报销流程.md"])
    _append(store, conv, "第二问", "第二答", ["报销标准.csv"])

    turns = store.turns(conv.conversation_id)
    assert [(t.seq, t.role, t.content) for t in turns] == [
        (1, "user", "第一问"), (2, "assistant", "第一答"),
        (3, "user", "第二问"), (4, "assistant", "第二答"),
    ]
    assert turns[1].citations == ("报销流程.md",)
    assert store.get(conv.conversation_id).turn_count == 4


def test_optimistic_lock_rejects_stale_append(store):
    """并发追加：以旧的 turn_count 写入必须失败，而不是把两轮交错插进去。"""
    conv = store.create("alice", FINANCE)
    _append(store, conv, "问一", "答一")
    stale = store.get(conv.conversation_id).turn_count

    assert store.append(conv.conversation_id, turn_pair("问二", "答二"),
                        expected_turn_count=stale) is True
    # 再用同样的旧计数写一次，必须被拒绝
    assert store.append(conv.conversation_id, turn_pair("问三", "答三"),
                        expected_turn_count=stale) is False
    assert [t.content for t in store.turns(conv.conversation_id)] == ["问一", "答一", "问二", "答二"]


def test_append_to_missing_conversation_fails(store):
    assert store.append("不存在", turn_pair("q", "a"), expected_turn_count=0) is False


def test_turns_limit_returns_most_recent(store):
    conv = store.create("alice", FINANCE)
    for index in range(3):
        _append(store, conv, f"问{index}", f"答{index}")
    recent = store.turns(conv.conversation_id, limit=2)
    assert [t.content for t in recent] == ["问2", "答2"]


def test_list_for_only_returns_own(store):
    alice = store.create("alice", FINANCE)
    store.create("bob", HR)
    listed = store.list_for("alice")
    assert [c.conversation_id for c in listed] == [alice.conversation_id]


# ---------- 规则 B：会话绑定单一身份 ----------

def test_authorize_allows_owner(store):
    conv = store.create("alice", FINANCE)
    assert authorize_session(store, conv.conversation_id, "alice", FINANCE) == conv


def test_authorize_none_creates_new(store):
    assert authorize_session(store, None, "alice", FINANCE) is None


def test_authorize_denies_other_principal(store):
    conv = store.create("alice", FINANCE)
    with pytest.raises(SessionDenied):
        authorize_session(store, conv.conversation_id, "bob", FINANCE)


def test_authorize_denies_missing_session(store):
    """不存在与无权必须是同一个异常——区分会泄露会话是否存在。"""
    with pytest.raises(SessionDenied):
        authorize_session(store, "随便编一个", "alice", FINANCE)


def test_authorize_denies_after_group_change(store):
    """组关系变化即拒绝会话。

    上一轮的助手回复正文可能含只对某组可见的内容；把历史回放给模型时，
    那部分内容不经过检索层，预过滤拦不住。所以只能整体拒绝会话。
    """
    conv = store.create("alice", FINANCE)
    _append(store, conv, "报销要几天", "单笔超五千需财务总监双签")
    with pytest.raises(SessionDenied):
        authorize_session(store, conv.conversation_id, "alice", {"public"})


def test_fingerprint_is_order_independent():
    assert fingerprint({"财务", "public"}) == fingerprint({"public", "财务"})
    assert fingerprint({"财务"}) != fingerprint({"人事"})


# ---------- 规则 C：历史上下文有界、无原文 ----------

def test_rewrite_context_included_and_bounded(store):
    conv = store.create("alice", FINANCE)
    for index in range(5):
        _append(store, conv, f"问{index}", f"答{index}")

    context = build_rewrite_context(store, conv, max_turns=2)
    assert context.startswith("<conversation_history>") and context.endswith("</conversation_history>")
    assert "问4" in context and "答4" in context       # 最近的保留
    assert "问0" not in context                         # 超出窗口的被裁掉


def test_rewrite_context_excludes_clarification_turns(store):
    """反问轮次不进常规上下文；用户对反问的回答是新 user turn，自然带上指代。"""
    conv = store.create("alice", FINANCE)
    _append(store, conv, "那住宿标准呢", "你指的是出差住宿还是会议住宿？",
            pending_clarification=True)
    context = build_rewrite_context(store, conv)
    assert "你指的是出差住宿还是会议住宿" not in context


def test_rewrite_context_is_empty_for_new_session(store):
    assert build_rewrite_context(store, None) == ""


def test_turn_never_stores_document_text(store):
    """Turn 只存来源名：原文入库就会变成跨轮证据。"""
    conv = store.create("alice", FINANCE)
    _append(store, conv, "问", "答", ["报销流程.md"])
    turn = store.turns(conv.conversation_id)[1]
    assert turn.citations == ("报销流程.md",)
    assert not hasattr(turn, "documents")
    # 存储层没有任何字段能装原文：字段就是 (role, content, citations, pending)
    assert set(turn.__dataclass_fields__) == {
        "conversation_id", "seq", "role", "content", "citations",
        "pending_clarification", "created_at",
    }


def test_delete_removes_turns(store):
    conv = store.create("alice", FINANCE)
    _append(store, conv)
    assert store.delete(conv.conversation_id) is True
    assert store.get(conv.conversation_id) is None
    assert store.turns(conv.conversation_id) == []


# ---------- 规则 A：本轮证据一律重建 ----------

def test_turn_state_always_starts_without_evidence(store):
    """规则 A 的唯一执行点：每轮 state 的证据字段都是空的。

    即使上一轮已经拿到过财务文档，本轮也必须从零开始检索——否则权限判定会
    退化成"首次检索时判定一次"，用户组关系变化后旧证据仍会出现在上下文里。
    """
    conv = store.create("alice", FINANCE)
    state = build_turn_state("那住宿标准呢", conv, build_rewrite_context(store, conv))
    assert state["hits"] == []
    assert state["table_rows"] == []
    assert state["search_memo"] == {}
    assert state["clarification_requested"] is False
    assert state["tool_rounds"] == 0


def test_turn_state_never_accepts_previous_evidence():
    """签名里没有"上一轮 state"这个入参：想复用旧证据必须改函数签名。

    用 introspection 断言，而不是靠约定——约定会被人忘掉，签名不会。
    """
    import inspect

    params = set(inspect.signature(build_turn_state).parameters)
    assert not params & {"hits", "table_rows", "search_memo", "state", "previous_state"}


def test_turn_state_carries_history_but_not_documents(store):
    conv = store.create("alice", FINANCE)
    _append(store, conv, "报销要几天内提交", "三十天内")
    history = build_rewrite_context(store, conv)
    state = build_turn_state("那住宿标准呢", conv, history)
    assert "报销要几天内提交" in state["history"]       # 指代改写所需
    assert state["hits"] == []                          # 但没有任何证据
