"""会话状态：一个用户一条连续线。

数据模型的两个刻意选择：

1. **Turn 里不存工具返回的原文**，只存问答文本与"引用了哪些来源"。
   原文一旦入库就成了"跨轮证据"，而本项目的原则是每轮重新检索、
   不复用跨轮证据（否则权限判定会退化成"首次检索时判定一次"）。
   只存来源名，既保证审计可追溯，又在结构上让"把上一轮检索正文回放给模型"
   这件事无从发生。

2. **会话绑定单一 Principal 与创建时的权限组指纹**。
   上一轮的助手回复正文里可能含有只对某组可见的内容；如果用户的组关系变了，
   把历史回放给模型就等于把那段受限内容重新送进上下文——检索层的预过滤拦不住它，
   因为它不经过检索层。指纹不匹配时直接拒绝会话（见 main.authorize_session）。

存储选型：单实例内网用 SQLite 足够，接口按 ConversationStore 抽象，
将来换 Redis 只需换实现。
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

PUBLIC_GROUP = "public"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id    TEXT PRIMARY KEY,
    principal_id       TEXT NOT NULL,
    groups_fingerprint TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    turn_count         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_conv_principal ON conversations(principal_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS turns (
    conversation_id       TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    seq                   INTEGER NOT NULL,
    role                  TEXT NOT NULL,
    content               TEXT NOT NULL,
    citations             TEXT NOT NULL DEFAULT '[]',
    pending_clarification INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT NOT NULL,
    PRIMARY KEY (conversation_id, seq)
);
"""


def fingerprint(groups) -> str:
    """权限组集合的稳定指纹：排序后哈希，与传入顺序无关。"""
    joined = "\x1f".join(sorted(groups))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    """128 位随机会话 ID。客户端持有它，等同于 bearer token——但权限判定永远看
    Principal，绝不看"知道这个 ID"。"""
    return secrets.token_urlsafe(16)


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    principal_id: str
    groups_fingerprint: str
    created_at: str
    updated_at: str
    turn_count: int


@dataclass(frozen=True)
class Turn:
    conversation_id: str
    seq: int
    role: str                      # "user" | "assistant"
    content: str
    citations: tuple[str, ...]     # 只记来源名，不含原文
    pending_clarification: bool
    created_at: str


def turn_pair(question: str, answer: str, sources: list[str] | None = None,
              *, pending_clarification: bool = False) -> list[Turn]:
    """构造一轮问答的两条 Turn（seq 由存储层分配）。sources 只记来源名，不含原文。"""
    stamp = _now()
    return [
        Turn("", 0, "user", question, (), False, stamp),
        Turn("", 0, "assistant", answer, tuple(sources or ()), pending_clarification, stamp),
    ]


class ConversationStore:
    """SQLite 实现。每次操作独立连接：sqlite3 连接不跨线程共享，避免
    check_same_thread 与锁的坑；写入用 BEGIN IMMEDIATE 串行化。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None：关闭 sqlite3 的隐式事务管理，让 BEGIN IMMEDIATE /
        # COMMIT / ROLLBACK 完全由本模块显式控制。否则驱动会在 DML 前自动开事务，
        # 与本模块的显式 BEGIN 冲突（"cannot start a transaction within a transaction"）。
        conn = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize(self) -> None:
        with self._init_lock:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)

    # ---------- 写 ----------

    def create(self, principal_id: str, groups) -> Conversation:
        record = Conversation(
            conversation_id=new_id(),
            principal_id=principal_id,
            groups_fingerprint=fingerprint(groups),
            created_at=_now(),
            updated_at=_now(),
            turn_count=0,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)",
                (record.conversation_id, record.principal_id, record.groups_fingerprint,
                 record.created_at, record.updated_at, record.turn_count),
            )
        return record

    def append(self, conversation_id: str, turns: list[Turn], *,
               expected_turn_count: int) -> bool:
        """追加轮次；与库内计数不一致时返回 False（乐观锁，防并发交错）。

        多轮会话天然有序：两个并发请求若各自读到同一 turn_count，后写的必须失败，
        而不是把两轮交错插进去。
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT turn_count FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None or row["turn_count"] != expected_turn_count:
                conn.rollback()
                return False
            seq = expected_turn_count
            for turn in turns:
                seq += 1
                conn.execute(
                    "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (conversation_id, seq, turn.role, turn.content,
                     json.dumps(list(turn.citations), ensure_ascii=False),
                     int(turn.pending_clarification), _now()),
                )
            conn.execute(
                "UPDATE conversations SET turn_count = ?, updated_at = ? WHERE conversation_id = ?",
                (seq, _now(), conversation_id),
            )
            conn.commit()
            return True

    def delete(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )
            return cursor.rowcount > 0

    # ---------- 读 ----------

    def get(self, conversation_id: str) -> Conversation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
        return self._to_conversation(row) if row else None

    def turns(self, conversation_id: str, limit: int | None = None) -> list[Turn]:
        """按时间正序返回轮次；limit 表示"最近 N 条"。"""
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    "SELECT * FROM turns WHERE conversation_id = ? ORDER BY seq",
                    (conversation_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM turns WHERE conversation_id = ? "
                    "ORDER BY seq DESC LIMIT ?) ORDER BY seq",
                    (conversation_id, limit),
                ).fetchall()
        return [self._to_turn(row) for row in rows]

    def list_for(self, principal_id: str, limit: int = 20) -> list[Conversation]:
        """只列自己的会话：按 principal 过滤是查询的天然形状，而不是事后过滤。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE principal_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (principal_id, limit),
            ).fetchall()
        return [self._to_conversation(row) for row in rows]

    @staticmethod
    def _to_conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(
            conversation_id=row["conversation_id"],
            principal_id=row["principal_id"],
            groups_fingerprint=row["groups_fingerprint"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            turn_count=row["turn_count"],
        )

    @staticmethod
    def _to_turn(row: sqlite3.Row) -> Turn:
        return Turn(
            conversation_id=row["conversation_id"],
            seq=row["seq"],
            role=row["role"],
            content=row["content"],
            citations=tuple(json.loads(row["citations"])),
            pending_clarification=bool(row["pending_clarification"]),
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------------
# 会话鉴权与上下文构造
# ---------------------------------------------------------------------------

class SessionDenied(Exception):
    """统一拒绝：不存在、不属于你、组关系已变更，对外都是同一个 404。

    区分这几种情况会泄露"会话是否存在"，与表格工具"表不存在与无权同文案"
    是同一套哲学。
    """


def authorize_session(store: ConversationStore, conversation_id: str | None,
                      principal_id: str, allowed_groups) -> Conversation | None:
    """conversation_id 为 None 时返回 None（由调用方新建）；任何校验失败抛 SessionDenied。"""
    if conversation_id is None:
        return None
    conversation = store.get(conversation_id)
    if conversation is None:
        raise SessionDenied
    if conversation.principal_id != principal_id:
        raise SessionDenied
    if conversation.groups_fingerprint != fingerprint(allowed_groups):
        raise SessionDenied
    return conversation


def build_rewrite_context(store: ConversationStore, conversation: Conversation | None,
                          max_turns: int = 3) -> str:
    """仅供查询改写的对话上下文：最近 N 轮问答，有界、无原文、明确标注不可信。

    反问轮次不进入常规上下文（它没提供新信息），用户对反问的回答本身就是新的
    user turn，会自然带上指代上下文。
    """
    if conversation is None:
        return ""
    recent = [turn for turn in store.turns(conversation.conversation_id, limit=max_turns * 2)
              if not turn.pending_clarification]
    if not recent:
        return ""
    lines = [
        f"{'用户' if turn.role == 'user' else '助手'}: {turn.content}"
        for turn in recent
    ]
    return "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"


def build_turn_state(question: str, conversation: Conversation | None,
                     history: str, **fields) -> dict:
    """构造本轮 agent state —— **规则 A 的唯一执行点**。

    证据字段一律硬编码为空：不跨轮继承，签名也不接受"上一轮的 state"。
    想让旧证据复活就得改这个函数，让违规变得显眼而不是顺手。

    `search_memo` 同样不跨轮：跨轮共享会让本轮命中上轮的缓存结果，
    等于绕开"本轮重新鉴权、重新检索"的语义。

    其余字段（tools / user_id / groups / conversation_id / top_k）由调用方传入，
    因此本函数保持纯函数，可在没有模型的环境下单测。
    """
    return {
        "question": question,
        "history": history,
        "messages": [],          # 由调用方填入首条 HumanMessage
        "hits": [],
        "table_rows": [],
        "search_memo": {},
        "claim_events": [],
        "clarification_requested": False,
        "tool_rounds": 0,
        **fields,
    }
