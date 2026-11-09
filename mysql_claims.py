"""报销提案与审计的 MySQL 实现。

## 为什么单独一个文件，而不是在 ClaimStore 里加 if

`ClaimStore` 的 SQL 里有一批 SQLite 特有写法（`AUTOINCREMENT`、`executescript`、
`isolation_level=None`），混在一个类里会变成"两个后端互相污染"的 if 森林。
独立实现 + 同一套契约测试，和 `vector_backend` 的处理方式一致。

## 一处刻意的取舍：时间戳继续用字符串

SQLite 版把时间存成 ISO-8601 字符串，`Proposal.created_at` 也是 `str`。
MySQL 版**保持字符串**（`VARCHAR(32)`）而不是 `DATETIME(6)`：

- 存储层不引入时区转换（MySQL 的 `DATETIME` 没有时区，`TIMESTAMP` 又会随时区变）
- `Proposal` 的字段类型不用改，渲染与测试全部保持一致
- ISO-8601 字符串的字典序等于时间序，需要排序时仍然可用

代价是失去了数据库侧的时间函数能力——但这里没有按时间范围查询的需求。

## 并发语义

`commit()` 与 SQLite 版**逐字一致**：单条带条件的 UPDATE，靠 `rowcount` 判定。
MySQL 的 `UPDATE ... WHERE status='pending'` 是当前读 + 行锁，所以并发提交时
同样只有一个能拿到 `rowcount == 1`——**不需要额外加锁，也不需要先读后写**。
"""
from __future__ import annotations

import json
from dataclasses import replace

from claims import (
    ALREADY_SUBMITTED,
    NOT_FOUND,
    SUBMITTED,
    Claim,
    ClaimEvent,          # noqa: F401  （重导出，便于调用方只 import 一个模块）
    ClaimStore,
    Proposal,
    _now,
    new_proposal_id,
)

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS proposals (
        proposal_id     VARCHAR(64)  NOT NULL,
        principal_id    VARCHAR(128) NOT NULL,
        conversation_id VARCHAR(64)  NOT NULL,
        status          VARCHAR(16)  NOT NULL,
        claim_json      TEXT         NOT NULL,
        needs_dual_sign TINYINT(1)   NOT NULL DEFAULT 0,
        created_at      VARCHAR(32)  NOT NULL,
        committed_at    VARCHAR(32)  NULL,
        PRIMARY KEY (proposal_id),
        KEY idx_proposal_owner (principal_id, conversation_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS claim_audit (
        id              BIGINT AUTO_INCREMENT PRIMARY KEY,
        at              VARCHAR(32)  NOT NULL,
        principal_id    VARCHAR(128) NOT NULL,
        conversation_id VARCHAR(64)  NOT NULL,
        proposal_id     VARCHAR(64)  NOT NULL,
        action          VARCHAR(16)  NOT NULL,
        result          VARCHAR(32)  NOT NULL,
        detail          TEXT         NOT NULL,
        KEY idx_audit_proposal (proposal_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """,
)


def parse_dsn(dsn: str) -> dict:
    """把 `mysql://user:pass@host:port/db` 拆成 pymysql 的连接参数。"""
    from urllib.parse import unquote, urlparse

    parsed = urlparse(dsn)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError(f"不支持的 DSN 协议：{parsed.scheme}")
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or "root"),
        "password": unquote(parsed.password or ""),
        "database": (parsed.path or "/rag_kb").lstrip("/"),
    }


class MySQLClaimStore(ClaimStore):
    """与 `ClaimStore` 行为一致的 MySQL 实现（继承只为复用类型标注与文档）。

    注意：**不调用父类的 `__init__`**——它建的是 SQLite 连接。
    """

    def __init__(self, dsn: str):
        import pymysql

        self.dsn = dsn
        self.params = parse_dsn(dsn)
        self._pymysql = pymysql
        # autocommit=True 与 SQLite 版的 isolation_level=None 对齐：
        # 每条语句各自成一个事务，状态迁移靠单条条件 UPDATE 的原子性。
        self._conn = pymysql.connect(
            **self.params, charset="utf8mb4", autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )
        with self._conn.cursor() as cursor:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)

    # ---------- 内部 ----------

    def close(self) -> None:
        self._conn.close()

    def reset(self) -> None:
        """清空两张表（测试用）：TRUNCATE 比 DELETE 更快，且自增 ID 归零。"""
        with self._conn.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            cursor.execute("TRUNCATE TABLE claim_audit")
            cursor.execute("TRUNCATE TABLE proposals")
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

    # ---------- 提案 ----------

    def create_proposal(self, principal_id: str, conversation_id: str,
                        claim: Claim, needs_dual_sign: bool) -> Proposal:
        proposal = Proposal(
            proposal_id=new_proposal_id(),
            principal_id=principal_id,
            conversation_id=conversation_id,
            status="pending",
            claim=claim,
            needs_dual_sign=needs_dual_sign,
            created_at=_now(),
            committed_at=None,
        )
        with self._conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO proposals (proposal_id, principal_id, conversation_id, status,"
                " claim_json, needs_dual_sign, created_at, committed_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (proposal.proposal_id, proposal.principal_id, proposal.conversation_id,
                 proposal.status, json.dumps(claim.as_dict(), ensure_ascii=False),
                 int(needs_dual_sign), proposal.created_at, None),
            )
        return proposal

    def get(self, principal_id: str, conversation_id: str,
            proposal_id: str) -> Proposal | None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM proposals WHERE proposal_id = %s"
                " AND principal_id = %s AND conversation_id = %s",
                (proposal_id, principal_id, conversation_id),
            )
            row = cursor.fetchone()
        return self._to_proposal(row) if row else None

    def commit(self, principal_id: str, conversation_id: str, proposal_id: str) -> str:
        """与 SQLite 版逐字一致的单条条件 UPDATE。

        `rowcount == 1` 才算成功；否则回查状态，区分 already_submitted / not_found。
        """
        with self._conn.cursor() as cursor:
            affected = cursor.execute(
                "UPDATE proposals SET status = 'committed', committed_at = %s"
                " WHERE proposal_id = %s AND principal_id = %s AND conversation_id = %s"
                " AND status = 'pending'",
                (_now(), proposal_id, principal_id, conversation_id),
            )
            if affected == 1:
                return SUBMITTED
            cursor.execute(
                "SELECT status FROM proposals WHERE proposal_id = %s"
                " AND principal_id = %s AND conversation_id = %s",
                (proposal_id, principal_id, conversation_id),
            )
            row = cursor.fetchone()
        if row is None:
            return NOT_FOUND          # 不存在 / 不属于你 / 不属于本会话 —— 同一结果
        return ALREADY_SUBMITTED if row["status"] == "committed" else NOT_FOUND

    # ---------- 审计 ----------

    def record(self, principal_id: str, conversation_id: str, proposal_id: str,
               action: str, result: str, detail: str = "") -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO claim_audit (at, principal_id, conversation_id, proposal_id,"
                " action, result, detail) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (_now(), principal_id, conversation_id, proposal_id, action, result, detail),
            )

    def audit_rows(self, proposal_id: str) -> list[dict]:
        with self._conn.cursor() as cursor:
            cursor.execute(
                "SELECT at, principal_id, conversation_id, proposal_id, action, result, detail"
                " FROM claim_audit WHERE proposal_id = %s ORDER BY id",
                (proposal_id,),
            )
            return list(cursor.fetchall())

    @staticmethod
    def _to_proposal(row: dict) -> Proposal:
        payload = json.loads(row["claim_json"])
        return Proposal(
            proposal_id=row["proposal_id"],
            principal_id=row["principal_id"],
            conversation_id=row["conversation_id"],
            status=row["status"],
            claim=Claim(
                category=payload["category"],
                amount=payload["amount"],
                nights=payload.get("nights", 0),
                city_tier=payload.get("city_tier", ""),
                invoice_ids=tuple(payload.get("invoice_ids", ())),
            ),
            needs_dual_sign=bool(row["needs_dual_sign"]),
            created_at=row["created_at"],
            committed_at=row["committed_at"],
        )


def open_claim_store(spec: str | None, default_path) -> ClaimStore:
    """按 URL 打开：`mysql://...` 走 MySQL，其余当成 SQLite 文件路径。"""
    if spec and spec.startswith(("mysql://", "mysql+pymysql://")):
        return MySQLClaimStore(spec)
    return ClaimStore(spec or default_path)
