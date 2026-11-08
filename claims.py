"""费用报销的写操作协议：提案 → （用户确认）→ 提交。

**这是项目里第一个有副作用的工具**，因此它和只读工具的设计取向完全不同：

- 模型不能直接提交，只能**提案**。提案由服务端校验并持久化，模型拿到的是
  一个 proposal_id，而不是"我说了算"的金额。
- **确认卡由服务端从提案渲染**，不经过 LLM。否则模型可能在答案里篡改金额。
- 提交是**幂等**的：同一个 proposal_id 重复提交返回 `already_submitted`，
  绝不产生第二条记录。模型的"超时重试"在只读工具下无害，在写操作下会变成重复报销。
- 结果用**结构化结果码**，而不是异常字符串。因为"提交超时"和"提交失败"在字符串
  层面无法区分，模型据此决定是否重试并不可靠。
- 每次尝试（包括被策略拒绝的）都写**审计**，只追加。

本模块不 import 任何检索/LLM 组件，因此可以完全离线单测。
"""
from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# ---------- 结果码 ----------
PROPOSED = "proposed"
SUBMITTED = "submitted"
ALREADY_SUBMITTED = "already_submitted"
REJECTED_BY_POLICY = "rejected_by_policy"
NOT_FOUND = "not_found"

# ---------- 费用口径（与 data/财务/报销标准.csv 对齐；tests 会断言两者一致） ----------
LODGING_LIMITS = {"一线": 500.0, "其他": 350.0}
DUAL_SIGN_THRESHOLD = 5000.0          # 单笔超五千需部门负责人与财务总监双签
KNOWN_CATEGORIES = ("住宿", "市内交通", "餐饮补贴", "高铁", "飞机")
# 凭票报销的类型：没有发票不能提案
INVOICE_REQUIRED = ("住宿", "市内交通", "高铁", "飞机")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_proposal_id() -> str:
    return "p_" + secrets.token_urlsafe(12)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

MAX_CLAIM_AMOUNT = 1_000_000.0   # 单笔上限：纯粹防离谱输入，业务上限由政策判定
MAX_NIGHTS = 366
MAX_INVOICES = 20
MAX_INVOICE_LEN = 64

# 模型会把说法写活（"住宿费""打车""机票"），这里做**受控**归一并保留未知值：
# 归一表之外的词原样保留，交给 evaluate_policy 判成"未知类别"，
# 这样"类型校验"和"业务规则"不会混在一起。
CATEGORY_ALIASES = {
    "住宿费": "住宿", "住宿": "住宿",
    "市内交通费": "市内交通", "市内交通": "市内交通", "打车": "市内交通", "出租车": "市内交通",
    "餐饮补贴": "餐饮补贴", "餐补": "餐饮补贴", "饭补": "餐饮补贴",
    "高铁": "高铁", "高铁票": "高铁", "火车票": "高铁",
    "飞机": "飞机", "机票": "飞机",
}
CITY_TIER_ALIASES = {
    "一线": "一线", "一线城市": "一线",
    "北京": "一线", "上海": "一线", "广州": "一线", "深圳": "一线",
    "其他": "其他", "二线": "其他", "其他城市": "其他",
}

_CURRENCY_NOISE = str.maketrans("", "", " ,，¥￥元人民币")


class ClaimValidationError(ValueError):
    """入参不合法（类型 / 范围 / 格式）。

    与 `REJECTED_BY_POLICY` 刻意区分：那是"数据合规但不符合公司政策"，
    这是"模型给的东西根本没法解析"。两者对用户的话术不同，混一起会误导排查。
    """


def _to_finite_amount(value: object) -> float:
    """把模型给的金额转成有限的 float，拒绝 bool / NaN / Inf / 空串。

    bool 必须显式拒绝：Python 里 `float(True) == 1.0`，模型偶尔会传
    `amount=true`，静默变成 1 块钱比报错难查得多。
    """
    if isinstance(value, bool):
        raise ValueError("金额必须是数字，不能是布尔值")
    if isinstance(value, (int, float)):
        number = Decimal(str(value))
    elif isinstance(value, str):
        cleaned = value.translate(_CURRENCY_NOISE).strip()
        if not cleaned:
            raise ValueError("金额不能为空")
        try:
            number = Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"金额无法解析为数字：{value!r}") from exc
    else:
        raise ValueError(f"金额必须是数字，收到 {type(value).__name__}")
    if not number.is_finite():
        raise ValueError("金额必须是有限数字（不接受 NaN / Infinity）")
    return float(number.quantize(Decimal("0.01")))


def _to_nights(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("晚数必须是非负整数，不能是布尔值")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"晚数必须是整数，收到 {value}")
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return 0
        if not cleaned.isdigit():
            raise ValueError(f"晚数必须是整数，收到 {value!r}")
        return int(cleaned)
    raise ValueError(f"晚数必须是整数，收到 {type(value).__name__}")


def _to_invoice_ids(value: object) -> tuple[str, ...]:
    """接受列表或 "FP-1, FP-2" 这类字符串；去空白、去重、限制条数与长度。"""
    if value is None:
        return ()
    if isinstance(value, str):
        raw = [piece for piece in value.replace("，", ",").replace("、", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raise ValueError(f"发票号必须是列表或字符串，收到 {type(value).__name__}")

    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("发票号必须是字符串")
        token = item.strip()
        if not token:
            continue
        if len(token) > MAX_INVOICE_LEN:
            raise ValueError(f"发票号过长（最多 {MAX_INVOICE_LEN} 字符）")
        if token not in cleaned:            # 去重但保留顺序
            cleaned.append(token)
    if len(cleaned) > MAX_INVOICES:
        raise ValueError(f"一次最多 {MAX_INVOICES} 张发票")
    return tuple(cleaned)


class Claim(BaseModel):
    """一条报销诉求。金额单位元。

    用 pydantic 而不是 dataclass 的理由：**入参来自模型**，它会把 `"450"`、`true`、
    `NaN`、`"450元"`、`"FP-1, FP-2"` 这些形态混着给你。dataclass 只能靠裸
    `float()` / `int()`，`float(True)` 会静默变成 1.0。
    这里把"能安全归一化的归一化、归不了的明确报错"做成模型的一部分。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    category: str = Field(min_length=1, max_length=32)
    amount: float = Field(gt=0, le=MAX_CLAIM_AMOUNT, allow_inf_nan=False)
    nights: int = Field(default=0, ge=0, le=MAX_NIGHTS)
    city_tier: str = Field(default="", max_length=16)
    invoice_ids: tuple[str, ...] = Field(default=())

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: object) -> object:
        if isinstance(value, str):
            key = value.strip()
            return CATEGORY_ALIASES.get(key, key)
        return value

    @field_validator("amount", mode="before")
    @classmethod
    def _normalize_amount(cls, value: object) -> object:
        return _to_finite_amount(value)

    @field_validator("nights", mode="before")
    @classmethod
    def _normalize_nights(cls, value: object) -> object:
        return _to_nights(value)

    @field_validator("city_tier", mode="before")
    @classmethod
    def _normalize_city_tier(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str):
            key = value.strip()
            return CITY_TIER_ALIASES.get(key, key)
        return value

    @field_validator("invoice_ids", mode="before")
    @classmethod
    def _normalize_invoice_ids(cls, value: object) -> object:
        return _to_invoice_ids(value)

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "amount": self.amount,
            "nights": self.nights,
            "city_tier": self.city_tier,
            "invoice_ids": list(self.invoice_ids),
        }


def parse_claim(**fields: object) -> Claim:
    """把模型给的原始参数安全地变成 Claim；失败时抛可读的 ClaimValidationError。

    错误文案会被直接回给模型（再由它转述给用户），所以按字段名 + 原值写清楚，
    例如「amount: 金额必须是数字，不能是布尔值」。
    """
    try:
        return Claim(**fields)
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            field = str(error["loc"][0]) if error["loc"] else "参数"
            message = error["msg"].removeprefix("Value error, ")
            problems.append(f"{field}: {message}")
        raise ClaimValidationError("；".join(problems)) from exc


@dataclass(frozen=True)
class ClaimEvent:
    """一次写操作尝试的结果，交给答案节点渲染。"""
    action: str                    # "propose" | "commit"
    result: str                    # 结果码之一
    proposal_id: str
    claim: Claim | None
    violations: tuple[str, ...] = ()
    needs_dual_sign: bool = False
    summary: str = ""


def evaluate_policy(claim: Claim) -> tuple[list[str], bool]:
    """返回 (违规原因列表, 是否需要双签)。空列表 = 通过。

    策略是**确定性**的：同样的入参永远得到同样结论，因此可以完全离线断言。
    """
    reasons: list[str] = []

    if claim.category not in KNOWN_CATEGORIES:
        reasons.append(f"未知费用类型：{claim.category}（可选：{'、'.join(KNOWN_CATEGORIES)}）")
        return reasons, False

    if claim.amount <= 0:
        reasons.append("金额必须大于 0")

    if claim.category in INVOICE_REQUIRED and not claim.invoice_ids:
        reasons.append(f"{claim.category}属于凭票报销，必须提供发票")

    if claim.category == "住宿":
        if claim.nights <= 0:
            reasons.append("住宿报销必须提供晚数")
        if claim.city_tier not in LODGING_LIMITS:
            reasons.append("住宿报销必须说明城市类型（一线 / 其他）")
        if claim.nights > 0 and claim.city_tier in LODGING_LIMITS:
            limit = LODGING_LIMITS[claim.city_tier] * claim.nights
            if claim.amount > limit:
                reasons.append(
                    f"住宿金额超出标准：{claim.city_tier}城市 {LODGING_LIMITS[claim.city_tier]:.0f} 元/晚"
                    f" × {claim.nights} 晚 = {limit:.0f} 元，申请 {claim.amount:.0f} 元"
                )

    return reasons, claim.amount > DUAL_SIGN_THRESHOLD


def summarize(claim: Claim, proposal_id: str) -> str:
    parts = [f"{claim.category}报销 {claim.amount:.2f} 元"]
    if claim.category == "住宿" and claim.nights:
        parts.append(f"{claim.city_tier}城市 {claim.nights} 晚")
    if claim.invoice_ids:
        parts.append(f"{len(claim.invoice_ids)} 张发票")
    return f"提案 {proposal_id}：" + "，".join(parts)


# ---------------------------------------------------------------------------
# 存储：提案 + 审计（单实例 SQLite，接口可替换）
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id     TEXT PRIMARY KEY,
    principal_id    TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    status          TEXT NOT NULL,          -- pending | committed
    claim_json      TEXT NOT NULL,
    needs_dual_sign INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    committed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposal_owner ON proposals(principal_id, conversation_id);
CREATE TABLE IF NOT EXISTS claim_audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    at              TEXT NOT NULL,
    principal_id    TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    proposal_id     TEXT NOT NULL,
    action          TEXT NOT NULL,          -- propose | commit
    result          TEXT NOT NULL,
    detail          TEXT NOT NULL DEFAULT ''
);
"""


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    principal_id: str
    conversation_id: str
    status: str
    claim: Claim
    needs_dual_sign: bool
    created_at: str
    committed_at: str | None


class ClaimStore:
    """提案与审计的持久化。

    所有查询都带 (principal_id, conversation_id) 双条件：提案归属**同时**绑定用户
    与会话，因此换一个会话或换一个用户拿着同一个 proposal_id 也读不到、提交不了。
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

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
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (proposal.proposal_id, proposal.principal_id, proposal.conversation_id,
                 proposal.status, json.dumps(claim.as_dict(), ensure_ascii=False),
                 int(needs_dual_sign), proposal.created_at, None),
            )
        return proposal

    def get(self, principal_id: str, conversation_id: str,
            proposal_id: str) -> Proposal | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM proposals WHERE proposal_id = ? "
                "AND principal_id = ? AND conversation_id = ?",
                (proposal_id, principal_id, conversation_id),
            ).fetchone()
        return self._to_proposal(row) if row else None

    def commit(self, principal_id: str, conversation_id: str, proposal_id: str) -> str:
        """把 pending 原子地置为 committed。

        用单条带条件的 UPDATE 完成状态迁移：并发提交时只有一个能拿到 rowcount=1，
        另一个走 already_submitted —— 不依赖先读后写，因此没有竞态窗口。
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE proposals SET status = 'committed', committed_at = ? "
                "WHERE proposal_id = ? AND principal_id = ? AND conversation_id = ? "
                "AND status = 'pending'",
                (_now(), proposal_id, principal_id, conversation_id),
            )
            if cursor.rowcount == 1:
                return SUBMITTED
            row = conn.execute(
                "SELECT status FROM proposals WHERE proposal_id = ? "
                "AND principal_id = ? AND conversation_id = ?",
                (proposal_id, principal_id, conversation_id),
            ).fetchone()
        if row is None:
            return NOT_FOUND          # 不存在、不属于你、不属于本会话 —— 同一结果
        return ALREADY_SUBMITTED if row["status"] == "committed" else NOT_FOUND

    # ---------- 审计 ----------

    def record(self, principal_id: str, conversation_id: str, proposal_id: str,
               action: str, result: str, detail: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO claim_audit (at, principal_id, conversation_id, proposal_id, "
                "action, result, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_now(), principal_id, conversation_id, proposal_id, action, result, detail),
            )

    def audit_rows(self, proposal_id: str) -> list[dict]:
        """测试与追溯用：某提案的所有审计记录，按时间正序。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT at, principal_id, conversation_id, proposal_id, action, result, detail "
                "FROM claim_audit WHERE proposal_id = ? ORDER BY id",
                (proposal_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _to_proposal(row: sqlite3.Row) -> Proposal:
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


# ---------------------------------------------------------------------------
# 渲染：确认卡由服务端生成，不经过 LLM
# ---------------------------------------------------------------------------

def render(event: ClaimEvent) -> tuple[str, bool, dict]:
    """把一次写操作结果渲染成 (答案文本, 是否待确认, 结构化卡片)。

    **刻意不调用 LLM**：金额、标准、提案 ID 都由服务端从提案数据渲染，
    模型无法在确认文案里改动任何一个数字。
    """
    if event.result == REJECTED_BY_POLICY:
        # 文案目标：让用户知道"下一步该做什么"，而不是一句"无法提交"。
        # 缺项在语义上确实是拒绝（不生成提案号），但对用户而言是"还差这些信息"。
        items = list(event.violations) or ["不符合报销政策"]
        listed = "\n".join(f"  {index}. {item}" for index, item in enumerate(items, 1))
        return (
            f"这次还不能发起提案，需要先补充以下信息：\n{listed}\n"
            f"补充后请再说一次（例如「住宿 450 元，1 晚，一线城市，发票号 FP-2026-001」），我再为您发起提案。",
            False,
            {"proposal_id": "", "action": event.action, "result": event.result,
             "summary": "", "amount": None, "needs_dual_sign": False,
             "violations": list(event.violations)},
        )

    if event.result == NOT_FOUND:
        # 与"表不存在与无权同文案"一致：不泄露提案是否存在
        return (
            "找不到可提交的报销提案，请重新发起申请。",
            False,
            {"proposal_id": "", "action": event.action, "result": event.result,
             "summary": "", "amount": None, "needs_dual_sign": False, "violations": []},
        )

    if event.result == ALREADY_SUBMITTED:
        return (
            f"该报销申请已经提交过了（提案 {event.proposal_id}），无需重复提交。",
            False,
            {"proposal_id": event.proposal_id, "action": "commit", "result": event.result,
             "summary": event.summary, "amount": None, "needs_dual_sign": False,
             "violations": []},
        )

    # proposed / submitted
    lines = [event.summary]
    if event.needs_dual_sign:
        lines.append("注意：单笔金额超过 5000 元，需要部门负责人与财务总监双签。")
    pending = event.result == PROPOSED
    lines.append("请确认后回复「确认提交」。" if pending else "已提交，等待审批。")
    return (
        "\n".join(lines),
        pending,
        {"proposal_id": event.proposal_id, "action": event.action, "result": event.result,
         "summary": event.summary,
         "amount": event.claim.amount if event.claim else None,
         "needs_dual_sign": event.needs_dual_sign, "violations": []},
    )
