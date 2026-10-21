"""写操作（报销提案与提交）单测：策略、幂等、归属、审计、渲染。

全部纯离线：不加载模型、不调用 LLM、不访问网络。
"""
import csv
from pathlib import Path

import pytest

from auth import Principal
from claims import (
    ALREADY_SUBMITTED,
    DUAL_SIGN_THRESHOLD,
    LODGING_LIMITS,
    NOT_FOUND,
    PROPOSED,
    REJECTED_BY_POLICY,
    SUBMITTED,
    Claim,
    ClaimEvent,
    ClaimStore,
    evaluate_policy,
    render,
)
from tools import commit_claim_impl, propose_claim_impl

ALICE = Principal("alice", frozenset({"财务", "public"}))
BOB = Principal("bob", frozenset({"财务", "public"}))
CONV = "conv-1"


def _lodging(amount=1000.0, nights=2, tier="一线", invoices=("inv-1",)):
    return Claim(category="住宿", amount=amount, nights=nights, city_tier=tier,
                 invoice_ids=invoices)


@pytest.fixture()
def store(tmp_path):
    return ClaimStore(tmp_path / "claims.db")


# ---------- 策略 ----------

def test_policy_accepts_within_standard():
    reasons, dual = evaluate_policy(_lodging(amount=1000.0, nights=2, tier="一线"))
    assert reasons == [] and dual is False


def test_policy_rejects_over_standard():
    reasons, _ = evaluate_policy(_lodging(amount=1200.0, nights=2, tier="一线"))
    assert len(reasons) == 1
    assert "超出标准" in reasons[0] and "1000" in reasons[0]


def test_policy_uses_city_tier_limits():
    """其他城市 350/晚：3 晚 1050 通过，1100 被拒。"""
    assert evaluate_policy(_lodging(amount=1050.0, nights=3, tier="其他"))[0] == []
    assert evaluate_policy(_lodging(amount=1100.0, nights=3, tier="其他"))[0]


def test_policy_requires_invoice_for_reimbursable_categories():
    reasons, _ = evaluate_policy(Claim(category="高铁", amount=500.0))
    assert any("发票" in reason for reason in reasons)
    # 餐饮补贴不凭票，无需发票
    assert evaluate_policy(Claim(category="餐饮补贴", amount=300.0))[0] == []


def test_policy_requires_nights_and_tier_for_lodging():
    reasons, _ = evaluate_policy(Claim(category="住宿", amount=500.0, invoice_ids=("i",)))
    assert any("晚数" in reason for reason in reasons)
    reasons, _ = evaluate_policy(Claim(category="住宿", amount=500.0, nights=1, invoice_ids=("i",)))
    assert any("城市类型" in reason for reason in reasons)


def test_policy_rejects_unknown_category_and_non_positive_amount():
    assert evaluate_policy(Claim(category="赌博", amount=10.0))[0]
    assert evaluate_policy(Claim(category="餐饮补贴", amount=0))[0]


def test_policy_flags_dual_sign_above_threshold():
    """超过 5000 不是拒绝，而是需要双签——这是"要求更严的审批"，不是"不许报销"。"""
    reasons, dual = evaluate_policy(_lodging(amount=6000.0, nights=12, tier="一线"))
    assert reasons == [] and dual is True
    assert evaluate_policy(_lodging(amount=DUAL_SIGN_THRESHOLD, nights=10, tier="一线"))[1] is False


def test_policy_constants_match_standards_csv():
    """策略里的金额口径必须与 data/财务/报销标准.csv 一致。

    这是防腐烂断言：改了标准表却忘了改策略（或反之），测试直接红。
    """
    path = Path(__file__).resolve().parent.parent / "data" / "财务" / "报销标准.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    limits = {}
    for row in rows:
        item = row["项目"]
        if item.startswith("住宿-"):
            tier = "一线" if "一线" in item else "其他"
            limits[tier] = float(row["标准"].replace("元/晚", ""))
    assert limits == LODGING_LIMITS


# ---------- 提案与提交 ----------

def test_propose_creates_pending_proposal(store):
    text, events = propose_claim_impl(_lodging(), ALICE, CONV, store)
    event = events[0]
    assert event.result == PROPOSED and event.proposal_id
    assert "提案" in text and "确认" in text
    proposal = store.get("alice", CONV, event.proposal_id)
    assert proposal is not None and proposal.status == "pending"


def test_propose_rejected_creates_no_proposal(store):
    text, events = propose_claim_impl(_lodging(amount=9999.0), ALICE, CONV, store)
    assert events[0].result == REJECTED_BY_POLICY
    assert "超出标准" in text
    # 被拒的提案不落库，因此也不存在可被提交的编号
    assert events[0].proposal_id == ""


def test_propose_requires_conversation(store):
    _, events = propose_claim_impl(_lodging(), ALICE, "", store)
    assert events[0].result == NOT_FOUND
    assert store.audit_rows("") == []


def test_commit_pending_proposal(store):
    _, events = propose_claim_impl(_lodging(), ALICE, CONV, store)
    proposal_id = events[0].proposal_id

    text, commit_events = commit_claim_impl(proposal_id, ALICE, CONV, store)
    assert commit_events[0].result == SUBMITTED
    assert "已提交" in text
    assert store.get("alice", CONV, proposal_id).status == "committed"


def test_commit_is_idempotent(store):
    """重复提交必须返回 already_submitted，且审计里只有一条 commit 记录。"""
    _, events = propose_claim_impl(_lodging(), ALICE, CONV, store)
    proposal_id = events[0].proposal_id

    assert commit_claim_impl(proposal_id, ALICE, CONV, store)[1][0].result == SUBMITTED
    text, again = commit_claim_impl(proposal_id, ALICE, CONV, store)
    assert again[0].result == ALREADY_SUBMITTED
    assert "已经提交过" in text

    commits = [row for row in store.audit_rows(proposal_id) if row["action"] == "commit"]
    assert len(commits) == 2                      # 两次尝试都留痕
    assert [row["result"] for row in commits] == [SUBMITTED, ALREADY_SUBMITTED]


def test_commit_unknown_proposal_is_not_found(store):
    assert commit_claim_impl("p_不存在", ALICE, CONV, store)[1][0].result == NOT_FOUND


def test_commit_other_principal_is_not_found(store):
    """他人的提案：与不存在同结果，不泄露存在性。"""
    _, events = propose_claim_impl(_lodging(), ALICE, CONV, store)
    proposal_id = events[0].proposal_id

    result = commit_claim_impl(proposal_id, BOB, CONV, store)[1][0].result
    assert result == NOT_FOUND
    assert store.get("alice", CONV, proposal_id).status == "pending"   # 未被改动


def test_commit_from_other_conversation_is_not_found(store):
    """提案同时绑定会话：换一个会话拿着同一个编号也提交不了。"""
    _, events = propose_claim_impl(_lodging(), ALICE, CONV, store)
    proposal_id = events[0].proposal_id
    assert commit_claim_impl(proposal_id, ALICE, "conv-2", store)[1][0].result == NOT_FOUND


def test_propose_and_commit_are_audited(store):
    _, events = propose_claim_impl(_lodging(), ALICE, CONV, store)
    proposal_id = events[0].proposal_id
    commit_claim_impl(proposal_id, ALICE, CONV, store)

    rows = store.audit_rows(proposal_id)
    assert [row["action"] for row in rows] == ["propose", "commit"]
    assert all(row["principal_id"] == "alice" and row["conversation_id"] == CONV for row in rows)


def test_rejected_proposal_is_audited(store):
    """被策略拒绝也要留痕——审计关注的是"有人尝试过什么"。"""
    propose_claim_impl(_lodging(amount=9999.0), ALICE, CONV, store)
    with store._connect() as conn:
        row = conn.execute("SELECT * FROM claim_audit WHERE result = ?",
                           (REJECTED_BY_POLICY,)).fetchone()
    assert row is not None and row["action"] == "propose"


# ---------- 渲染 ----------

def test_render_proposed_is_pending_and_shows_amount():
    claim = _lodging()
    event = ClaimEvent("propose", PROPOSED, "p_1", claim, (), False, "提案 p_1：住宿 1000.00 元")
    text, pending, card = render(event)
    assert pending is True
    assert "确认" in text
    assert card["amount"] == 1000.0 and card["proposal_id"] == "p_1"


def test_render_submitted_is_not_pending():
    event = ClaimEvent("commit", SUBMITTED, "p_1", _lodging(), (), False, "提案 p_1")
    text, pending, card = render(event)
    assert pending is False and "已提交" in text


def test_render_rejection_lists_reasons():
    event = ClaimEvent("propose", REJECTED_BY_POLICY, "", _lodging(),
                       ("住宿金额超出标准：一线城市 500 元/晚 × 2 晚 = 1000 元",))
    text, pending, card = render(event)
    assert pending is False
    assert "超出标准" in text
    assert card["violations"] and card["proposal_id"] == ""


def test_render_dual_sign_notice():
    event = ClaimEvent("propose", PROPOSED, "p_1", _lodging(amount=6000.0), (), True, "提案 p_1")
    text, _, _ = render(event)
    assert "双签" in text
