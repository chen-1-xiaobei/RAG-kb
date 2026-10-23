"""端到端回放测试：用录制好的固件确定性跑完整 agent 图。

这是项目 README「已知边界」里承认的那条空白的补丁——
"多轮对话的检索改写质量未纳入自动评测，因为那需要真实 LLM"。

现在第 2 轮模型写了什么查询被完整录在 cassette 里，于是可以断言；
而且整个过程**不需要密钥、不联网、不花一分钱**，可以直接进 CI。
"""
from pathlib import Path

import pytest

from cassette import CassetteMissing, CassetteModel, CassetteStore
from cassette_scenario import MODEL_NAME, run_scenario, run_turn

CASSETTE = Path(__file__).parent / "cassettes" / "multiturn_rewrite.json"


@pytest.fixture()
def replayer():
    if not CASSETTE.exists():
        pytest.skip(f"固件不存在：{CASSETTE}")
    store = CassetteStore(CASSETTE, "replay")
    return store, CassetteModel(store, model_name=MODEL_NAME)


def test_cassette_fixture_is_present_and_readable():
    assert CASSETTE.is_file(), "多轮回放依赖这份固件，不要误删"
    text = CASSETTE.read_text(encoding="utf-8")
    assert "报销" in text, "固件里应包含可读的中文，便于 review"
    assert "tool_calls" in text


def test_two_turn_replay_produces_rewritten_query(replayer):
    """第 2 轮「那住宿标准呢」必须被改写成不依赖上下文的完整查询。"""
    _, model = replayer
    first, second = run_scenario(model)

    # 第 1 轮：问题本身已经自洽，查询可以就是它的浓缩
    assert len(first.queries) == 1
    assert "报销" in first.queries[0]

    # 第 2 轮：这是重点
    assert len(second.queries) == 1
    rewritten = second.queries[0]
    assert rewritten != "那住宿标准呢", "不能把指代句直接丢给检索"
    assert "住宿" in rewritten and "标准" in rewritten, f"改写后应包含关键实体，实际：{rewritten}"


def test_replay_consumes_all_recorded_entries(replayer):
    """固件里的记录数应与实际调用次数一致，防止录制与场景漂移。"""
    store, model = replayer
    assert store.recorded_count == 4, "两轮 x（一次决策 + 一次收手）= 4"
    run_scenario(model)
    assert store._cursor and sum(store._cursor.values()) == 4


def test_replay_is_deterministic(replayer):
    """同一份固件跑两遍，结果必须逐字节一致——这是它能当回归测试的前提。"""
    store, model = replayer
    first_run = run_scenario(model)

    store.reset_cursor()
    second_run = run_scenario(model)

    assert first_run[0].queries == second_run[0].queries
    assert first_run[1].queries == second_run[1].queries


def test_unrecorded_question_fails_loudly(replayer):
    """没录过的问题必须报错，绝不能悄悄返回空结果让测试绿着过去。"""
    _, model = replayer
    with pytest.raises(CassetteMissing):
        run_turn(model, "从没录过的问题：公司的火星基地地址是什么？")
