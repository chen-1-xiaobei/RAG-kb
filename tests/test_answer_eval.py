"""答案层评测的数据集自检（离线，不加载模型、不调用 LLM）。

这里测的是"评测集本身是否可信"，不是模型表现。答案层评测最容易出的问题是
假阳性——把正确答案判错——所以下面有几条断言专门盯这个。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate_answers  # noqa: E402  只导入模块，不会拉起 app/模型

CASES = ROOT / "evals" / "answers.json"
FIXTURE = ROOT / "evals" / "answers.fixture.json"
DATASET = ROOT / "evals" / "dataset.json"


def load_cases() -> list[dict]:
    return json.loads(CASES.read_text(encoding="utf-8"))["cases"]


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_case_ids_are_unique_and_well_formed():
    cases = load_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "用例 id 重复"
    for case in cases:
        assert case["expect"] in {"answer", "refuse"}, case["id"]
        assert case.get("note"), f"{case['id']} 缺少依据说明，容易出成知识库里没有答案的题"


def test_refusal_cases_expect_no_citations():
    for case in load_cases():
        if case["expect"] == "refuse":
            assert case["required_sources"] == [], f"{case['id']} 拒答用例不应要求引用"
            assert case["must_contain"] == [], f"{case['id']} 拒答用例不应要求出现事实"


def test_answer_cases_declare_required_sources_and_facts():
    for case in load_cases():
        if case["expect"] == "answer":
            assert case["required_sources"], f"{case['id']} 未声明允许引用的来源"
            assert case["must_contain"], f"{case['id']} 未声明必须出现的事实"


def test_forbidden_facts_are_statements_not_bare_words():
    """禁词必须是**错误断言**，不能是孤立词。

    踩过的坑：a06 曾把「三倍」列为禁词，而正确回答是「休息日两倍（法定节假日三倍）」——
    模型答对了却被判错。同一段文档里正确与错误的数字常常并存，孤立词无法区分
    它们在讲哪一档，因此这里要求禁词至少是一个短语（≥4 字）而不是单词或数字。
    """
    for case in load_cases():
        for forbidden in case["must_not_contain"]:
            assert len(forbidden) >= 4, (
                f"{case['id']} 的禁词「{forbidden}」太短，容易误伤正确回答；"
                "请写成完整的错误断言"
            )
            assert not forbidden.replace(".", "").isdigit(), f"{case['id']} 的禁词是纯数字"


def test_required_sources_exist_and_are_visible_to_case_groups():
    """期望来源必须真实存在，且对该用例声明的权限组可见——否则是出题错误。"""
    registry = {
        entry["source"]: entry
        for entry in json.loads(DATASET.read_text(encoding="utf-8"))["sources"]
    }
    for case in load_cases():
        for name in case["required_sources"]:
            assert name in registry, f"{case['id']} 引用了未登记的来源 {name}"
            entry = registry[name]
            assert entry.get("indexed", True), f"{case['id']} 引用的来源 {name} 未参与索引"
            visible = set(entry["groups"]) & {"public", *case["groups"]}
            assert visible, f"{case['id']} 的权限组 {case['groups']} 看不到 {name}"


def test_fixture_covers_every_answer_case():
    fixture = load_fixture()
    recorded = {entry["id"] for entry in fixture["drafts"]}
    answer_cases = {case["id"] for case in load_cases() if case["expect"] == "answer"}
    assert answer_cases <= recorded, (
        f"以下用例缺少录制草稿: {answer_cases - recorded}；运行 python evaluate_answers.py --record"
    )


def test_fixture_drafts_have_expected_shape():
    for entry in load_fixture()["drafts"]:
        draft = entry["draft"]
        assert isinstance(draft["answer"], str)
        assert all(isinstance(n, int) and n >= 1 for n in draft["citations"])
        assert isinstance(draft["insufficient_context"], bool)


def test_missing_draft_is_reported_not_silently_passed():
    """没有草稿的 answer 用例必须被记入 missing，而不是当成通过。

    这里注入桩：不建索引、不导入 app。原来的写法在 CI 上直接
    FileNotFoundError（单测任务不建索引），属于"本地绿、CI 红"。
    """
    cases = [{"id": "x", "question": "q", "groups": ["public"], "expect": "answer",
              "required_sources": [], "must_contain": [], "must_not_contain": []}]
    called = {"retrieve": 0}

    def fake_retrieve(question, k, principal):
        called["retrieve"] += 1
        return []

    results, missing = evaluate_answers.evaluate(
        cases, {}, retrieve_fn=fake_retrieve, finalize_fn=lambda *args: None
    )
    assert missing == ["x"]
    assert results == []
    assert called["retrieve"] == 0, "缺固件时不应白跑一次检索"


def test_summarize_counts_each_metric():
    results = [
        {"expect": "answer", "behavior_ok": True, "citation_ok": True, "fact_ok": True},
        {"expect": "answer", "behavior_ok": True, "citation_ok": False, "fact_ok": True},
        {"expect": "refuse", "behavior_ok": True, "citation_ok": True, "fact_ok": False},
    ]
    metrics = evaluate_answers.summarize(results)
    assert metrics["cases"] == 3
    assert metrics["behavior_accuracy"] == 1.0
    assert abs(metrics["citation_precision"] - 2 / 3) < 1e-9
    assert abs(metrics["fact_presence"] - 2 / 3) < 1e-9
    assert abs(metrics["fully_correct"] - 1 / 3) < 1e-9
