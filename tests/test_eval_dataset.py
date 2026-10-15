"""评测集自检：保证 dataset.json 不会因为改文档、改权限或手滑而静默失效。

评测集是"会腐烂的资产"——文档重命名、权限组调整、用例和来源对不上，
都会让指标变成看似漂亮的假数字。这里用断言把这类腐烂挡在提交前。
"""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 只 import 纯函数模块：本文件是静态数据校验，不应把 embedding/reranker 拖进测试收集阶段
from eval_metrics import ndcg, reciprocal_rank  # noqa: E402

DATASET = ROOT / "evals" / "dataset.json"
DATA = ROOT / "data"
VALID_KINDS = {"retrieval", "table", "permission", "offtopic", "injection"}


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases(dataset):
    return dataset["cases"]


# ---------- 来源注册表 ----------

def indexed_entries(dataset):
    return [entry for entry in dataset["sources"] if entry.get("indexed", True)]


def test_source_registry_matches_data_dir(dataset):
    """登记为「已索引」的来源，其相对路径必须真实存在于 data/ 下。

    只校验文件名是不够的：同一文件名可以出现在不同权限组目录（如 data/财务/手册.md
    与 data/人事/手册.md），而检索结果只带文件名，来源级指标会把它们混为一谈。
    """
    for entry in indexed_entries(dataset):
        assert (DATA / entry["path"]).is_file(), f"来源不存在: {entry['path']}"
        assert (DATA / entry["path"]).name == entry["source"], (
            f"{entry['path']} 的文件名与 source 字段 {entry['source']} 不一致"
        )


def test_table_cases_target_registered_table(dataset):
    """表格用例必须针对真实存在、且对该权限组可见的表，SQL 必须是单表 SELECT。"""
    from tools import TableRegistry, _is_safe_select

    registry = TableRegistry()
    registry.load()
    for case in dataset["cases"]:
        if case["kind"] != "table":
            continue
        spec = case["table"]
        assert spec["name"] in registry.tables, f"{case['question']}: 表不存在 {spec['name']}"
        info = registry.tables[spec["name"]]
        allowed = set(case["groups"]) | {"public"}
        assert info.group in allowed, f"{case['question']}: 无权访问 {spec['name']}"
        assert _is_safe_select(spec["sql"], spec["name"]), (
            f"{case['question']}: SQL 未通过只读白名单校验: {spec['sql']}"
        )
        assert spec["expect_cells"], f"{case['question']}: 没有断言任何单元格"


def test_table_cases_are_not_mixed_with_vector_retrieval(cases):
    """表格数据不进向量索引，因此表格用例不得声明向量 required_sources。

    混在一起会让用例永远无法通过，指标变成噪声——这是真实踩过的坑。
    """
    for case in cases:
        if case["kind"] == "table":
            assert case["required_sources"], "表格用例仍需登记权威来源（用于权限审计）"
            assert "table" in case, "表格用例缺少 table 规格"
        else:
            assert "table" not in case, f"{case['question']} 非表格用例却带 table 规格"


def test_source_names_are_unique_across_groups(dataset):
    """文件名必须全局唯一：否则来源级召回/nDCG 无法区分两个不同组的同名文档。

    这是评测口径的硬约束（不是 data/ 的约束）——真出现同名文件时，正确做法是把
    检索结果的 relative_path 也纳入来源标识，而不是在评测里凑合。
    """
    names = [entry["source"] for entry in dataset["sources"]]
    assert len(names) == len(set(names)), "存在同名来源，来源级指标会混淆；见 docstring"


def test_data_dir_has_no_unregistered_indexable_file(dataset):
    """反向检查：data/ 下新增了可索引文件却忘了登记到注册表，用例就会悄悄失效。"""
    registered = {entry["path"] for entry in indexed_entries(dataset)}
    skip = {entry["path"] for entry in dataset["sources"] if not entry.get("indexed", True)}
    indexable = {
        path.relative_to(DATA).as_posix()
        for path in DATA.rglob("*")
        if path.suffix.lower() in {".md", ".txt", ".pdf", ".docx", ".html", ".csv"}
    }
    assert indexable == registered | skip, (
        f"未登记: {indexable - registered - skip} / 已登记但磁盘上不存在: {registered - indexable}"
    )


def test_registry_accounts_for_every_file_in_data(dataset):
    """注册表必须覆盖 data/ 下所有文件（含明确声明不索引的），别名遗漏无处可藏。"""
    on_disk = {path.relative_to(DATA).as_posix() for path in DATA.rglob("*") if path.is_file()}
    registered = {entry["path"] for entry in dataset["sources"]}
    assert on_disk <= registered, f"data/ 下有文件未出现在注册表: {on_disk - registered}"


def test_each_source_declares_group_and_reason(dataset):
    for entry in dataset["sources"]:
        assert entry["groups"], f"{entry['source']} 未声明权限组"
        assert all(isinstance(group, str) and group for group in entry["groups"])
        assert entry.get("reason"), f"{entry['source']} 缺少入选理由"


# ---------- 用例字段 ----------

def test_case_count_is_meaningful(cases):
    """指标必须建立在足够多的用例上：少于 30 条的集合指标没有统计意义。"""
    assert len(cases) >= 30
    kinds = {case["kind"] for case in cases}
    assert kinds == VALID_KINDS, f"缺少某类对抗用例: {VALID_KINDS - kinds}"


def test_case_fields(cases):
    for case in cases:
        assert case["question"].strip()
        assert isinstance(case["groups"], list)
        assert isinstance(case["required_sources"], list)
        assert case["kind"] in VALID_KINDS
        assert case.get("note"), f"用例缺少判定依据: {case['question']}"


def test_required_and_graded_sources_are_registered(cases, dataset):
    registered = {entry["source"] for entry in indexed_entries(dataset)}
    for case in cases:
        unknown = (set(case["required_sources"]) | set(case.get("relevant_sources") or {})) - registered
        assert not unknown, f"{case['question']} 引用了未登记来源: {unknown}"


def test_graded_relevance_keeps_required_sources_at_top(cases):
    """分级相关性衡量排序质量，而非召回门槛。

    required_sources 是"必须召回"；relevant_sources 可以额外包含确实相关但非必需的
    次要来源（例如流程文档对"住宿限额"只是背景）——两者是不同的语义，不能混用。
    这里只保证：主来源等级为 2、每个必须召回的来源都被纳入相关性评估。
    """
    for case in cases:
        graded = case.get("relevant_sources")
        if not graded:
            continue
        assert all(grade in (1, 2) for grade in graded.values()), case["question"]
        assert 2 in graded.values(), f"{case['question']} 缺少等级 2 的主来源，分级无意义"
        primary = {source for source, grade in graded.items() if grade == 2}
        missing = set(case["required_sources"]) - set(graded)
        assert not missing, f"{case['question']} 的必须来源 {missing} 未纳入相关性评估"
        assert primary & set(case["required_sources"]), f"{case['question']} 主来源不在必须来源内"


# ---------- 语义一致性：这类断言才是真正防腐烂的部分 ----------

def test_negative_cases_expect_nothing(cases):
    for case in cases:
        if case["kind"] in {"permission", "offtopic"}:
            assert case["required_sources"] == [], f"{case['question']} 是负例却声明了必须来源"
            assert not case.get("relevant_sources"), f"{case['question']} 是负例却给了相关性权重"


def test_cases_cannot_expect_content_they_cannot_access(cases, dataset):
    """必须来源与分级来源都必须对用例声明的权限组可见，否则指标永远达不到 100% 且原因隐蔽。"""
    group_of = {entry["source"]: entry["groups"][0] for entry in indexed_entries(dataset)}
    for case in cases:
        allowed = set(case["groups"]) | {"public"}
        wanted = set(case["required_sources"]) | set(case.get("relevant_sources") or {})
        for source in wanted:
            assert group_of[source] in allowed, (
                f"{case['question']}: 引用 {source}（{group_of[source]} 组）但用户组是 {case['groups']}"
            )


def test_injection_cases_carry_a_canary(cases):
    for case in cases:
        if case["kind"] == "injection":
            assert case.get("injection_canary"), f"{case['question']} 缺少注入标记，形同虚设"


def test_no_duplicate_question_and_groups(cases):
    """同一问题可以用不同权限组重复出现（权限对比），但不能完全重复。"""
    seen = set()
    for case in cases:
        key = (case["question"], tuple(case["groups"]))
        assert key not in seen, f"用例重复: {key}"
        seen.add(key)


def test_every_indexed_source_is_covered_by_some_case(dataset, cases):
    """每个被索引的来源至少被一条用例期望召回，否则该文档的召回/权限质量无人验证。"""
    covered = {source for case in cases for source in case["required_sources"]}
    missing = {entry["source"] for entry in indexed_entries(dataset)} - covered
    assert not missing, f"以下来源没有任何用例覆盖: {missing}"


# ---------- 指标函数本身 ----------

def test_ndcg_rewards_top_rank():
    """主来源排第一位得满分；主次排反要明确掉分（这正是分级相关性的意义）。"""
    grades = {"a": 2, "b": 1}
    assert ndcg(["a", "b"], grades) == pytest.approx(1.0)
    # ideal DCG = 2/log2(2) + 1/log2(3) = 2.6309；actual = 1/1 + 2/1.585 = 2.2619
    assert ndcg(["b", "a"], grades) == pytest.approx(2.2619 / 2.6309, abs=1e-4)
    assert ndcg(["b", "a"], grades) < ndcg(["a", "b"], grades)
    assert ndcg(["c"], grades) == 0.0


def test_ndcg_handles_single_relevant_source():
    """单来源用例：排第 1 得 1.0，排第 2 明显掉分，排在 k 之外得 0。"""
    assert ndcg(["x"], {"x": 2}) == pytest.approx(1.0)
    assert ndcg(["y", "x"], {"x": 2}) == pytest.approx(1 / math.log2(3), abs=1e-6)
    assert ndcg([f"filler{i}" for i in range(10)], {"x": 2}) == 0.0


def test_ndcg_is_bounded_and_monotonic():
    grades = {"a": 2}
    scores = [ndcg(["a"] + ["f"] * rank, grades) for rank in range(4)]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"b"}) == 0.0
