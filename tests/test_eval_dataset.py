"""评测集自检：保证 dataset.json 不会因为改文档、改权限或手滑而静默失效。

评测集是"会腐烂的资产"——文档重命名、权限组调整、用例和来源对不上，
都会让指标变成看似漂亮的假数字。这里用断言把这类腐烂挡在提交前。
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 本文件只做静态数据校验：不 import evaluate（会连带拉起检索/模型链路），
# 指标函数的单测在 tests/test_eval_metrics.py。这样测试套件始终能离线秒跑。
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
    """反向检查：data/ 下新增了可索引文件却忘了登记到注册表，用例就会悄悄失效。

    这条只比"可索引后缀"与"声明了 indexed=true 的登记项"。声明为
    `indexed: false` 的文件（如 data/README.md、未被 loader 支持的 .xlsx）
    由下一条 `test_registry_accounts_for_every_file_in_data` 负责覆盖——
    两条各管一个方向，不要混在一起，否则新增一种"不索引"的文件类型就会误报。
    """
    registered = {entry["path"] for entry in indexed_entries(dataset)}
    indexable = {
        path.relative_to(DATA).as_posix()
        for path in DATA.rglob("*")
        if path.suffix.lower() in {".md", ".txt", ".pdf", ".docx", ".html", ".csv"}
    }
    # 明确声明"不索引"的文件里，只有后缀本身可索引的那部分才需要参与对齐。
    # 例如 data/README.md 是 .md（可索引但声明跳过）→ 参与；.xlsx 不在 loader
    # 支持列表里 → 与这一步无关，由 test_registry_accounts_for_every_file_in_data 覆盖。
    skipped = {
        entry["path"] for entry in dataset["sources"] if not entry.get("indexed", True)
    } & indexable
    assert indexable == registered | skipped, (
        f"未登记: {indexable - registered - skipped} / "
        f"已登记但磁盘上不存在: {registered - indexable}"
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


def test_evaluate_does_not_depend_on_llm_module():
    """评测脚本不得 import app —— app 需要 DEEPSEEK_API_KEY，而评测不需要 LLM。

    这是**静态代理断言**，不是行为验证：真实验证是 CI 的 eval 任务（在没有密钥的
    环境里跑 evaluate.py）。单测里不能真去 import evaluate，那会把 embedding 模型
    拖进快任务——所以这里只守住"依赖方向"这个最容易退化的点。
    背景：这条依赖曾让 CI 的评测任务因缺少 key 直接失败，而调用链上没有任何一处
    真正需要 LLM。
    """
    import ast

    tree = ast.parse((ROOT / "evaluate.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
    assert "app" not in imported, (
        "evaluate.py 又 import app 了：app 构造 LLM 客户端需要密钥，"
        "评测应当只依赖 retrieval"
    )
    assert "retrieval" in imported, "评测应通过 retrieval 使用真实检索管线"
