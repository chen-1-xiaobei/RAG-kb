"""GraphRAG 隔离实验的离线单测：只测实验自身的机制，不碰模型与网络。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # tests/ -> 仓库根目录
EXPERIMENT = ROOT / "experiments" / "graphrag"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT))

from langchain_core.documents import Document  # noqa: E402

import run_experiment as exp  # noqa: E402


def doc(source: str, text: str = "x") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_coverage_counts_only_required_hits():
    assert exp.coverage(["a.md", "b.md"], ["a.md", "b.md"]) == 1.0
    assert exp.coverage(["a.md"], ["a.md", "b.md"]) == 0.5
    assert exp.coverage(["x.md"], ["a.md"]) == 0.0


def test_best_per_document_keeps_highest_scoring_chunk():
    """同一份文档的多个 chunk 只算一次，且取分数最高的那次。"""
    ranked = exp.best_per_document([
        (doc("a.md"), 0.2),
        (doc("a.md"), 0.9),
        (doc("b.md"), 0.5),
    ])
    assert ranked == ["a.md", "b.md"], "应按最高分排序，且每份文档只出现一次"


def test_bigram_matching_handles_longer_entities():
    """实体比问题长时也要能命中——第一版就是因为方向反了，5 个问题漏了 4 个。"""
    index = {"五个工作日": [0], "报销流程": [1]}
    matched = exp.match_entities("纸质发票要几个工作日交？", index)
    assert "五个工作日" in matched, "「五个工作日」应通过「工作/作日」二元组命中"
    assert "报销流程" not in matched


def test_exact_substring_still_matches():
    index = {"报销流程": [0]}
    assert exp.match_entities("报销流程是什么", index) == ["报销流程"]


def test_graph_candidates_respects_cap():
    """候选必须封顶，否则精排会退化成几十秒一次。"""
    big_index = {f"实体{i}": list(range(i, i + 20)) for i in range(50)}
    store = type("S", (), {"_records": [{"id": str(i)} for i in range(500)]})()
    positions, matched = exp.graph_candidates(
        "实体1 实体2", big_index, {str(i): [] for i in range(500)}, store
    )
    assert matched, "应当命中实体"
    assert len(positions) <= exp.MAX_CANDIDATES


def test_questions_have_required_sources():
    payload = exp.load(exp.HERE / "questions.json")
    for case in payload["cases"]:
        assert case["required_sources"], f"{case['id']} 没有要求文档，测不出覆盖率"
        assert len(case["required_sources"]) >= 2, (
            f"{case['id']} 只要求 {len(case['required_sources'])} 份文档，不构成聚合型问题"
        )


def test_entity_index_is_committed_and_well_formed():
    payload = exp.load(exp.HERE / "entities.json")
    assert payload["chunks"], "实体索引为空"
    assert payload["failures"] == 0, "存在抽取失败，索引不完整"
    for entry in payload["chunks"]:
        for entity in entry["entities"]:
            assert 2 <= len(entity) <= 8, f"实体长度异常: {entity}"
