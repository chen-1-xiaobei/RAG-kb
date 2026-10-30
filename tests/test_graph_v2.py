"""graph_v2 的离线单测：PPR 扩散、文档配额、查询实体缓存。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / "experiments" / "graphrag"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT))

import graph_v2  # noqa: E402
import keywords  # noqa: E402
import run_experiment as exp  # noqa: E402


class FakeStore:
    def __init__(self, records):
        self._records = records

    def _to_document(self, index):
        from langchain_core.documents import Document

        record = self._records[index]
        return Document(page_content="x", metadata={"source": record["source"]})


def test_ppr_ranks_seeded_chunks_above_unrelated_ones():
    store = FakeStore([{"id": "c1", "source": "a.md"}, {"id": "c2", "source": "b.md"}])
    graph = graph_v2.build_bipartite(store, {"c1": ["报销"], "c2": ["请假"]})

    scores = graph_v2.personalized_pagerank(graph, ["报销"])
    assert scores.get(0, 0.0) > 0, "种子实体挂的 chunk 必须有权重"
    assert scores.get(1, 0.0) == 0, "完全无关的 chunk 不应分到权重"


def test_ppr_rewards_chunks_bridging_multiple_seeds():
    """同时连到多个种子实体的 chunk 得分应更高——这正是聚合型问题需要的桥接信号。"""
    store = FakeStore([
        {"id": "c1", "source": "a.md"},
        {"id": "c2", "source": "b.md"},
    ])
    graph = graph_v2.build_bipartite(store, {"c1": ["报销", "审批"], "c2": ["报销"]})
    scores = graph_v2.personalized_pagerank(graph, ["报销", "审批"])
    assert scores[0] > scores[1]


def test_ppr_without_matching_seed_returns_nothing():
    store = FakeStore([{"id": "c1", "source": "a.md"}])
    graph = graph_v2.build_bipartite(store, {"c1": ["报销"]})
    assert graph_v2.personalized_pagerank(graph, ["不存在的实体"]) == {}


def test_document_quota_caps_candidates_per_source():
    """旧实现里《社会保险》一个文档占了 12/28 个候选位，配额就是治这个。"""
    store = FakeStore([{"id": f"c{i}", "source": "same.md"} for i in range(10)])
    kept = graph_v2.apply_document_quota(store, list(range(10)), per_doc_cap=2, limit=8)
    assert kept == [0, 1], "同一文档最多占 2 个名额"


def test_document_quota_keeps_diversity_across_sources():
    store = FakeStore(
        [{"id": "a1", "source": "a.md"}, {"id": "a2", "source": "a.md"},
         {"id": "b1", "source": "b.md"}]
    )
    kept = graph_v2.apply_document_quota(store, [0, 1, 2], per_doc_cap=1, limit=8)
    assert kept == [0, 2]


def test_query_keyword_cache_covers_every_question():
    cache = keywords.load_cache()
    questions = exp.load(EXPERIMENT / "questions.json")["cases"]
    for case in questions:
        assert case["id"] in cache, f"{case['id']} 缺少查询实体缓存"
        assert cache[case["id"]]["selected"], f"{case['id']} 没选到任何实体"


def test_query_keyword_selections_exist_in_entity_vocabulary():
    """LLM 选出的实体必须真的在实体表里，否则等于凭空造词。"""
    entities = exp.load(EXPERIMENT / "entities.json")
    vocabulary = {entity for chunk in entities["chunks"] for entity in chunk["entities"]}
    cache = keywords.load_cache()
    for qid, entry in cache.items():
        for selected in entry["selected"]:
            assert selected in vocabulary, f"{qid} 选出了不存在的实体: {selected}"
