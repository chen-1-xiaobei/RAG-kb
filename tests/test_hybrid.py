"""混合检索的离线单测：分词、BM25 打分、RRF 融合、权限过滤。

不加载任何模型，纯 CPU 毫秒级跑完。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from hybrid import BM25Index, bm25_search, rrf_fuse, tokenize


def test_tokenize_keeps_chinese_bigrams_and_ascii_words():
    tokens = tokenize("报销流程 OA 系统")
    assert "报" in tokens and "销" in tokens
    assert "报销" in tokens            # 二元组
    assert "oa" in tokens              # 英文整词，转小写


def test_tokenize_drops_punctuation():
    tokens = tokenize("报销？流程！")
    assert "？" not in tokens and "！" not in tokens


def test_bm25_prefers_document_with_rare_term():
    """稀有词命中应该压过常见词命中——这正是补向量召回短板的地方。"""
    index = BM25Index([
        "员工在费用发生后三十天内提交报销单",
        "年假未使用部分不折现",
        "公司实行弹性工作制，核心工作时间为上午十点到下午四点",
    ])
    scores = index.scores("没休完能折现吗")
    assert scores[1] > scores[0] and scores[1] > scores[2]


def test_bm25_returns_zero_without_overlap():
    index = BM25Index(["报销流程", "考勤制度"])
    assert index.scores("完全不相关的英文 query") == [0.0, 0.0]


def test_rrf_fuses_by_rank_not_by_raw_score():
    """两路分数尺度不同（余弦 0-1、BM25 无上界），RRF 只看名次。"""
    doc_a = Document(page_content="A", metadata={"id": "a"})
    doc_b = Document(page_content="B", metadata={"id": "b"})
    fused = rrf_fuse([[(doc_a, 0.9)], [(doc_b, 999.0)]], top_k=2)
    assert {doc.metadata["id"] for doc, _ in fused} == {"a", "b"}


def test_rrf_rewards_documents_hit_by_both_channels():
    both = Document(page_content="both", metadata={"id": "both"})
    only = Document(page_content="only", metadata={"id": "only"})
    fused = rrf_fuse(
        [[(both, 0.5), (only, 0.4)], [(both, 3.0)]],
        top_k=2,
    )
    assert fused[0][0].metadata["id"] == "both", "两路都命中的应排第一"


def test_rrf_respects_top_k():
    docs = [Document(page_content=str(i), metadata={"id": str(i)}) for i in range(10)]
    fused = rrf_fuse([[(doc, 1.0) for doc in docs]], top_k=3)
    assert len(fused) == 3


class _FakeStore:
    """最小替身：只提供 bm25_search 需要的那几个属性。"""

    def __init__(self, texts, groups):
        self._records = [
            {"id": f"d{i}", "text": text, "group": group}
            for i, (text, group) in enumerate(zip(texts, groups))
        ]

    def _to_document(self, index):
        record = self._records[index]
        return Document(
            page_content=record["text"],
            metadata={k: v for k, v in record.items() if k != "text"},
        )


def test_bm25_search_respects_permission_filter():
    """未授权组的文档不得出现在结果里——与向量检索的预过滤同一要求。"""
    store = _FakeStore(
        ["住宿报销标准五百元", "住宿报销标准是财务内部规定"],
        ["public", "财务"],
    )
    results = bm25_search(
        store, "住宿报销标准", k=10, filter_fn=lambda r: r["group"] == "public"
    )
    assert [doc.metadata["group"] for doc, _ in results] == ["public"]


def test_bm25_search_skips_zero_score_documents():
    store = _FakeStore(["报销流程", "考勤制度"], ["public", "public"])
    results = bm25_search(store, "完全不相关", k=10, filter_fn=None)
    assert results == []
