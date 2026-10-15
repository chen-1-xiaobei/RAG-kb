"""rerank 精排逻辑单测（stub 模型，不加载真实 2GB 模型）。"""
import pytest

import rerank


class StubModel:
    """记录 predict 被调用的次数与规模，用于验证缓存真的省掉了计算。

    同时校验入参契约：真 CrossEncoder 只接受 (str, str)，喂 Document 会抛
    ValueError，而调用方会把异常降级成"未检索到相关文档"——表现为静默检索失效。
    这个断言就是为了让那类 bug 在单测里立刻暴露。
    """

    def __init__(self, scores):
        self._scores = scores
        self.calls = 0
        self.pairs_seen = 0

    def predict(self, pairs, batch_size=8):
        for pair in pairs:
            assert isinstance(pair, tuple) and len(pair) == 2, f"predict 入参应为二元组: {pair!r}"
            assert all(isinstance(item, str) for item in pair), (
                f"predict 只接受字符串对，收到 {[type(item).__name__ for item in pair]}"
            )
        self.calls += 1
        self.pairs_seen += len(pairs)
        return self._scores[: len(pairs)]


class FakeDoc:
    def __init__(self, text, doc_id=None):
        self.page_content = text
        self.metadata = {"id": doc_id if doc_id is not None else text}


@pytest.fixture(autouse=True)
def clean_cache():
    """每条用例都从空缓存开始：缓存是模块级全局状态，否则用例相互污染。"""
    rerank.clear_cache()
    yield
    rerank.clear_cache()


def test_rerank_sorts_desc_and_truncates(monkeypatch):
    monkeypatch.setattr(rerank, "get_model", lambda: StubModel([1.0, 5.0, 2.0]))
    hits = [(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8), (FakeDoc("c"), 0.7)]
    out = rerank.rerank("q", hits, top_n=2, threshold=0.5)
    assert [doc.page_content for doc, _ in out] == ["b", "c"]
    assert [score for _, score in out] == sorted([score for _, score in out], reverse=True)


def test_rerank_filters_below_threshold(monkeypatch):
    monkeypatch.setattr(rerank, "get_model", lambda: StubModel([-2.0, -3.0]))
    hits = [(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8)]
    assert rerank.rerank("q", hits, top_n=4, threshold=0.5) == []


def test_rerank_empty_input_is_noop():
    assert rerank.rerank("q", [], top_n=4, threshold=0.5) == []


# ---------- 逐对结果缓存 ----------

def test_cache_avoids_recomputing_same_pairs(monkeypatch):
    """同一批候选重复精排（agent「换词重试」时的常见情形）不应再进模型。"""
    model = StubModel([3.0, 1.0])
    monkeypatch.setattr(rerank, "get_model", lambda: model)
    hits = [(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8)]

    first = rerank.rerank("q", hits, top_n=2, threshold=0.0)
    assert model.calls == 1 and model.pairs_seen == 2

    second = rerank.rerank("q", hits, top_n=2, threshold=0.0)
    assert model.calls == 1, "第二次应全部命中缓存"
    assert [(doc.page_content, score) for doc, score in second] == [
        (doc.page_content, score) for doc, score in first
    ]


def test_cache_recomputes_only_missing_pairs(monkeypatch):
    """部分命中时只把未命中的候选送进模型，且分数按原顺序对齐。"""
    model = StubModel([2.0, 1.0])
    monkeypatch.setattr(rerank, "get_model", lambda: model)
    rerank.rerank("q", [(FakeDoc("a"), 0.9)], top_n=1, threshold=0.0)

    model2 = StubModel([9.0])
    monkeypatch.setattr(rerank, "get_model", lambda: model2)
    out = rerank.rerank("q", [(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8)], top_n=2, threshold=0.0)
    assert model2.pairs_seen == 1, "只有 b 需要重新计算"
    assert [score for _, score in out] == [9.0, 2.0]


def test_cache_key_depends_on_question_and_content(monkeypatch):
    """换问题、或同一 doc_id 内容变了（索引重建），都必须重算而不是吃陈旧分数。"""
    model = StubModel([1.0])
    monkeypatch.setattr(rerank, "get_model", lambda: model)

    rerank.rerank("问题一", [(FakeDoc("正文", doc_id="d1"), 0.9)], top_n=1, threshold=0.0)
    assert model.calls == 1

    rerank.rerank("问题二", [(FakeDoc("正文", doc_id="d1"), 0.9)], top_n=1, threshold=0.0)
    assert model.calls == 2, "换了问题必须重算"

    rerank.rerank("问题二", [(FakeDoc("正文已改", doc_id="d1"), 0.9)], top_n=1, threshold=0.0)
    assert model.calls == 3, "同 id 但内容变化必须重算"


def test_cache_does_not_change_results(monkeypatch):
    """缓存只应省计算，不能改变排序与阈值过滤结果。"""
    model = StubModel([1.0, 5.0, 2.0])
    monkeypatch.setattr(rerank, "get_model", lambda: model)
    hits = [(FakeDoc("a"), 0.9), (FakeDoc("b"), 0.8), (FakeDoc("c"), 0.7)]

    cold = rerank.rerank("q", hits, top_n=2, threshold=0.5)
    warm = rerank.rerank("q", hits, top_n=2, threshold=0.5)
    assert [(doc.page_content, score) for doc, score in cold] == [
        (doc.page_content, score) for doc, score in warm
    ]


def test_cache_evicts_oldest_beyond_limit(monkeypatch):
    """超过上限时淘汰最久未使用项，缓存不会无限增长。"""
    monkeypatch.setattr(rerank, "_MAX_CACHE_ENTRIES", 2)
    model = StubModel([1.0])
    monkeypatch.setattr(rerank, "get_model", lambda: model)

    for name in ("a", "b", "c"):
        rerank.rerank("q", [(FakeDoc(name), 0.9)], top_n=1, threshold=0.0)
    assert rerank.cache_stats()["entries"] == 2

    rerank.rerank("q", [(FakeDoc("c"), 0.9)], top_n=1, threshold=0.0)
    assert model.calls == 3, "c 仍在缓存中，不应重算"
    rerank.rerank("q", [(FakeDoc("a"), 0.9)], top_n=1, threshold=0.0)
    assert model.calls == 4, "a 已被淘汰，应重算"
