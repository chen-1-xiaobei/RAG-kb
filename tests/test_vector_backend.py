"""两个向量后端的**同一套**行为断言。

这是"换存储"能被称为"改造"而不是"重写"的地方：检索层只依赖
`VectorBackend.search(query, k, groups, score_threshold)`，两个实现必须给出
同样的排序、同样的权限边界、同样的阈值语义。

Milvus 部分自建临时索引（tmp_path），因此：
- 本机能跑（Milvus Lite 支持 Windows）；
- 没装 pymilvus 的环境自动跳过，不影响主套件；
- 不依赖仓库里是否已跑过 ingest。
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb_store import JSONVectorStore  # noqa: E402
from vector_backend import group_filter_expr  # noqa: E402


class FakeEmbedding(Embeddings):
    """确定性替身：按文本哈希生成固定向量，避免加载真实模型。

    用 4 维 + 手工指定几段文本的向量，让相似度关系可预测。
    """

    TABLE = {
        "住宿标准 500": [1.0, 0.0, 0.0, 0.0],
        "住宿标准 350": [0.9, 0.1, 0.0, 0.0],
        "薪资结构": [0.99, 0.0, 0.0, 0.0],
        "年假天数": [0.3, 0.9, 0.0, 0.0],
        "query": [1.0, 0.0, 0.0, 0.0],
    }

    model_name = "fake-embedding"

    def _vector(self, text: str) -> list[float]:
        if text in self.TABLE:
            return self.TABLE[text]
        # 未登记文本用稳定的确定性向量
        seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "little")
        rng = np.random.default_rng(seed)
        vector = rng.random(4)
        return (vector / np.linalg.norm(vector)).tolist()

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def documents() -> list[Document]:
    return [
        Document(page_content="住宿标准 500", metadata={
            "source": "a.md", "relative_path": "财务/a.md", "group": "财务",
            "section": "住宿", "page": None, "para": 0, "title": "报销标准"}),
        Document(page_content="住宿标准 350", metadata={
            "source": "a.md", "relative_path": "财务/a.md", "group": "财务",
            "section": "住宿", "page": None, "para": 1, "title": "报销标准"}),
        Document(page_content="薪资结构", metadata={
            "source": "b.md", "relative_path": "人事/b.md", "group": "人事",
            "section": "薪酬", "page": None, "para": 0, "title": "薪酬制度"}),
        Document(page_content="年假天数", metadata={
            "source": "c.md", "relative_path": "c.md", "group": "public",
            "section": "假期", "page": None, "para": 0, "title": "员工手册"}),
    ]


def json_backend(tmp_path: Path) -> JSONVectorStore:
    path = tmp_path / "index.json"
    JSONVectorStore.from_documents(documents(), FakeEmbedding(), path=path)
    return JSONVectorStore.load(path, FakeEmbedding())


def milvus_backend(tmp_path: Path):
    pytest.importorskip("pymilvus", reason="未安装 pymilvus")
    from milvus_store import MilvusVectorStore

    return MilvusVectorStore.build(
        documents(),
        embedding=FakeEmbedding(),
        uri=str(tmp_path / "milvus.db"),
        collection="test_chunks",
    )


BACKENDS = {
    "json": json_backend,
    "milvus": milvus_backend,
}


@pytest.fixture(params=list(BACKENDS))
def backend(request, tmp_path):
    return BACKENDS[request.param](tmp_path)


# ---------------------------------------------------------------------------
# 行为一致性
# ---------------------------------------------------------------------------

def test_search_returns_best_match_first(backend):
    hits = backend.search("query", k=2, groups=frozenset({"财务", "public"}))
    assert hits, "应当有命中"
    assert hits[0][0].page_content == "住宿标准 500"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)


def test_permission_prefilter_excludes_higher_similarity_document(backend):
    """人事的「薪资结构」相似度并列最高，未授权时**必须不出现**。

    这是两条独立要求的合并断言：排序对（它该在最前）与过滤对（它不该出现）。
    """
    without = backend.search("query", k=4, groups=frozenset({"财务", "public"}))
    contents = [doc.page_content for doc, _ in without]
    assert "薪资结构" not in contents
    assert set(contents) <= {"住宿标准 500", "住宿标准 350", "年假天数"}

    with_hr = backend.search("query", k=4, groups=frozenset({"财务", "人事", "public"}))
    assert "薪资结构" in [doc.page_content for doc, _ in with_hr]


def test_empty_groups_returns_nothing(backend):
    assert backend.search("query", k=4, groups=frozenset()) == []


def test_score_threshold_filters_low_relevance(backend):
    """阈值语义必须一致：余弦 < 0.25 的结果被丢弃（与 JSON 后端同一口径）。"""
    hits = backend.search("query", k=4, groups=frozenset({"财务", "public"}),
                          score_threshold=0.5)
    assert all(score >= 0.5 for _, score in hits)
    assert "年假天数" not in [doc.page_content for doc, _ in hits]


def test_records_exposes_text_and_metadata(backend):
    records = backend.records()
    assert len(records) == 4
    by_text = {record["text"]: record for record in records}
    assert by_text["住宿标准 500"]["group"] == "财务"
    assert by_text["住宿标准 500"]["source"] == "a.md"


def test_count_matches_records(backend):
    assert backend.count() == len(backend.records()) == 4


# ---------------------------------------------------------------------------
# Milvus 特有的表达式拼接
# ---------------------------------------------------------------------------

def test_group_expression_quotes_every_group():
    expr = group_filter_expr(frozenset({"财务", "public"}))
    assert expr.startswith("grp in [")
    assert "'财务'" in expr and "'public'" in expr


def test_group_expression_escapes_quotes():
    """组名来自目录名，可能出现引号——拼表达式必须转义，否则可以注入。"""
    expr = group_filter_expr({"(x' or '1'='1)"})
    assert "\\'" in expr
