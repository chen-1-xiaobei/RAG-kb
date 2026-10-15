import json

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from kb_store import INDEX_SCHEMA_VERSION, JSONVectorStore


class FakeEmbeddings(Embeddings):
    model_name = "fake-test-model"

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    @staticmethod
    def _embed(text):
        return [float("finance" in text), float("people" in text), 1.0]


def test_filter_happens_before_top_k(tmp_path):
    store = JSONVectorStore.from_texts(
        ["finance private", "people public"],
        FakeEmbeddings(),
        metadatas=[
            {"source": "a.md", "group": "finance", "relative_path": "finance/a.md"},
            {"source": "b.md", "group": "public", "relative_path": "b.md"},
        ],
        path=tmp_path / "index.json",
    )
    hits = store.similarity_search_with_score(
        "finance", k=1, filter=lambda record: record["group"] == "public"
    )
    assert len(hits) == 1
    assert hits[0][0].metadata["group"] == "public"


def test_index_is_versioned_and_loadable(tmp_path):
    path = tmp_path / "index.json"
    store = JSONVectorStore.from_texts(
        ["people handbook"],
        FakeEmbeddings(),
        metadatas=[{"source": "handbook.md", "relative_path": "hr/handbook.md", "group": "hr"}],
        path=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == INDEX_SCHEMA_VERSION
    assert payload["metadata"]["vector_dimension"] == 3
    loaded = JSONVectorStore.load(path, FakeEmbeddings())
    assert np.allclose(loaded._vectors, store._vectors)
    assert not path.with_suffix(".json.tmp").exists()


def test_ids_are_stable(tmp_path):
    kwargs = {
        "texts": ["same content"],
        "embedding": FakeEmbeddings(),
        "metadatas": [{"source": "a.md", "relative_path": "x/a.md", "page": 1, "para": 0}],
    }
    first = JSONVectorStore.from_texts(**kwargs)
    second = JSONVectorStore.from_texts(**kwargs)
    assert first._records[0]["id"] == second._records[0]["id"]


def test_delete_then_add_reuses_no_ids():
    store = JSONVectorStore.from_texts(
        ["finance private"],
        FakeEmbeddings(),
        metadatas=[{"source": "a.md", "relative_path": "finance/a.md", "group": "finance"}],
    )
    removed = store.add_texts(["people one"], [{"source": "b.md"}])
    store.delete(removed)
    added = store.add_texts(["people two"], [{"source": "b.md"}])
    assert set(added).isdisjoint(removed)  # 删除后新加记录不复用已删除的编号
    ids = [record["id"] for record in store._records]
    assert len(ids) == len(set(ids))


def test_add_texts_rejects_metadatas_length_mismatch():
    store = JSONVectorStore.from_texts(["seed"], FakeEmbeddings(), metadatas=[{"source": "a.md"}])
    with pytest.raises(ValueError):
        store.add_texts(["one", "two"], [{"source": "b.md"}])

