"""实现 LangChain VectorStore 接口、用 JSON 文件持久化的小型向量存储。"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def _cosine_relevance(score: float) -> float:
    """余弦相似度 -> [0,1] 相关度分数。

    LangChain 契约：relevance score 必须在 [0,1]，越大越相关。
    归一化向量点积即余弦，落在 [-1,1]，负值截断为 0。
    """
    return max(0.0, score)


INDEX_SCHEMA_VERSION = 2


class JSONVectorStore(VectorStore):
    """把 (record, vector) 持久化到 index.json 的极简向量存储。

    向量入库前已归一化，所以相似度 = 点积 = 余弦。
    基类提供 similarity_search_with_relevance_scores(score_threshold=...)，
    本类只需给出原始余弦分数和分数映射函数。

    record 结构：{"id", "text", **metadata}——metadata 随 Document 透传，
    检索方法支持 filter 参数（callable: record -> bool）做预过滤，
    未通过的记录不参与相似度计算与 top-k 竞争。
    """

    def __init__(
        self,
        embedding: Embeddings,
        records: list[dict],
        vectors: np.ndarray,
        index_metadata: dict[str, Any] | None = None,
    ):
        self._embedding = embedding
        self._records = records
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self.index_metadata = index_metadata or {}
        self._id_seq = len(self._records)  # add_texts 无显式 ids 时的单调序号，只增不减
        if len(self._records) != len(self._vectors):
            raise ValueError("Index records and vectors have different lengths")

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ---------- 持久化 ----------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(
                {
                    "schema_version": INDEX_SCHEMA_VERSION,
                    "metadata": self.index_metadata,
                    "records": self._records,
                    "vectors": self._vectors.tolist(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temp_path, path)

    @classmethod
    def load(cls, path: Path | str, embedding: Embeddings) -> "JSONVectorStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        schema_version = data.get("schema_version", 1)
        if schema_version > INDEX_SCHEMA_VERSION:
            raise ValueError(f"Unsupported index schema version: {schema_version}")
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        metadata = data.get("metadata", {})
        expected_dimension = metadata.get("vector_dimension")
        if expected_dimension and vectors.ndim == 2 and vectors.shape[1] != expected_dimension:
            raise ValueError("Index vector dimension does not match its metadata")
        return cls(embedding, data["records"], vectors, metadata)

    # ---------- 构造 ----------

    @classmethod
    def _build(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        ids: list[str] | None = None,
    ) -> "JSONVectorStore":
        texts = [doc.page_content for doc in documents]
        vectors = np.asarray(embedding.embed_documents(texts), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        records = []
        for i, (doc, text) in enumerate(zip(documents, texts)):
            identity = "|".join(
                str(doc.metadata.get(key, ""))
                for key in ("relative_path", "page", "para")
            ) + f"|{text}"
            record_id = ids[i] if ids else hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            records.append({"id": record_id, "text": text, **doc.metadata})
        model_name = getattr(embedding, "model_name", embedding.__class__.__name__)
        index_metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": model_name,
            "vector_dimension": int(vectors.shape[1]),
            "record_count": len(records),
        }
        return cls(embedding, records, vectors, index_metadata)

    @classmethod
    def from_documents(
        cls, documents: list[Document], embedding: Embeddings, *, path: Path | str, **kwargs: Any
    ) -> "JSONVectorStore":
        store = cls._build(documents, embedding)
        store.save(path)
        return store

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "JSONVectorStore":
        texts = list(texts)
        documents = [
            Document(page_content=text, metadata=dict(meta or {}))
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        store = cls._build(documents, embedding, ids)
        if "path" in kwargs:
            store.save(kwargs["path"])
        return store

    # ---------- VectorStore 接口 ----------

    def _next_ids(self, metadatas: list[dict]) -> list[str]:
        """生成不与现存记录冲突的 ID。

        序号只增不减，删除记录后不会复用已删除的编号；再与现存 ID 去重，
        防止与早前会话写入的计数器格式 ID 相撞。
        """
        used = {record["id"] for record in self._records}
        generated = []
        for meta in metadatas:
            while True:
                self._id_seq += 1
                candidate = f"{meta.get('source', '')}:{self._id_seq}"
                if candidate not in used:
                    break
            used.add(candidate)
            generated.append(candidate)
        return generated

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        metadatas = metadatas or [{}] * len(texts)
        if len(metadatas) != len(texts):
            raise ValueError("metadatas and texts must have the same length")
        ids = ids or self._next_ids(metadatas)
        vectors = np.asarray(self._embedding.embed_documents(texts), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        self._vectors = np.vstack([self._vectors, vectors]) if len(self._records) else vectors
        for record_id, text, meta in zip(ids, texts, metadatas):
            self._records.append({"id": record_id, "text": text, **meta})
        return list(ids)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if ids is None:
            self._records = []
            self._vectors = np.empty((0, self._vectors.shape[1]), dtype=np.float32)
            return True
        doomed = set(ids)
        keep = [i for i, r in enumerate(self._records) if r["id"] not in doomed]
        self._records = [self._records[i] for i in keep]
        self._vectors = self._vectors[keep]
        return True

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        if not self._records:
            return []
        filter_fn = kwargs.pop("filter", None)
        # 预过滤：未通过的记录直接排除，不参与相似度计算与 top-k 竞争
        allowed = [
            i for i, r in enumerate(self._records) if filter_fn is None or filter_fn(r)
        ]
        if not allowed:
            return []
        q = np.asarray(self._embedding.embed_query(query), dtype=np.float32)
        q /= np.linalg.norm(q).clip(min=1e-12)
        scores = self._vectors[allowed] @ q
        order = np.argsort(scores)[::-1][:k]
        return [(self._to_document(allowed[i]), float(scores[i])) for i in order]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return _cosine_relevance

    def _to_document(self, i: int) -> Document:
        record = self._records[i]
        return Document(
            page_content=record["text"],
            metadata={k: v for k, v in record.items() if k != "text"},
        )
