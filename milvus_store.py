"""Milvus 向量后端（本地文件即 Milvus Lite，填 http:// 则连独立服务）。

为什么字段这样设计：

- `id` / `vector` / `text` / `grp` / `meta` 五个字段，**不把 metadata 摊平成列**。
  元数据（source、relative_path、section、page…）在切分策略变化时会增删，
  摊平成列意味着每次都要改 schema 并重建 collection；塞进一个 JSON 字段则不用。
- 但 `grp` 必须**单独成列**：权限过滤发生在检索**之前**（预过滤），
  Milvus 的过滤表达式只能引用字段，不能解析 JSON 里的键。
- 过滤条件是拼出来的表达式，因此组名要转义——和 SQL 注入是同一类问题。

一个刻意的取舍：`score_threshold` 在 Python 侧过滤，而不是用 Milvus 的
range search（radius/range_filter）。因为本项目的阈值语义是"余弦 ≥ 0.25"，
两边的比较口径必须完全一致，否则评测换后端就不可比；候选数只有几十条，
多一次列表过滤的成本可以忽略。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from vector_backend import BackendConfigError, VectorBackendError, group_filter_expr

ID_FIELD = "id"
VECTOR_FIELD = "vector"
TEXT_FIELD = "text"
GROUP_FIELD = "grp"
META_FIELD = "meta"

MAX_TEXT_LENGTH = 65535
MAX_META_LENGTH = 8192
MAX_ID_LENGTH = 128
MAX_GROUP_LENGTH = 64
# 元数据表后缀：存"索引是用哪个 embedding 模型建的、多少条、什么距离度量"。
# 为什么不放本地旁车文件：服务模式（MILVUS_URI=http://…）下没有本地文件，
# 多副本也各自一份，校验会永远跳过——换模型后索引不兼容却查不出来。
META_SUFFIX = "__meta"
META_KEY_FIELD = "meta_key"
META_VALUE_FIELD = "meta_value"
META_DUMMY_FIELD = "dummy"       # Milvus 要求集合必须有向量字段，1 维哑向量不参与检索


def _is_local(uri: str) -> bool:
    return not uri.startswith(("http://", "https://", "unix:"))


def meta_path_for(uri: str) -> Path:
    """旧版本的本地旁车文件路径（仅作为读取时的兼容回退）。"""
    return Path(f"{uri}.meta.json")


class MilvusVectorStore:
    """把 chunk 存进 Milvus，检索时用表达式做组预过滤。"""

    def __init__(self, embedding, client, collection: str, metric: str,
                 index_metadata: dict[str, Any]):
        self._embedding = embedding
        self._client = client
        self._collection = collection
        self.metric = metric
        self.index_metadata = index_metadata or {}

    @property
    def embeddings(self):
        return self._embedding

    # ---------- 打开 / 构建 ----------

    @classmethod
    def _connect(cls, uri: str):
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可操作提示
            raise VectorBackendError(
                "未安装 pymilvus：pip install pymilvus milvus-lite"
            ) from exc
        if _is_local(uri):
            path = Path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)
        return MilvusClient(uri)

    @classmethod
    def open(cls, *, embedding, uri: str, collection: str,
             metric: str = "COSINE") -> "MilvusVectorStore":
        client = cls._connect(uri)
        if not client.has_collection(collection):
            raise FileNotFoundError(
                f"Milvus collection '{collection}' 不存在（uri={uri}）：请先运行 python ingest.py"
            )
        # 重新打开本地库时 collection 会处于 released 状态，必须先 load 才能检索。
        # 这不是可选的优化——不调用会直接报 "call load() before search/get/query"。
        client.load_collection(collection)
        metadata = cls._read_meta(client, collection, uri)
        model_name = getattr(embedding, "model_name", None)
        indexed_model = metadata.get("embedding_model")
        if indexed_model and model_name and indexed_model != model_name:
            raise BackendConfigError(
                f"索引使用 {indexed_model}，当前配置是 {model_name}"
            )
        return cls(embedding, client, collection, metric, metadata)

    @classmethod
    def build(cls, documents: list[Document], *, embedding, uri: str,
              collection: str, metric: str = "COSINE") -> "MilvusVectorStore":
        client = cls._connect(uri)
        if client.has_collection(collection):
            client.drop_collection(collection)

        texts = [doc.page_content for doc in documents]
        vectors = embedding.embed_documents(texts)
        dimension = len(vectors[0])
        cls._create_collection(client, collection, dimension, metric)

        rows = []
        for index, (doc, text, vector) in enumerate(zip(documents, texts, vectors)):
            metadata = dict(doc.metadata)
            identity = "|".join(
                str(metadata.get(key, "")) for key in ("relative_path", "page", "para")
            ) + f"|{text}"
            import hashlib

            row_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            rows.append({
                ID_FIELD: row_id,
                VECTOR_FIELD: list(vector),
                TEXT_FIELD: text[:MAX_TEXT_LENGTH],
                GROUP_FIELD: str(metadata.get("group", "public")),
                META_FIELD: json.dumps(metadata, ensure_ascii=False)[:MAX_META_LENGTH],
            })

        # 分批插入：Milvus 单次插入有大小上限，且分批便于观察进度
        batch = 512
        for start in range(0, len(rows), batch):
            client.insert(collection_name=collection, data=rows[start:start + batch])

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": getattr(embedding, "model_name", embedding.__class__.__name__),
            "vector_dimension": dimension,
            "record_count": len(rows),
            "collection": collection,
            "metric": metric,
        }
        cls._write_meta(client, collection, metadata)
        return cls(embedding, client, collection, metric, metadata)

    @classmethod
    def _create_collection(cls, client, collection: str, dimension: int, metric: str) -> None:
        from pymilvus import DataType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(ID_FIELD, DataType.VARCHAR, is_primary=True, max_length=MAX_ID_LENGTH)
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=MAX_TEXT_LENGTH)
        schema.add_field(GROUP_FIELD, DataType.VARCHAR, max_length=MAX_GROUP_LENGTH)
        schema.add_field(META_FIELD, DataType.VARCHAR, max_length=MAX_META_LENGTH)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD, index_type="AUTOINDEX", metric_type=metric
        )
        client.create_collection(
            collection_name=collection, schema=schema, index_params=index_params
        )

    # ---------- 索引元数据：存在库里，不依赖本地文件 ----------

    @classmethod
    def _meta_collection(cls, collection: str) -> str:
        return f"{collection}{META_SUFFIX}"

    @classmethod
    def _write_meta(cls, client, collection: str, metadata: dict[str, Any]) -> None:
        """把元数据写成一张小表的若干行。

        Milvus 不允许集合没有向量字段，所以带一个 1 维哑向量；它不参与检索，
        只是让这张表能被 load / query。
        """
        from pymilvus import DataType

        name = cls._meta_collection(collection)
        if client.has_collection(name):
            client.drop_collection(name)
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(META_KEY_FIELD, DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(META_VALUE_FIELD, DataType.VARCHAR, max_length=4096)
        schema.add_field(META_DUMMY_FIELD, DataType.FLOAT_VECTOR, dim=1)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=META_DUMMY_FIELD, index_type="AUTOINDEX", metric_type="L2"
        )
        client.create_collection(collection_name=name, schema=schema,
                                 index_params=index_params)
        client.insert(collection_name=name, data=[
            {META_KEY_FIELD: key,
             META_VALUE_FIELD: json.dumps(value, ensure_ascii=False),
             META_DUMMY_FIELD: [0.0]}
            for key, value in metadata.items()
        ])

    @classmethod
    def _read_meta(cls, client, collection: str, uri: str) -> dict[str, Any]:
        """先读库里的元数据表；读不到再回退到旧版本的本地旁车文件。"""
        from_metadata = cls._read_meta_from_collection(client, collection)
        if from_metadata:
            return from_metadata
        if _is_local(uri):
            path = meta_path_for(uri)
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return {}
        return {}

    @classmethod
    def _read_meta_from_collection(cls, client, collection: str) -> dict[str, Any]:
        name = cls._meta_collection(collection)
        try:
            if not client.has_collection(name):
                return {}
            client.load_collection(name)
            rows = client.query(collection_name=name, filter=f'{META_KEY_FIELD} != ""',
                                output_fields=[META_KEY_FIELD, META_VALUE_FIELD])
        except Exception:      # 元数据表损坏不应让整个检索不可用
            return {}
        metadata: dict[str, Any] = {}
        for row in rows:
            try:
                metadata[row[META_KEY_FIELD]] = json.loads(row[META_VALUE_FIELD])
            except (json.JSONDecodeError, KeyError):
                continue
        return metadata

    # ---------- 检索 ----------

    def search(self, query: str, k: int, groups: frozenset[str],
               score_threshold: float | None = None) -> list[tuple[Document, float]]:
        if not groups:
            return []
        vector = self._embedding.embed_query(query)
        hits = self._client.search(
            collection_name=self._collection,
            data=[vector],
            limit=max(1, int(k)),
            filter=group_filter_expr(groups),
            output_fields=[TEXT_FIELD, GROUP_FIELD, META_FIELD],
        )
        results: list[tuple[Document, float]] = []
        for hit in hits[0] if hits else []:
            score = max(0.0, float(hit["distance"]))  # 与 JSON 后端同一套相关度映射
            if score_threshold is not None and score < score_threshold:
                continue
            results.append((self._to_document(hit), score))
        return results

    def _to_document(self, hit: dict) -> Document:
        entity = hit.get("entity", {})
        try:
            metadata = json.loads(entity.get(META_FIELD) or "{}")
        except json.JSONDecodeError:
            metadata = {}
        # grp 是过滤列，group 是元数据键：两者同值，但读回来时以列为准
        metadata["group"] = entity.get(GROUP_FIELD, metadata.get("group", ""))
        metadata["id"] = hit.get("id", metadata.get("id", ""))
        return Document(page_content=entity.get(TEXT_FIELD, ""), metadata=metadata)

    # ---------- 供 BM25 / 健康检查 ----------

    def records(self) -> list[dict]:
        rows = self._client.query(
            collection_name=self._collection,
            filter=f'{ID_FIELD} != ""',
            output_fields=[ID_FIELD, TEXT_FIELD, GROUP_FIELD, META_FIELD],
            limit=16384,
        )
        records = []
        for row in rows:
            try:
                metadata = json.loads(row.get(META_FIELD) or "{}")
            except json.JSONDecodeError:
                metadata = {}
            metadata.setdefault("group", row.get(GROUP_FIELD, ""))
            records.append({"id": row.get(ID_FIELD, ""), "text": row.get(TEXT_FIELD, ""), **metadata})
        return records

    def count(self) -> int:
        try:
            return int(self._client.get_collection_stats(self._collection).get("row_count", 0))
        except Exception:
            return 0
