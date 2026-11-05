"""向量后端协议：检索层只依赖这个接口，不依赖具体实现。

为什么要有这层：

开头的实现是 `JSONVectorStore`——126 chunk 全量点积，简单且够快。后来接 Milvus，
两件事必须同时成立：

1. **检索逻辑不能重写**。`retrieval.py` 只该关心"给我 k 个候选"，不该关心存储长什么样。
2. **评测必须能对比**。两个后端要能被同一套评测跑出同一批指标，否则"换库"无法验证。

所以这里定义的是**能力**而不是**存储**：给查询、给允许的组、给条数，返回
(Document, 相似度) 列表。过滤参数刻意设计成**结构化的一组组名**，而不是
`JSONVectorStore` 用过的 Python 谓词——谓词无法翻译成 Milvus 的过滤表达式，
而组名两边都能翻译（JSON 侧在内存里比，Milvus 侧转成 `grp in [...]`）。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document


class VectorBackendError(RuntimeError):
    """后端不可用（依赖缺失、文件损坏等）。"""


class BackendConfigError(VectorBackendError):
    """索引与当前配置不兼容：embedding 模型、向量维度或 schema 版本不一致。"""


@runtime_checkable
class VectorBackend(Protocol):
    """向量后端需要提供的最小能力。"""

    index_metadata: dict[str, Any]

    def search(
        self,
        query: str,
        k: int,
        groups: frozenset[str],
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """按组预过滤后检索。未授权的记录不参与 top-k 竞争。"""
        ...

    def records(self) -> list[dict]:
        """全量记录（含 text 与 metadata）。BM25 需要它在内存里建倒排。"""
        ...

    def count(self) -> int:
        """记录条数，用于健康检查与日志。"""
        ...


def group_filter_expr(groups: frozenset[str] | set[str]) -> str:
    """把允许组翻译成 Milvus 过滤表达式。

    组名来自 data/ 的目录名（财务 / 人事 / public），不含引号；
    这里仍然做一次转义，避免目录名里出现单引号把表达式结构破坏掉——
    这是**表达式拼接**，和 SQL 注入是同一类问题。
    """
    escaped = [str(group).replace("\\", "\\\\").replace("'", "\\'") for group in groups]
    return "grp in [" + ", ".join(f"'{group}'" for group in escaped) + "]"


def open_backend(
    kind: str,
    *,
    embedding: Any,
    index_path: Any,
    milvus_uri: str,
    milvus_collection: str,
    milvus_metric: str = "COSINE",
) -> VectorBackend:
    """按配置打开后端。导入延迟到函数内，避免不装 pymilvus 时影响 JSON 路径。"""
    if kind == "json":
        from kb_store import JSONVectorStore

        return JSONVectorStore.load(index_path, embedding)
    if kind == "milvus":
        from milvus_store import MilvusVectorStore

        return MilvusVectorStore.open(
            embedding=embedding,
            uri=milvus_uri,
            collection=milvus_collection,
            metric=milvus_metric,
        )
    raise VectorBackendError(f"未知的向量后端：{kind}")
