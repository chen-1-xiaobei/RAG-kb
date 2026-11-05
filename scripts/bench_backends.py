"""对比 JSON 与 Milvus 两个后端的检索延迟。

    python scripts/bench_backends.py            # 真实语料（126 chunk）
    python scripts/bench_backends.py --synthetic 2000

为什么要这张表：

"换成 Milvus"本身不是结论。**在 126 chunk 这个规模上，全量点积很可能比走一次
gRPC + 本地服务更快**——如果实测是这样，那结论应该是"小规模不需要向量库，
大规模才需要"，而不是"Milvus 更好"。这种表才回答得了"什么时候该换"。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

import numpy as np  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from auth import Principal  # noqa: E402
from config import settings  # noqa: E402
from kb_store import JSONVectorStore  # noqa: E402
from milvus_store import MilvusVectorStore  # noqa: E402
from retrieval import embeddings  # noqa: E402

GROUPS = frozenset({"public", "财务", "人事"})
K = 8


class HashEmbedding(Embeddings):
    """确定性替身：只用于合成语料，不需要真实模型的语义。"""

    model_name = "hash-embedding"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "little")
        rng = np.random.default_rng(seed)
        vector = rng.random(self.dim)
        return (vector / np.linalg.norm(vector)).tolist()

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def timed(store, questions: list[str], rounds: int = 3) -> dict:
    """测检索延迟，并单独测出查询编码耗时。

    为什么要拆开：`search()` 内部包含 embed_query。真实语料用的是 bge（约 10ms/次），
    合成语料用的是哈希向量（约 0.1ms/次），直接比总耗时等于在比 embedding 模型，
    而不是比存储。**净存储耗时 = 总耗时 − 编码耗时**，这才是换库真正影响的部分。
    """
    embedding = store.embeddings
    store.search(questions[0], K, GROUPS)          # 预热：排除连接/首次加载成本

    encode_latencies: list[float] = []
    latencies: list[float] = []
    for _ in range(rounds):
        for question in questions:
            started = time.perf_counter()
            embedding.embed_query(question)
            encode_latencies.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            store.search(question, K, GROUPS)
            latencies.append((time.perf_counter() - started) * 1000)

    latencies.sort()
    encode_latencies.sort()
    encode_mean = statistics.fmean(encode_latencies)
    total_mean = statistics.fmean(latencies)
    return {
        "n": len(latencies),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 2),
        "encode_ms": round(encode_mean, 2),
        "store_ms": round(max(0.0, total_mean - encode_mean), 2),
    }


def real_corpus_questions() -> list[str]:
    payload = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))
    questions = [
        case["question"] for case in payload["cases"]
        if case["kind"] == "retrieval" and case.get("required_sources")
    ]
    return questions[:20]


def bench_real() -> dict:
    """真实语料：index.json（若存在）+ milvus.db（若存在）。"""
    questions = real_corpus_questions()
    rows = {}

    index_path = ROOT / "index.json"
    if index_path.exists():
        store = JSONVectorStore.load(index_path, embeddings)
        rows["json"] = {"records": store.count(), **timed(store, questions)}
    else:
        rows["json"] = {"skipped": "index.json 不存在，先跑 python ingest.py --backend json"}

    try:
        store = MilvusVectorStore.open(
            embedding=embeddings,
            uri=str(ROOT / settings.milvus_uri) if not settings.milvus_uri.startswith("http")
            else settings.milvus_uri,
            collection=settings.milvus_collection,
            metric=settings.milvus_metric,
        )
        rows["milvus"] = {"records": store.count(), **timed(store, questions)}
    except Exception as exc:
        rows["milvus"] = {"skipped": f"{type(exc).__name__}: {exc}"}

    return rows


def bench_synthetic(total: int) -> dict:
    """合成语料：同一批文档分别写入两个后端，再测同样的查询。"""
    embedding = HashEmbedding()
    documents = [
        Document(
            page_content=f"合成文档 {i} 的正文内容，用于规模压测。",
            metadata={"source": f"doc{i % 50}.md", "relative_path": f"财务/doc{i % 50}.md",
                      "group": ["public", "财务", "人事"][i % 3], "title": f"文档{i % 50}",
                      "section": f"第{i % 7}节", "page": None, "para": i % 11},
        )
        for i in range(total)
    ]
    questions = [f"合成查询 {i}" for i in range(20)]
    rows = {}

    # 不用 TemporaryDirectory：Windows 上 milvus-lite 的服务进程仍持有文件句柄，
    # 退出时清理会抛 PermissionError。放在 .tmp/ 下由 .gitignore 兜住即可。
    temp = ROOT / ".tmp" / f"bench_{int(time.time())}"
    temp.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    json_path = temp / "index.json"
    JSONVectorStore.from_documents(documents, embedding, path=json_path)
    json_build = (time.perf_counter() - started) * 1000
    store = JSONVectorStore.load(json_path, embedding)
    rows["json"] = {"records": store.count(), "build_ms": round(json_build),
                    **timed(store, questions)}

    started = time.perf_counter()
    milvus = MilvusVectorStore.build(
        documents, embedding=embedding, uri=str(temp / "milvus.db"),
        collection="bench_chunks",
    )
    milvus_build = (time.perf_counter() - started) * 1000
    rows["milvus"] = {"records": milvus.count(), "build_ms": round(milvus_build),
                      **timed(milvus, questions)}

    return rows


def render(title: str, rows: dict) -> None:
    print(f"\n=== {title}")
    print(f"{'后端':<8}{'记录数':>8}{'建库ms':>9}{'p50总':>9}{'p95总':>9}"
          f"{'编码':>8}{'净存储':>9}")
    for name, row in rows.items():
        if "skipped" in row:
            print(f"{name:<8}{'-':>8}{'-':>9}{'-':>9}{'-':>9}{'-':>8}{'-':>9}   {row['skipped']}")
            continue
        print(f"{name:<8}{row['records']:>8}{row.get('build_ms', '-'):>9}"
              f"{row['p50_ms']:>9}{row['p95_ms']:>9}"
              f"{row['encode_ms']:>8}{row['store_ms']:>9}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=int, default=0, help="合成语料条数")
    args = parser.parse_args()

    render("真实语料（126 chunk）", bench_real())
    if args.synthetic:
        render(f"合成语料（{args.synthetic} chunk）", bench_synthetic(args.synthetic))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
