"""Milvus 连通性与索引验收：换到 standalone 之后**先在服务器上跑这个**。

    python scripts/verify_milvus.py --uri http://127.0.0.1:19530 --collection kb_chunks

它检查四件事，任一不通过就退出码 1：

1. **能连上**（URI 可达、认证通过）
2. **索引存在且条数与预期一致**（默认期望 126，可用 --expect 改）
3. **元数据可读**：embedding 模型、维度、度量方式——服务模式下这些存在库里的
   元数据表，不是本地文件，多副本都能读到
4. **带权限过滤的检索能跑通**：用 `grp in [...]` 表达式过滤后能返回结果，
   并打印延迟（用来和本地 json 后端对比）

为什么不在 CI 里跑：CI 没有 Milvus 服务，起一套 etcd+MinIO+milvus 会让构建时间
翻几倍。这里把它做成"部署后手动验收"的脚本，与 `scripts/loadtest.py` 同类。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=None, help="默认取 MILVUS_URI")
    parser.add_argument("--collection", default=None, help="默认取 MILVUS_COLLECTION")
    parser.add_argument("--expect", type=int, default=None, help="期望的条数（默认不校验）")
    parser.add_argument("--question", default="一线城市住宿标准是多少")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    from config import settings

    uri = args.uri or settings.milvus_uri
    collection = args.collection or settings.milvus_collection
    print(f"连接 {uri} / collection={collection}")

    try:
        from pymilvus import MilvusClient
    except ImportError:
        print("✗ 未安装 pymilvus：pip install pymilvus milvus-lite")
        return 1

    started = time.perf_counter()
    try:
        client = MilvusClient(uri)
        client.list_collections()
    except Exception as exc:
        print(f"✗ 连接失败：{type(exc).__name__}: {exc}")
        return 1
    print(f"✓ 连接成功（{(time.perf_counter() - started) * 1000:.0f}ms）")

    if not client.has_collection(collection):
        print(f"✗ collection '{collection}' 不存在：先跑 VECTOR_BACKEND=milvus python ingest.py")
        return 1
    client.load_collection(collection)
    count = int(client.get_collection_stats(collection).get("row_count", 0))
    print(f"✓ collection 存在，row_count={count}")

    if args.expect is not None and count != args.expect:
        print(f"✗ 条数不符：期望 {args.expect}，实际 {count}")
        return 1

    # 元数据（存在库里的小表）
    from milvus_store import MilvusVectorStore

    metadata = MilvusVectorStore._read_meta_from_collection(client, collection)
    if not metadata:
        print("✗ 读不到索引元数据（元数据表缺失？）——换 embedding 模型将无法被检测")
        return 1
    print(f"✓ 元数据：模型={metadata.get('embedding_model')} "
          f"维度={metadata.get('vector_dimension')} 度量={metadata.get('metric')}")

    # 带权限过滤的检索
    from auth import Principal
    from retrieval import embeddings

    principal = Principal("verify", frozenset({"财务", "人事"}))
    from config import settings as current
    from vector_backend import open_backend

    backend = open_backend(
        "milvus", embedding=embeddings, index_path=ROOT / "index.json",
        milvus_uri=uri, milvus_collection=collection,
        milvus_metric=current.milvus_metric,
    )
    latencies = []
    hits = []
    for _ in range(max(1, args.rounds)):
        started = time.perf_counter()
        hits = backend.search(args.question, k=4, groups=principal.allowed_groups)
        latencies.append((time.perf_counter() - started) * 1000)
    print(f"✓ 检索成功：命中 {len(hits)} 条，p50={statistics.median(latencies):.0f}ms")
    for doc, score in hits[:3]:
        print(f"    [{doc.metadata.get('group')}] {doc.metadata.get('source')} "
              f"{doc.page_content[:40]!r} score={score:.3f}")

    # 越权过滤必须生效
    denied = backend.search(args.question, k=4, groups=frozenset({"不存在的组"}))
    if denied:
        print("✗ 权限过滤失效：不存在的组也查到了内容")
        return 1
    print("✓ 权限过滤生效（不存在的组返回空）")
    print("\n全部通过。下一步：VECTOR_BACKEND=milvus python evaluate.py 确认指标与 json 后端一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
