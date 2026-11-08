"""CPU vs GPU 精排基准：同一批候选、同一个模型，只换设备。

    python scripts/bench_rerank_device.py

它回答一个问题：**把 568M 的 cross-encoder 从 CPU 挪到 GPU，单对耗时能降多少？**
同时对比 batch 大小——CPU 上"batch 越大越慢"的结论在 GPU 上通常反转。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

from sentence_transformers import CrossEncoder  # noqa: E402

from config import settings  # noqa: E402

PAIRS = 8   # 与候选数一致（top_k 4 × factor 2）


def load_pairs() -> list[tuple[str, str]]:
    """用真实语料的一批候选，保证文档长度分布和线上一致。"""
    payload = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))
    question = next(
        case["question"] for case in payload["cases"]
        if case["kind"] == "retrieval" and case.get("required_sources")
    )
    from retrieval import index_manager, embeddings      # 延迟导入：需要索引
    from auth import Principal

    store = index_manager.get()
    docs = store.search(question, k=PAIRS, groups=frozenset({"public", "财务", "人事"}))
    return [(question, doc.page_content) for doc, _ in docs] or [(question, "占位")]


def bench(device: str, batch_size: int, pairs: list[tuple[str, str]]) -> dict:
    model = CrossEncoder(settings.rerank_model, device=device)

    # 预热：排除首次推理的 kernel 编译 / 显存分配
    model.predict(pairs[:1], batch_size=1)

    latencies = []
    for _ in range(3):
        started = time.perf_counter()
        model.predict(pairs, batch_size=batch_size)
        latencies.append((time.perf_counter() - started) * 1000)

    if device == "cuda":
        import torch
        vram = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
    else:
        vram = None

    return {
        "device": device,
        "batch_size": batch_size,
        "pairs": len(pairs),
        "median_ms": round(statistics.median(latencies)),
        "per_pair_ms": round(statistics.median(latencies) / len(pairs)),
        "vram_mb": vram,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="*", default=["cpu", "cuda"])
    parser.add_argument("--batches", nargs="*", type=int, default=[2, 8])
    args = parser.parse_args()

    pairs = load_pairs()
    print(f"候选 {len(pairs)} 对，平均文档长度 "
          f"{round(statistics.fmean(len(doc) for _, doc in pairs))} 字\n")

    rows = []
    for device in args.devices:
        for batch in args.batches:
            try:
                row = bench(device, batch, pairs)
            except Exception as exc:
                print(f"  {device:<5} batch={batch:<3} 失败：{type(exc).__name__}: {exc}")
                continue
            rows.append(row)
            print(f"  {row['device']:<5} batch={row['batch_size']:<3} "
                  f"总 {row['median_ms']:>6}ms  单对 {row['per_pair_ms']:>5}ms"
                  + (f"  显存 {row['vram_mb']}MB" if row.get("vram_mb") else ""))

    if rows:
        cpu = next((r for r in rows if r["device"] == "cpu"), None)
        gpu = next((r for r in rows if r["device"] == "cuda"), None)
        if cpu and gpu:
            print(f"\n单请求（{len(pairs)} 对）加速比："
                  f"{cpu['median_ms'] / gpu['median_ms']:.1f}x")

    out = ROOT / "eval-results" / "bench-rerank-device.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
