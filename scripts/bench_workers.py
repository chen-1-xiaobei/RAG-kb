"""多进程实验：worker 数 × intra-op 线程数，对冷路径吞吐的影响。

    python scripts/bench_workers.py

背景（前一次压测结论）：单个 uvicorn 进程下，并发 1 → 4 时冷路径吞吐只从
0.08 涨到 0.10 req/s，而单请求从 13s 涨到 37s。原因是 torch 默认用 8 个
intra-op 线程，4 个并发请求就是 32 个线程抢 12 个逻辑核。

因此这里要回答的是：**用多进程 + 减少每进程线程数，能不能提高总吞吐？**
对照组刻意包含"2 进程但线程数不调"（预期更差），用来验证"不是进程越多越好，
而是要在核数与线程数之间对齐"。

每组配置都跑同样的冷路径问题（每题不同，避开精排逐对缓存）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

import httpx  # noqa: E402

HEADERS = {"X-Dev-User": "bench", "X-Dev-Groups": "%E8%B4%A2%E5%8A%A1"}


def load_questions() -> list[str]:
    payload = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))
    questions = [
        case["question"] for case in payload["cases"]
        if case["kind"] == "retrieval" and case.get("required_sources")
        and set(case.get("groups") or []) <= {"财务", "人事"}
    ]
    return questions


async def one(client: httpx.AsyncClient, url: str, question: str) -> tuple[float, bool]:
    started = time.perf_counter()
    try:
        response = await client.post(
            url, json={"question": question, "top_k": 4}, headers=HEADERS, timeout=300
        )
        return (time.perf_counter() - started) * 1000, response.status_code == 200
    except Exception:
        return (time.perf_counter() - started) * 1000, False


async def measure(url: str, questions: list[str], concurrency: int) -> dict:
    """并发 concurrency，每个请求用一道**没用过**的问题（冷缓存）。"""
    async with httpx.AsyncClient() as client:
        await one(client, url, questions[0])          # 预热
        batch = questions[1 : 1 + concurrency]
        started = time.perf_counter()
        results = await asyncio.gather(*[one(client, url, q) for q in batch])
        wall = time.perf_counter() - started
    latencies = sorted(latency for latency, _ in results)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "failures": sum(1 for _, ok in results if not ok),
        "qps": round(len(results) / wall, 3),
        "p50_ms": round(statistics.median(latencies)),
        "max_ms": round(latencies[-1]),
    }


def wait_health(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_tree(process: subprocess.Popen) -> None:
    """uvicorn --workers 会 fork 子进程，必须连子进程一起杀。"""
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def warmup_all_workers(port: int, workers: int, question: str) -> int:
    """逐个确认每个 worker 都能真正完成一次请求。

    健康检查只代表"至少一个 worker 就绪"。多进程模式下，未就绪的 worker 仍可能被连上，
    导致请求卡在排队里——实测表现为 300 秒超时，会被误读成"多进程性能差"。
    这里按 worker 数逐个打通，每个给足 180 秒。
    """
    ready = 0
    for _ in range(workers):
        try:
            response = httpx.post(
                f"http://127.0.0.1:{port}/ask",
                json={"question": question, "top_k": 4},
                headers=HEADERS, timeout=180,
            )
            if response.status_code == 200:
                ready += 1
        except Exception:
            pass
    return ready


def run_config(workers: int, threads: int, port: int, questions: list[str],
               levels: list[int]) -> dict:
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "PYTHONUTF8": "1",
        "VECTOR_BACKEND": "json",
    })
    env.pop("NO_PROXY", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--workers", str(workers), "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_health(port, timeout=420):
            return {"error": "服务未就绪"}
        ready = warmup_all_workers(port, workers, questions[-1])
        print(f"  已就绪 worker 数：{ready}/{workers}", flush=True)
        if ready < workers:
            return {"error": f"只有 {ready}/{workers} 个 worker 就绪"}
        url = f"http://127.0.0.1:{port}/ask"
        rows = [asyncio.run(measure(url, questions, level)) for level in levels]
        return {"workers": workers, "threads": threads, "ready": ready, "levels": rows}
    finally:
        kill_tree(process)
        time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--levels", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--output", default="eval-results/bench-workers.json")
    parser.add_argument(
        "--configs", default="1:8,2:4,2:8",
        help="worker:threads 列表，默认 1:8 基线、2:4 对齐、2:8 超额订阅对照",
    )
    args = parser.parse_args()

    questions = load_questions()
    configs = [tuple(int(part) for part in item.split(":")) for item in args.configs.split(",")]
    results = []
    for index, (workers, threads) in enumerate(configs):
        print(f"\n=== 配置 {workers} 进程 × {threads} 线程（端口 {args.port + index}）", flush=True)
        row = run_config(workers, threads, args.port + index, questions, args.levels)
        results.append(row)
        for level in row.get("levels", []):
            print(
                f"  并发={level['concurrency']:>2}  请求={level['requests']:>2}  "
                f"吞吐={level['qps']:>6.3f} req/s  p50={level['p50_ms']:>6}ms  "
                f"max={level['max_ms']:>6}ms  失败={level['failures']}",
                flush=True,
            )
        if "error" in row:
            print(f"  失败：{row['error']}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"configs": results, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
