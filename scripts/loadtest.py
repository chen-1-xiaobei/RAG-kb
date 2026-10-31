"""并发压测：测出 QPS 与延迟分位，并区分两条路径。

    python scripts/loadtest.py --base-url http://127.0.0.1:8016

两条路径的瓶颈完全不同，必须分开测：

- **CPU 路径**（`/ask` + 无关问题）：命中证据门槛的拒答分支，**不调用 LLM**，
  只跑粗排 + 精排。测的是"精排能不能并行、线程池够不够"。
- **LLM 路径**（`/agent/ask` + 真问题）：每次请求都要等 DeepSeek。测的是外部依赖，
  与机器 CPU 无关——这条路径的 QPS 主要受网络与供应商限流约束。

用它来回答"要不要把 SQLite 换成 MySQL"：如果瓶颈是精排或 LLM，换存储不会有任何改善。

服务端必须支持 dev 鉴权（`AUTH_MODE=dev`）。脚本同时抓 `/metrics`，
把服务端记录的精排耗时与客户端观测的端到端耗时分开看——两者一起才能定位瓶颈。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须早于 httpx 客户端创建：本机 NO_PROXY 里有 `[::1]`，httpx 会把它当 URL 解析并
# 抛 ValueError: invalid literal for int() with base 10: ':1]'。
# 服务端当初就踩过这个坑（envfix.py），压测脚本同样需要。
from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

import httpx  # noqa: E402

DEV_HEADERS = {
    "X-Dev-User": "loadtest",
    # 「财务」的 URL 编码：X-Dev-Groups 只接受 ASCII
    "X-Dev-Groups": "%E8%B4%A2%E5%8A%A1",
    "Content-Type": "application/json",
}

# 无关问题：知识库里没有答案 → 粗排按阈值过滤成 0 候选 → **精排提前返回、不调 LLM**
REFUSAL_QUESTION = "公司年会的抽奖奖品是什么"
# 有证据的问题：会真的跑完粗排 + 精排 + 一次 LLM 生成
EVIDENCE_QUESTION = "报销单的审批流程是什么"


def load_question_pool() -> list[str]:
    """从评测集里取真实问题，用于"每题只用一次"的冷路径压测。

    为什么必须换题：精排有逐对 LRU 缓存（键含 query），同一个问题重复打会全部命中，
    实测服务端 rerank 阶段耗时直接归零——那测到的是缓存，不是检索。
    """
    path = ROOT / "evals" / "dataset.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        case["question"]
        for case in payload["cases"]
        if case["kind"] == "retrieval" and case.get("required_sources")
        and set(case.get("groups") or []) <= {"财务", "人事"}
    ]
    random.Random(20261031).shuffle(questions)
    return questions


async def one_request(client: httpx.AsyncClient, url: str, payload: dict) -> tuple[float, bool]:
    started = time.perf_counter()
    try:
        response = await client.post(url, json=payload, headers=DEV_HEADERS, timeout=300)
        elapsed = (time.perf_counter() - started) * 1000
        return elapsed, response.status_code == 200
    except Exception:
        return (time.perf_counter() - started) * 1000, False


async def run_level(url: str, question: str, concurrency: int, rounds: int,
                    pool: list[str] | None = None) -> dict:
    """固定并发打 rounds 轮。

    `pool` 非空时，每个请求取一道**没用过**的问题（冷路径）；否则所有请求用同一题（热路径）。
    """
    latencies: list[float] = []
    failures = 0
    cursor = {"next": 0}

    async with httpx.AsyncClient() as client:
        await one_request(client, url, {"question": question, "top_k": 4})  # 预热

        started = time.perf_counter()
        for _ in range(rounds):
            async def task() -> tuple[float, bool]:
                if pool:
                    index = cursor["next"]
                    cursor["next"] += 1
                    asked = pool[index % len(pool)]
                else:
                    asked = question
                return await one_request(client, url, {"question": asked, "top_k": 4})

            results = await asyncio.gather(*[task() for _ in range(concurrency)])
            for elapsed, ok in results:
                latencies.append(elapsed)
                if not ok:
                    failures += 1
        wall = time.perf_counter() - started

    total = len(latencies)
    ordered = sorted(latencies)
    return {
        "concurrency": concurrency,
        "requests": total,
        "failures": failures,
        "wall_seconds": round(wall, 1),
        "qps": round(total / wall, 2) if wall else 0.0,
        "p50_ms": round(statistics.median(ordered), 0),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 0),
        "max_ms": round(ordered[-1], 0),
    }


def _parse_metric(line: str) -> tuple[str, str, float] | None:
    """把 `name{labels} value` 拆成 (name, labels, value)。

    第一版直接对整行 `rpartition(" ")` 取名字再判断 `endswith("_count")`，
    但名字里带着 `{stage="rerank"}`，后缀永远不是 `_count`——于是所有阶段耗时
    都被静默丢掉。压测报告里"服务端"一栏全空，差点把结论建立在空数据上。
    """
    metric, _, raw_value = line.rpartition(" ")
    if not metric:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    if "{" in metric:
        name, _, rest = metric.partition("{")
        return name, rest.rstrip("}"), value
    return metric, "", value


def scrape_metrics(base_url: str) -> dict[tuple[str, str], float]:
    """抓一次 /metrics，键是 (指标名, 标签串)。"""
    try:
        text = httpx.get(f"{base_url}/metrics", timeout=30).text
    except Exception:
        return {}
    values: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parsed = _parse_metric(line)
        if parsed:
            name, labels, value = parsed
            values[(name, labels)] = value
    return values


def summarize_metrics(before: dict, after: dict) -> dict:
    """两次快照之间的平均耗时（秒），按 summary 的 _count/_sum 算。"""
    result = {}
    for (name, labels), value in after.items():
        if not name.endswith("_count"):
            continue
        base = name[: -len("_count")]
        delta_count = value - before.get((name, labels), 0.0)
        if delta_count <= 0:
            continue
        delta_sum = after.get((f"{base}_sum", labels), 0.0) - before.get(
            (f"{base}_sum", labels), 0.0
        )
        result[f"{base}[{labels}]"] = round(delta_sum / delta_count, 3)
    return result


async def main_async(args) -> int:
    base = args.base_url.rstrip("/")
    report = {"base_url": base, "phases": {}}
    pool = load_question_pool()

    phases = (
        # 拒答快路径：0 候选 → 不跑精排、不调 LLM。测粗排与线程池的上限。
        ("refusal_fast", f"{base}/ask", REFUSAL_QUESTION, None, args.refusal_levels),
        # 检索热路径：同一问题重复打 → 精排缓存命中。反映"重复问题"这种理想情况。
        ("retrieval_warm", f"{base}/ask", EVIDENCE_QUESTION, None, args.warm_levels),
        # 检索冷路径：每题只用一次 → 精排真的在算。**真实流量的估算应以这条为准。**
        ("retrieval_cold", f"{base}/ask", "", pool, args.cold_levels),
        # Agent 路径：多一次 LLM 决策
        ("agent", f"{base}/agent/ask", EVIDENCE_QUESTION, None, args.agent_levels),
    )

    for phase, url, question, question_pool, levels in phases:
        if not levels:
            continue
        print(f"\n=== {phase}（{url}）")
        rows = []
        for concurrency in levels:
            before = scrape_metrics(base)
            row = await run_level(url, question, concurrency, args.rounds, pool=question_pool)
            after = scrape_metrics(base)
            row["server_stage_avg_s"] = summarize_metrics(before, after)
            rows.append(row)
            stages = " ".join(
                f"{_label(key)}={value}s" for key, value in row["server_stage_avg_s"].items()
            )
            print(
                f"  并发={concurrency:>2}  请求={row['requests']:>2}  "
                f"QPS={row['qps']:>6.2f}  p50={row['p50_ms']:>6.0f}ms  "
                f"p95={row['p95_ms']:>6.0f}ms  失败={row['failures']}  {stages}"
            )
        report["phases"][phase] = rows

    out = Path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写入 {out}")
    return 0


def _label(metric_key: str) -> str:
    """把 `rag_stage_seconds[stage="rerank"]` 显示成 `stage_rerank`。"""
    name, _, labels = metric_key.partition("[")
    name = name.replace("rag_", "").replace("_seconds", "")
    label = labels.rstrip("]").replace('stage="', "").rstrip('"')
    label = label.replace('endpoint="', "").rstrip('"')
    return f"{name}_{label}" if label else name


def main() -> int:
    parser = argparse.ArgumentParser(description="并发压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8017")
    parser.add_argument("--rounds", type=int, default=2, help="每个并发档打几轮")
    parser.add_argument("--refusal-levels", type=int, nargs="*", default=[1, 4, 16])
    parser.add_argument("--warm-levels", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--cold-levels", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--agent-levels", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--output", default="eval-results/loadtest.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
