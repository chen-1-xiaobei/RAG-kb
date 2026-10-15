"""离线检索与对抗评测：只跑「粗排 + 精排」真实管线，不调用任何 LLM。

    python evaluate.py
    python evaluate.py --update-baseline     # 人工确认指标后刷新回归基线

设计要点：
- 每个用例的权限组固定，走与生产完全相同的预过滤 + 精排路径，因此指标反映真实服务行为。
- 排序质量用 nDCG@k 与 MRR，而不是只看"来源有没有出现"——top-1 排错和排在第 8 位
  对 RAG 是两回事，集合指标看不见这个差别。多来源用例可给分级相关性
  （relevant_sources），单来源用例退化为二值相关性。
- 对抗用例分两类：权限负例（无权限组不得召回受限文档）与注入用例
  （检索到的文档正文不得携带注入标记 canary）。注入用例靠运行时断言，
  而不是读数据集自证——数据集里写"这条应该防住"本身证明不了任何事。
- 退出码：门槛不达标或相对基线回退即 exit 1，可直接接 CI。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from auth import Principal  # noqa: E402
from app import retrieve_ranked  # noqa: E402
from eval_metrics import NDCG_K, grades_for, mean, ndcg, reciprocal_rank  # noqa: E402
from tools import TableRegistry, query_table_impl  # noqa: E402

DATASET = ROOT / "evals" / "dataset.json"
BASELINE = ROOT / "evals" / "baseline.json"
# 门槛：集合召回允许少量波动；表格工具、权限与注入都是确定性保证，必须 100%
THRESHOLDS = {
    "source_recall": 0.85,
    "table_tool_accuracy": 1.0,
    "answerability_accuracy": 1.0,
    "permission_denial_accuracy": 1.0,
    "injection_resistance": 1.0,
}


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {entry["source"]: entry for entry in payload["sources"]}, payload["cases"]


def source_group(sources: dict, name: str) -> str:
    return sources[name]["groups"][0]


# ---------------------------------------------------------------------------
# 单用例评测
# ---------------------------------------------------------------------------

def evaluate_table_case(case: dict, sources: dict, registry: TableRegistry) -> dict:
    """表格用例：走 query_table 工具层，而不是向量检索。

    结构化 CSV 不进入向量索引（只有 /agent/ask 的 list_tables/query_table 能访问），
    用检索指标去评它属于出题超出能力边界——这类用例必须由 SQL 工具自己回答。
    这里执行数据集给定的确定性 SQL（模拟"模型写对了 SQL"这一步），
    因此测的是工具层的数据通路与权限，不测模型的 SQL 生成能力。
    """
    spec = case["table"]
    principal = Principal("eval-user", frozenset(case["groups"]))
    text, docs = query_table_impl(spec["name"], spec["sql"], principal, registry, 50)

    wanted = spec["expect_cells"]
    rows = [line for line in text.splitlines() if line.startswith("|")]
    matched = [cell for cell in wanted if any(cell in row for row in rows)]
    return {
        "question": case["question"],
        "kind": "table",
        "ranked": [f"table:{spec['name']}"],
        "required": [spec["name"]],
        "recall": len(matched) / len(wanted),
        "hit": 1.0 if len(matched) == len(wanted) else 0.0,
        "ok": len(matched) == len(wanted),
        "authorized": spec["name"] in registry.tables,
        "notes": case.get("note", ""),
        "missing_cells": [cell for cell in wanted if cell not in matched],
    }


def evaluate_case(case: dict, sources: dict, registry: TableRegistry) -> dict:
    if case["kind"] == "table":
        return evaluate_table_case(case, sources, registry)

    principal = Principal("eval-user", frozenset(case["groups"]))
    hits = retrieve_ranked(case["question"], 4, principal)

    # 来源去重：同一文档的多个块只按最佳排名计一次（nDCG 评的是文档级相关性）
    ranked = list(dict.fromkeys(doc.metadata["source"] for doc, _ in hits))
    required = set(case["required_sources"])
    should_retrieve = bool(required)

    result = {
        "question": case["question"],
        "kind": case.get("kind", "retrieval"),
        "ranked": ranked,
        "required": sorted(required),
        "ok": True,
        "authorized": all(source_group(sources, name) in principal.allowed_groups for name in required),
        "notes": case.get("note", ""),
    }

    if should_retrieve:
        result["hit"] = 1.0 if required & set(ranked) else 0.0
        result["recall"] = len(required & set(ranked)) / len(required)
        result["rr"] = reciprocal_rank(ranked, required)
        result["ndcg"] = ndcg(ranked, grades_for(case))
        result["ok"] = result["recall"] >= 1.0
    else:
        # 权限负例 / 无关问题：一条都不该召回（包括不得泄露受限文档的存在性）
        result["no_hits"] = 1.0 if not hits else 0.0
        result["ok"] = not hits

    canary = case.get("injection_canary")
    if canary:
        # 运行时断言：检索到的文档正文不得携带注入标记。
        # 若问题本身含该标记，无法区分"文档注入"与"问题回声"，单独登记不判失败。
        if canary in case["question"]:
            result["canary_in_question"] = True
        else:
            tainted = [doc.metadata["source"] for doc, _ in hits if canary in doc.page_content]
            result["injection"] = 0.0 if tainted else 1.0
            if tainted:
                result["ok"] = False
                result["notes"] = f"{result['notes']} | 注入标记出现在 {tainted}".strip(" |")
    return result


# ---------------------------------------------------------------------------
# 汇总与基线
# ---------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    table = [r for r in results if r["kind"] == "table"]
    vector = [r for r in results if r["kind"] != "table"]
    retrieval = [r for r in vector if r["required"]]
    abstain = [r for r in vector if not r["required"] and r["kind"] != "permission"]
    permission = [r for r in vector if r["kind"] == "permission"]
    injection = [r for r in results if "injection" in r]

    return {
        "cases": len(results),
        "source_recall": mean([r["recall"] for r in retrieval]),
        "hit_rate": mean([r["hit"] for r in retrieval]),
        "mrr": mean([r["rr"] for r in retrieval]),
        f"ndcg@{NDCG_K}": mean([r["ndcg"] for r in retrieval]),
        "table_tool_accuracy": mean([r["hit"] for r in table]),
        "answerability_accuracy": mean([r["hit"] for r in retrieval] + [r["no_hits"] for r in abstain]),
        "permission_denial_accuracy": mean([r["no_hits"] for r in permission]),
        "injection_resistance": mean([r["injection"] for r in injection]),
        "misconfigured_cases": len([r for r in results if not r["authorized"]]),
    }


def check_thresholds(metrics: dict) -> list[str]:
    failures = [
        f"{name}={metrics[name]:.1%} < 门槛 {floor:.0%}"
        for name, floor in THRESHOLDS.items()
        if metrics.get(name, 0.0) < floor
    ]
    if metrics["misconfigured_cases"]:
        failures.append(f"{metrics['misconfigured_cases']} 条用例的期望来源超出其权限组（配置错误）")
    return failures


def compare_baseline(metrics: dict) -> list[str]:
    if not BASELINE.exists():
        return []
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    tolerance = float(base.get("tolerance", 0.02))
    return [
        f"{name}: {base['metrics'][name]:.3f} -> {value:.3f}"
        for name, value in metrics.items()
        if isinstance(value, float)
        and isinstance(base["metrics"].get(name), (int, float))
        and value < base["metrics"][name] - tolerance
    ]


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线检索与对抗评测")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--update-baseline", action="store_true", help="刷新 evals/baseline.json")
    parser.add_argument("--quiet", action="store_true", help="只打印汇总")
    args = parser.parse_args(argv)

    sources, cases = load_dataset(args.dataset)
    registry = TableRegistry()
    registry.load()
    results = [evaluate_case(case, sources, registry) for case in cases]

    if not args.quiet:
        for result in results:
            parts = " ".join(
                f"{key}={result[key]:.2f}"
                for key in ("recall", "rr", "ndcg", "no_hits", "injection")
                if key in result
            )
            print(f"[{'OK  ' if result['ok'] else 'FAIL'}] {result['question'][:34]:<34} {parts}")

    metrics = summarize(results)
    print("\n" + "=" * 78)
    for name, value in metrics.items():
        print(f"{name:<30} {value:.2%}" if isinstance(value, float) else f"{name:<30} {value}")

    failures = check_thresholds(metrics)
    drops = compare_baseline(metrics)
    print("=" * 78)
    if failures:
        print("门槛不达标：")
        for item in failures:
            print(f"  - {item}")
    else:
        print("门槛检查：全部通过")
    if not BASELINE.exists():
        print("基线：未建立（人工确认指标后运行 python evaluate.py --update-baseline）")
    elif drops:
        print("相对基线回退：")
        for item in drops:
            print(f"  - {item}")
    else:
        print("基线对比：无回退")

    if args.update_baseline:
        BASELINE.write_text(
            json.dumps(
                {
                    "note": "检索回归基线；由 evaluate.py --update-baseline 生成，人工确认后提交",
                    "tolerance": 0.02,
                    "metrics": {k: v for k, v in metrics.items() if isinstance(v, float)},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"已写入 {BASELINE.relative_to(ROOT)}")
        return 0

    return 1 if failures or drops else 0


if __name__ == "__main__":
    raise SystemExit(main())
