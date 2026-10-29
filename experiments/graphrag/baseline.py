"""基线：普通 top-k 检索在全局聚合题上能覆盖多少要求文档。

    python experiments/graphrag/baseline.py

这是实验的前置：**先证明靶子存在**。如果普通 RAG 已经能把要求文档全覆盖，
那么加图召回就没有可测量的空间，实验应当诚实收尾停止。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from auth import Principal  # noqa: E402
from retrieval import retrieve_ranked  # noqa: E402

HERE = Path(__file__).resolve().parent
TOP_K = 4          # 与生产一致


def load_questions() -> list[dict]:
    return json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["cases"]


def coverage(ranked: list[str], required: list[str]) -> tuple[float, list[str]]:
    hit = [name for name in required if name in ranked]
    return len(hit) / len(required), [name for name in required if name not in ranked]


def main() -> int:
    cases = load_questions()
    rows = []
    total = 0.0

    for case in cases:
        principal = Principal("graphrag-eval", frozenset(case["groups"]))
        hits = retrieve_ranked(case["question"], TOP_K, principal)
        ranked = list(dict.fromkeys(doc.metadata["source"] for doc, _ in hits))
        score, missed = coverage(ranked, case["required_sources"])
        total += score
        rows.append({"id": case["id"], "coverage": score, "missing": missed, "ranked": ranked})
        print(f"[{case['id']}] 覆盖 {score:.0%}  缺: {missed}")

    mean = total / len(cases)
    print(f"\n平均文档覆盖率（top-{TOP_K}）: {mean:.1%}")
    (HERE / "baseline.json").write_text(
        json.dumps({"top_k": TOP_K, "mean_coverage": mean, "rows": rows},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
