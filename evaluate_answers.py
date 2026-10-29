"""答案层评测：只评「检索之后」的那一段。

    python evaluate_answers.py            # 离线回放（读 fixture，不需要密钥）
    python evaluate_answers.py --record   # 录制草稿（需要 DEEPSEEK_API_KEY）

为什么要单独一层：检索评测（evaluate.py）只回答"正确来源有没有排上来"，
但用户看到的是答案。检索对了，答案仍可能：
- 引用到不该引用的来源（跨权限或跑题）；
- 该拒答时硬答（编造）；
- 把数值写错（"三十天"答成"六十天"）。

三个指标：
- `behavior_accuracy`：该答的答了、该拒的拒了（对照真实生产逻辑）
- `citation_precision`：被引用的来源是否都落在允许集合内
- `fact_presence`：答案里出现了期望事实串、且没有出现错误事实串

**设计要点：复用生产代码而不是重抄一遍。** 引用校验与"证据不足"判定都在
`app.finalize_answer` 里，评测直接调用它、只把模型草稿换成录制好的那份。
否则评测测的是"我以为生产会怎么判"，而不是生产真的怎么判。

**已知边界**：`must_contain` / `must_not_contain` 是字符串包含判定，
不是语义蕴含。它抓得住数值写错与答案缺关键事实，抓不住"用正确词汇说错话"。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from auth import Principal  # noqa: E402
from clarification import REFUSAL_TEXT  # noqa: E402  纯函数模块，不拉起模型栈


@dataclass
class RecordedDraft:
    """录制下来的模型草稿：只保留 finalize_answer 需要的那三个字段。"""

    answer: str
    citations: list[int] = field(default_factory=list)
    insufficient_context: bool = False


def as_draft(payload: dict | None) -> RecordedDraft | None:
    if payload is None:
        return None
    return RecordedDraft(
        answer=payload.get("answer", ""),
        citations=list(payload.get("citations") or []),
        insufficient_context=bool(payload.get("insufficient_context")),
    )


CASES = ROOT / "evals" / "answers.json"
FIXTURE = ROOT / "evals" / "answers.fixture.json"


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def load_fixture(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {entry["id"]: entry["draft"] for entry in payload.get("drafts", [])}


# ---------------------------------------------------------------------------
# 录制
# ---------------------------------------------------------------------------

def record(cases: list[dict], fixture_path: Path) -> int:
    """用真实模型跑一遍，把草稿存下来。

    只录 `expect == "answer"` 的用例：拒答路径在检索层就返回了，根本不会调用模型，
    录它没有意义，也测不出东西。
    """
    from app import finalize_answer, generate_answer  # 延迟导入：需要密钥与模型
    from retrieval import retrieve_ranked

    drafts = []
    for case in cases:
        if case["expect"] != "answer":
            continue
        principal = Principal("eval-user", frozenset(case["groups"]))
        hits = retrieve_ranked(case["question"], 4, principal)
        if not hits:
            print(f"跳过 {case['id']}：检索零召回，无法录制答案草稿", file=sys.stderr)
            continue
        # 走真实链路拿到草稿：generate_answer 内部会调用模型
        from app import format_docs, get_answer_chain

        draft = get_answer_chain().invoke(
            {"context": format_docs([doc for doc, _ in hits]), "question": case["question"]}
        )
        drafts.append({
            "id": case["id"],
            "question": case["question"],
            "draft": {
                "answer": draft.answer,
                "citations": list(draft.citations),
                "insufficient_context": bool(draft.insufficient_context),
            },
        })
        print(f"已录制 {case['id']}: citations={draft.citations} "
              f"insufficient={draft.insufficient_context}")

    fixture_path.write_text(
        json.dumps(
            {
                "note": "答案层评测固件。由 evaluate_answers.py --record 生成；"
                        "内容是真实模型的草稿（answer / citations / insufficient_context），"
                        "评测时喂给 app.finalize_answer，因此不需要密钥也不联网。",
                "drafts": drafts,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {fixture_path.relative_to(ROOT)}（{len(drafts)} 条）")
    return 0


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

def evaluate(cases: list[dict], drafts: dict[str, dict],
             retrieve_fn=None, finalize_fn=None) -> tuple[list[dict], list[str]]:
    """跑一遍评测。

    `retrieve_fn` / `finalize_fn` 可注入：生产路径默认用真实实现，
    单测里换成桩就不需要建索引、也不需要导入 app（它会在 import 期拉起模型栈）。
    这一层注入是踩过坑之后加的——原来的实现把 import 写死在函数体内，
    单测在 CI 上没有 index.json 就直接 FileNotFoundError。
    """
    if retrieve_fn is None:
        from retrieval import retrieve_ranked as retrieve_fn
    if finalize_fn is None:
        from app import finalize_answer as finalize_fn

    results: list[dict] = []
    missing: list[str] = []

    for case in cases:
        draft = drafts.get(case["id"])

        # 先看固件在不在，再去检索：没有草稿就没有可评测的东西，
        # 没必要白跑一次检索（也会让"缺固件"这件事被检索错误盖住）。
        if case["expect"] == "answer" and draft is None:
            missing.append(case["id"])
            continue

        principal = Principal("eval-user", frozenset(case["groups"]))
        hits = retrieve_fn(case["question"], 4, principal)

        response = finalize_fn(
            case["question"], hits, as_draft(draft) if draft else None, "eval"
        )

        refused = (not response.sources) or response.answer == REFUSAL_TEXT \
            or response.insufficient_context
        expected = case["expect"]

        cited = {source.source for source in response.sources}
        allowed = set(case["required_sources"])
        citation_ok = cited <= allowed if expected == "answer" else not cited

        text = response.answer or ""
        contains = [item for item in case.get("must_contain", []) if item not in text]
        forbidden = [item for item in case.get("must_not_contain", []) if item in text]

        behavior_ok = refused if expected == "refuse" else not refused
        fact_ok = not contains and not forbidden

        results.append({
            "id": case["id"],
            "question": case["question"],
            "expect": expected,
            "refused": refused,
            "cited": sorted(cited),
            "behavior_ok": behavior_ok,
            "citation_ok": citation_ok,
            "fact_ok": fact_ok,
            "missing_facts": contains,
            "forbidden_facts": forbidden,
            "note": case.get("note", ""),
        })
    return results, missing


def summarize(results: list[dict]) -> dict:
    total = len(results) or 1
    answer_cases = [r for r in results if r["expect"] == "answer"]
    refuse_cases = [r for r in results if r["expect"] == "refuse"]
    return {
        "cases": len(results),
        "behavior_accuracy": sum(r["behavior_ok"] for r in results) / total,
        "answer_recall": (sum(r["behavior_ok"] for r in answer_cases) / len(answer_cases))
        if answer_cases else 1.0,
        "refusal_accuracy": (sum(r["behavior_ok"] for r in refuse_cases) / len(refuse_cases))
        if refuse_cases else 1.0,
        "citation_precision": sum(r["citation_ok"] for r in results) / total,
        "fact_presence": sum(r["fact_ok"] for r in results) / total,
        "fully_correct": sum(
            r["behavior_ok"] and r["citation_ok"] and r["fact_ok"] for r in results
        ) / total,
    }


THRESHOLDS = {
    "behavior_accuracy": 1.0,     # 该答/该拒是硬要求
    "refusal_accuracy": 1.0,      # 拒答是这个项目的核心承诺，不接受"大多数时候"
    "citation_precision": 1.0,    # 引用落在允许来源之外即为错
    "fact_presence": 0.9,         # 措辞有自由，允许少量未命中
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="答案层评测")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--record", action="store_true", help="用真实模型重新录制草稿")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)

    if args.record:
        return record(cases, args.fixture)

    drafts = load_fixture(args.fixture)
    results, missing = evaluate(cases, drafts)

    if not args.quiet:
        for result in results:
            flags = "".join([
                "行为" if result["behavior_ok"] else "行为✗",
                " 引用" if result["citation_ok"] else " 引用✗",
                " 事实" if result["fact_ok"] else " 事实✗",
            ])
            print(f"[{result['id']}] {result['question'][:26]:<26} {result['expect']:<6} "
                  f"refused={result['refused']!s:<5} {flags}")
            if result["missing_facts"]:
                print(f"        缺少事实: {result['missing_facts']}")
            if result["forbidden_facts"]:
                print(f"        出现错误事实: {result['forbidden_facts']}")

    metrics = summarize(results)
    print("\n" + "=" * 70)
    for name, value in metrics.items():
        print(f"{name:<22} {value:.2%}" if isinstance(value, float) else f"{name:<22} {value}")
    print("=" * 70)

    failures = [
        f"{name}={metrics[name]:.1%} < 门槛 {floor:.0%}"
        for name, floor in THRESHOLDS.items()
        if metrics.get(name, 0.0) < floor
    ]
    if missing:
        failures.append(f"缺少录制草稿: {missing}（先运行 --record）")

    if failures:
        print("门槛不达标：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("门槛检查：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
