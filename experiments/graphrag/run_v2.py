"""改进版对比：LLM 选实体 + PPR 扩散 + 文档配额 + 与向量融合。

    python experiments/graphrag/run_v2.py

五条臂，最终都只取 4 份文档、都过同一个精排模型：

1. baseline      生产同款：向量 top-8 → 精排 → top-4 chunk（不做文档去重）
2. dense8        向量 top-8 → 精排 → 按文档去重
3. graph_v2      LLM 选实体 → PPR 扩散 → 文档配额 → 精排 → 按文档去重
4. fused         dense8 与 graph_v2 用 RRF 融合 → 精排 → 按文档去重
5. wide40        向量 top-40 → 精排 → 按文档去重（上限参考）

2 是 3、4 的等预算对照：**没有它就无法区分"图有用"和"候选变多有用"**——
第一版实验就是缺了这个对照，才误以为图有效。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph_v2
import keywords
import run_experiment as exp
from auth import Principal
from hybrid import rrf_fuse
from retrieval import index_manager, retrieve

HERE = Path(__file__).resolve().parent
TOP_K = 4
EQUAL = 8
PER_DOC_CAP = 2


def finalize(question: str, candidates: list, budget: int) -> list[str]:
    return exp.rank_documents(question, candidates, budget)


def main() -> int:
    questions = exp.load(HERE / "questions.json")["cases"]
    entities = exp.load(HERE / "entities.json")
    entity_to_chunks, chunk_entities = exp.build_graph(entities)
    keyword_cache = keywords.load_cache()
    store = index_manager.get()
    graph = graph_v2.build_bipartite(store, chunk_entities)

    rows = []
    for case in questions:
        principal = Principal("graphrag-eval", frozenset(case["groups"]))
        question = case["question"]
        required = case["required_sources"]
        cached = keyword_cache.get(case["id"], {})
        selected = cached.get("selected", [])

        # 1) baseline：生产同款
        hits = exp.retrieve_ranked(question, TOP_K, principal)
        baseline = list(dict.fromkeys(doc.metadata["source"] for doc, _ in hits))

        # 2) dense8 + 去重
        dense_raw = retrieve(question, EQUAL, principal)
        dense8 = finalize(question, dense_raw, TOP_K)

        # 3) graph_v2：LLM 选实体 -> PPR -> 配额 -> 精排 -> 去重
        ppr = graph_v2.personalized_pagerank(graph, selected)
        ranked_positions = [pos for pos, _ in sorted(ppr.items(), key=lambda kv: kv[1], reverse=True)]
        quota_positions = graph_v2.apply_document_quota(store, ranked_positions, PER_DOC_CAP, EQUAL)
        graph_docs = [(store._to_document(pos), ppr[pos]) for pos in quota_positions]
        graph_v2_docs = finalize(question, graph_docs, TOP_K)

        # 3a) 消融：LLM 选实体 + 1 跳扩散（不做 PPR）
        direct: set[int] = set()
        for entity in selected:
            direct.update(entity_to_chunks.get(entity, []))
        direct_positions = graph_v2.apply_document_quota(
            store, sorted(direct), PER_DOC_CAP, EQUAL
        )
        no_ppr_docs = finalize(
            question, [(store._to_document(pos), 0.0) for pos in direct_positions], TOP_K
        )

        # 3b) 消融：字面匹配的实体 + PPR（不用 LLM 选）
        literal_seeds = keywords.candidates_for(question, entity_to_chunks)
        literal_ppr = graph_v2.personalized_pagerank(graph, literal_seeds)
        literal_positions = graph_v2.apply_document_quota(
            store,
            [pos for pos, _ in sorted(literal_ppr.items(), key=lambda kv: kv[1], reverse=True)],
            PER_DOC_CAP, EQUAL,
        )
        no_llm_docs = finalize(
            question, [(store._to_document(pos), literal_ppr[pos]) for pos in literal_positions], TOP_K
        )

        # 4) fused：图召回与向量召回 RRF 融合，再精排去重
        fused_input = rrf_fuse([dense_raw, graph_docs], top_k=EQUAL)
        fused_docs = finalize(question, fused_input, TOP_K)

        # 5) wide40：候选上限参考
        wide = finalize(question, retrieve(question, exp.MAX_CANDIDATES, principal), TOP_K)

        row = {
            "id": case["id"],
            "required": required,
            "selected_entities": selected,
            "baseline": exp.coverage(baseline, required),
            "dense8": exp.coverage(dense8, required),
            "graph_v2": exp.coverage(graph_v2_docs, required),
            "no_ppr": exp.coverage(no_ppr_docs, required),
            "no_llm": exp.coverage(no_llm_docs, required),
            "fused": exp.coverage(fused_docs, required),
            "wide40": exp.coverage(wide, required),
            "docs": {
                "baseline": baseline, "dense8": dense8,
                "graph_v2": graph_v2_docs, "no_ppr": no_ppr_docs,
                "no_llm": no_llm_docs, "fused": fused_docs, "wide40": wide,
            },
        }
        rows.append(row)
        print(f"[{case['id']}] 生产 {row['baseline']:>4.0%} | dense8 {row['dense8']:>4.0%} | "
              f"graph_v2 {row['graph_v2']:>4.0%} | 无PPR {row['no_ppr']:>4.0%} | "
              f"无LLM {row['no_llm']:>4.0%} | fused {row['fused']:>4.0%} | "
              f"wide40 {row['wide40']:>4.0%}")

    arms = ("baseline", "dense8", "graph_v2", "no_ppr", "no_llm", "fused", "wide40")
    summary = {arm: sum(row[arm] for row in rows) / len(rows) for arm in arms}
    print("\n平均文档覆盖率（5 条聚合题）：")
    for arm in arms:
        print(f"  {arm:<10} {summary[arm]:.1%}")

    (HERE / "result_v2.json").write_text(
        json.dumps({"top_k": TOP_K, "equal_budget": EQUAL, "per_doc_cap": PER_DOC_CAP,
                    "summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {(HERE / 'result_v2.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
