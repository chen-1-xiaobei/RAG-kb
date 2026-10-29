"""对比实验：普通 top-k 检索 vs 实体图检索，在全局聚合题上的文档覆盖率。

    python experiments/graphrag/run_experiment.py

两种方法用**同样的 chunk 预算**（4 个 chunk）与**同一个精排模型**，
唯一变量是"候选从哪来"：

- baseline：向量检索 top-8 → 精排 → 取 4 个 chunk
- graph：问题里出现的实体 → 这些实体的全部 chunk（+1 跳邻居）→ 精排
         → **每份文档只保留最好的那个 chunk** → 按文档取前 4 份

最后一步（按文档去重）是两种方法共有的收敛步骤，但只有图召回能提供"覆盖多份文档"
的候选池——向量 top-8 天然集中在少数几段最像的文本上。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from auth import Principal  # noqa: E402
from rerank import rerank  # noqa: E402
from retrieval import index_manager, retrieve_ranked  # noqa: E402

HERE = Path(__file__).resolve().parent
TOP_K = 4
MAX_CANDIDATES = 40        # 图召回候选上限，防止爆炸
ONE_HOP = True             # 是否扩展到共现实体的邻居 chunk


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(entities_payload: dict) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """返回 (实体 -> chunk 下标列表, chunk id -> 实体列表)。"""
    entity_to_chunks: dict[str, list[int]] = {}
    chunk_entities: dict[str, list[str]] = {}

    store = index_manager.get()
    id_to_position = {record["id"]: i for i, record in enumerate(store._records)}

    for entry in entities_payload["chunks"]:
        position = id_to_position.get(entry["id"])
        if position is None:
            continue
        chunk_entities[entry["id"]] = entry["entities"]
        for entity in entry["entities"]:
            entity_to_chunks.setdefault(entity, []).append(position)
    return entity_to_chunks, chunk_entities


def _bigrams(text: str) -> set[str]:
    return {text[index:index + 2] for index in range(len(text) - 1)}


def match_entities(question: str, entity_to_chunks: dict[str, list[int]]) -> list[str]:
    """问题与实体命中的判定：字面包含 **或** 二元组重叠。

    只用「实体是问题的子串」太严：抽取出的实体常常更长（「五个工作日」），
    而问题里说的是「工作日」，子串方向反了就一个都匹配不到——第一版就是这么
    在 5 个问题里漏了 4 个。改成二元组重叠后，「五个工作日」的二元组
    「工作/作日」能在问题里找到，才会被算命中。
    """
    question_bigrams = _bigrams(question)
    matched = []
    for entity in entity_to_chunks:
        if entity in question or _bigrams(entity) & question_bigrams:
            matched.append(entity)
    return matched


def graph_candidates(question: str, entity_to_chunks: dict, chunk_entities: dict,
                     store) -> tuple[list[int], list[str]]:
    matched = match_entities(question, entity_to_chunks)
    positions: set[int] = set()
    for entity in matched:
        positions.update(entity_to_chunks[entity])

    if ONE_HOP and matched:
        # 1 跳：与命中实体共现过的其它实体，把它们挂的 chunk 也拉进来
        neighbours: set[str] = set()
        id_to_position = {record["id"]: i for i, record in enumerate(store._records)}
        for entry_id, entities in chunk_entities.items():
            if id_to_position[entry_id] in positions:
                neighbours.update(entities)
        neighbours -= set(matched)
        for entity in neighbours:
            positions.update(entity_to_chunks.get(entity, []))

    return sorted(positions)[:MAX_CANDIDATES], matched


def best_per_document(docs: list[tuple]) -> list[str]:
    """按精排分数保留每份文档最好的 chunk，返回按分数排序的文档名。"""
    best: dict[str, float] = {}
    for doc, score in docs:
        source = doc.metadata.get("source", "")
        if source not in best or score > best[source]:
            best[source] = score
    return [name for name, _ in sorted(best.items(), key=lambda item: item[1], reverse=True)]


def coverage(ranked: list[str], required: list[str]) -> float:
    return sum(1 for name in required if name in ranked) / len(required)


def rank_documents(question: str, candidates: list, budget: int) -> list[str]:
    """精排 -> 每份文档只留最好的 chunk -> 按分数取前 budget 份文档。"""
    if not candidates:
        return []
    scored = rerank(question, candidates, top_n=len(candidates), threshold=-999.0)
    return best_per_document(scored)[:budget]


def main() -> int:
    questions = load(HERE / "questions.json")["cases"]
    entities_payload = load(HERE / "entities.json")
    entity_to_chunks, chunk_entities = build_graph(entities_payload)
    store = index_manager.get()
    from retrieval import retrieve

    EQUAL = 8          # 等预算对比：两种方法都只看 8 个候选

    rows = []
    for case in questions:
        principal = Principal("graphrag-eval", frozenset(case["groups"]))
        required = case["required_sources"]
        question = case["question"]

        # arm 1：生产同款（dense 8 -> 精排 -> top-4 chunk，不做文档去重）
        started = time.perf_counter()
        hits = retrieve_ranked(question, TOP_K, principal)
        baseline_ms = (time.perf_counter() - started) * 1000
        baseline_ranked = list(dict.fromkeys(doc.metadata["source"] for doc, _ in hits))
        baseline_cov = coverage(baseline_ranked, required)

        # arm 2：dense 8 + 文档去重（与 arm 3 等预算，唯一差别是没有图）
        started = time.perf_counter()
        dense8_raw = retrieve(question, EQUAL, principal)
        dense8_ranked = rank_documents(question, dense8_raw, TOP_K)
        dense8_cov = coverage(dense8_ranked, required)
        dense8_ms = (time.perf_counter() - started) * 1000

        # arm 3：图召回，同样截断到 8 个候选
        started = time.perf_counter()
        positions, matched = graph_candidates(question, entity_to_chunks, chunk_entities, store)
        graph8_positions = positions[:EQUAL]
        if graph8_positions:
            graph8_ranked = rank_documents(
                question, [(store._to_document(i), 0.0) for i in graph8_positions], TOP_K
            )
        else:
            graph8_ranked = []
        graph8_cov = coverage(graph8_ranked, required)
        graph8_ms = (time.perf_counter() - started) * 1000

        # arm 4：宽候选（40）——用来看"放宽候选到底能到多少"
        started = time.perf_counter()
        wide_ranked = rank_documents(question, retrieve(question, MAX_CANDIDATES, principal), TOP_K)
        wide_cov = coverage(wide_ranked, required)
        wide_ms = (time.perf_counter() - started) * 1000

        rows.append({
            "id": case["id"],
            "required": required,
            "baseline_coverage": baseline_cov,
            "dense8_coverage": dense8_cov,
            "graph8_coverage": graph8_cov,
            "wide_coverage": wide_cov,
            "baseline_docs": baseline_ranked,
            "dense8_docs": dense8_ranked,
            "graph8_docs": graph8_ranked,
            "wide_docs": wide_ranked,
            "matched_entities": matched[:8],
            "graph_candidates": len(positions),
            "baseline_ms": round(baseline_ms, 1),
            "dense8_ms": round(dense8_ms, 1),
            "graph8_ms": round(graph8_ms, 1),
            "wide_ms": round(wide_ms, 1),
        })
        print(f"[{case['id']}] 生产 {baseline_cov:>5.0%} | dense8 {dense8_cov:>5.0%} | "
              f"graph8 {graph8_cov:>5.0%} | wide40 {wide_cov:>5.0%}")

    def mean(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    summary = {
        "baseline": mean("baseline_coverage"),
        "dense8": mean("dense8_coverage"),
        "graph8": mean("graph8_coverage"),
        "wide40": mean("wide_coverage"),
    }
    print("\n平均文档覆盖率：")
    for name, value in summary.items():
        print(f"  {name:<10} {value:.1%}")

    (HERE / "result.json").write_text(
        json.dumps({
            "top_k": TOP_K,
            "equal_budget": EQUAL,
            "max_candidates": MAX_CANDIDATES,
            "summary": summary,
            "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {(HERE / 'result.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
