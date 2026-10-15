"""检索评测的纯函数部分：指标计算与相关性映射，不依赖 embedding / reranker / app。

单独成模块的理由（而不是塞在 evaluate.py 里）：
- `evaluate.py` 需要 import `app` 才能拿到真实检索管线，而 `app` 在 import 期就会构造
  embedding/LLM 客户端。若指标函数留在 evaluate.py，任何想复用指标的测试或脚本
  都会被拖进模型链路——离线单测会因此依赖模型缓存与网络环境。
- 这里没有任何 IO 与全局状态，可被 tests/ 直接 import，也可单独在 CI 中跑。
"""
from __future__ import annotations

import math

NDCG_K = 10


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def reciprocal_rank(ranked: list[str], expected: set[str]) -> float:
    """第一个相关项排名的倒数；一个都没召回时为 0。"""
    for position, source in enumerate(ranked, 1):
        if source in expected:
            return 1.0 / position
    return 0.0


def dcg(gains: list[float], k: int) -> float:
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains[:k], 1))


def ndcg(ranked: list[str], grades: dict[str, int], k: int = NDCG_K) -> float:
    """nDCG@k：grades 为 source -> 相关性等级（>0 表示相关）。

    理想排序 = 所有相关来源按等级降序排在最前。没有任何相关来源时返回 0，
    调用方对"本就不该召回"的用例跳过该指标。
    """
    actual = [float(grades.get(source, 0)) for source in ranked[:k]]
    ideal = sorted((float(grade) for grade in grades.values() if grade > 0), reverse=True)
    perfect = dcg(ideal, k)
    return dcg(actual, k) / perfect if perfect else 0.0


def grades_for(case: dict) -> dict[str, int]:
    """相关性映射：给了 relevant_sources 用分级权重，否则 required_sources 全部按 1 计。

    required_sources 是召回门槛（必须命中），relevant_sources 是排序质量（谁更该排前面），
    两者可以不同：次要来源可能确实相关、但缺了它答案依然完整。
    """
    graded = case.get("relevant_sources")
    if graded:
        return {source: int(grade) for source, grade in graded.items()}
    return {source: 1 for source in case.get("required_sources", [])}
