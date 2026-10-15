"""评测指标函数单测（纯计算，无 IO、无模型）。

本文件依赖的模块图必须是"纯"的：只 import eval_metrics。
`test_eval_dataset.py` 曾在本文件的位置 import evaluate 来取指标函数，
而 evaluate 会连带 import 检索/模型链路，导致一个纯数据校验在测试收集阶段
就去加载 embedding 模型——CI 上因此暴露过一次密钥依赖问题。
指标测试与数据集校验分开、且都不碰模型，是为了让测试套件始终能离线秒跑。
"""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_metrics import grades_for, ndcg, reciprocal_rank  # noqa: E402


def test_ndcg_rewards_top_rank():
    """主来源排第一位得满分；主次排反要明确掉分（这正是分级相关性的意义）。"""
    grades = {"a": 2, "b": 1}
    assert ndcg(["a", "b"], grades) == pytest.approx(1.0)
    # ideal DCG = 2/log2(2) + 1/log2(3) = 2.6309；actual = 1/1 + 2/1.585 = 2.2619
    assert ndcg(["b", "a"], grades) == pytest.approx(2.2619 / 2.6309, abs=1e-4)
    assert ndcg(["b", "a"], grades) < ndcg(["a", "b"], grades)
    assert ndcg(["c"], grades) == 0.0


def test_ndcg_handles_single_relevant_source():
    """单来源用例：排第 1 得 1.0，排第 2 明显掉分，排在 k 之外得 0。"""
    assert ndcg(["x"], {"x": 2}) == pytest.approx(1.0)
    assert ndcg(["y", "x"], {"x": 2}) == pytest.approx(1 / math.log2(3), abs=1e-6)
    assert ndcg([f"filler{i}" for i in range(10)], {"x": 2}) == 0.0


def test_ndcg_is_bounded_and_monotonic():
    grades = {"a": 2}
    scores = [ndcg(["a"] + ["f"] * rank, grades) for rank in range(4)]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"b"}) == 0.0


def test_grades_for_prefers_explicit_grading():
    """给了 relevant_sources 用分级权重；否则 required_sources 一律按 1 计。

    两者语义不同：required_sources 是召回门槛，relevant_sources 是排序质量，
    因此后者允许包含"确实相关但非必需"的次要来源。
    """
    assert grades_for({"required_sources": ["a", "b"]}) == {"a": 1, "b": 1}
    assert grades_for({"required_sources": ["a"], "relevant_sources": {"a": 2, "b": 1}}) == {
        "a": 2,
        "b": 1,
    }
    assert grades_for({"required_sources": []}) == {}
