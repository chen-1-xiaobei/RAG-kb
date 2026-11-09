"""token 用量与成本记账的测试。

两个来源的用量字段不一样（LangChain 标准 vs OpenAI 风格），还有"拿不到"的情况。
这里把它们都固定下来：拿不到的必须记一个缺失计数，而不是静默当作 0——
"成本统计一直是 0"和"这个 provider 不回用量"是两件事。
"""
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import obs  # noqa: E402
import llm_usage  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_metrics():
    obs.reset()
    yield
    obs.reset()


def test_extract_from_langchain_usage_metadata():
    message = AIMessage(content="hi", usage_metadata={
        "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
    })
    assert llm_usage.extract_usage(message) == {"prompt": 120, "completion": 30}


def test_extract_from_openai_style_token_usage():
    message = AIMessage(content="hi", response_metadata={
        "token_usage": {"prompt_tokens": 200, "completion_tokens": 40},
    })
    assert llm_usage.extract_usage(message) == {"prompt": 200, "completion": 40}


def test_extract_returns_none_when_absent():
    assert llm_usage.extract_usage(AIMessage(content="hi")) is None
    assert llm_usage.extract_usage(None) is None


def test_record_increments_token_counters_per_direction():
    message = AIMessage(content="hi", usage_metadata={
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    })
    llm_usage.record_usage(message, stage="agent_decision")
    rendered = obs.render_prometheus()
    assert 'rag_llm_tokens_total{direction="prompt"' in rendered
    assert 'direction="completion"' in rendered
    assert 'stage="agent_decision"' in rendered
    assert " 10" in rendered and " 5" in rendered


def test_missing_usage_is_counted_not_silently_zero():
    llm_usage.record_usage(AIMessage(content="hi"), stage="answer_generate")
    rendered = obs.render_prometheus()
    assert 'rag_llm_usage_missing_total{stage="answer_generate"} 1' in rendered
    # 没有用量就不该出现 token 计数，避免把"不知道"记成 0
    assert "rag_llm_tokens_total" not in rendered


def _stub_settings(monkeypatch, *, price_in: float, price_out: float, currency: str = "CNY"):
    """settings 是 frozen dataclass，不能直接改字段——换成替身对象。"""
    from types import SimpleNamespace

    monkeypatch.setattr(llm_usage, "settings", SimpleNamespace(
        llm_price_input_per_million=price_in,
        llm_price_output_per_million=price_out,
        llm_price_currency=currency,
        chat_model="stub-model",
    ))


def test_cost_is_not_emitted_when_price_unset(monkeypatch):
    """没配价格时不输出成本指标——一个恒为 0 的成本会误导人。"""
    _stub_settings(monkeypatch, price_in=0.0, price_out=0.0)
    message = AIMessage(content="hi", usage_metadata={
        "input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000,
    })
    llm_usage.record_usage(message, stage="agent_decision")
    assert "rag_llm_cost_total" not in obs.render_prometheus()


def test_cost_is_computed_from_configured_prices(monkeypatch):
    _stub_settings(monkeypatch, price_in=1.0, price_out=2.0)
    message = AIMessage(content="hi", usage_metadata={
        "input_tokens": 1_000_000, "output_tokens": 500_000, "total_tokens": 1_500_000,
    })
    llm_usage.record_usage(message, stage="answer_generate")
    # 1M 输入 × 1 元 + 0.5M 输出 × 2 元 = 2 元
    assert 'rag_llm_cost_total{currency="CNY",stage="answer_generate"} 2' in obs.render_prometheus()


def test_estimate_cost_returns_none_without_prices(monkeypatch):
    _stub_settings(monkeypatch, price_in=0.0, price_out=0.0)
    assert llm_usage.estimate_cost(1_000_000, 1_000_000) is None
