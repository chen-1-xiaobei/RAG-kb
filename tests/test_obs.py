"""可观测性：请求上下文注入日志 + 指标渲染。"""
import logging
import re

import obs


def setup_function() -> None:
    obs.reset()
    obs.reset_context()


def test_counter_renders_prometheus_text():
    obs.inc("rag_requests_total", status="ok")
    obs.inc("rag_requests_total", status="ok")
    obs.inc("rag_requests_total", status="refused")

    text = obs.render_prometheus()
    assert "# TYPE rag_requests_total counter" in text
    assert 'rag_requests_total{status="ok"} 2' in text
    assert 'rag_requests_total{status="refused"} 1' in text


def test_summary_renders_quantiles_count_and_sum():
    for value in (1.0, 2.0, 3.0, 4.0, 10.0):
        obs.observe("rag_latency_seconds", value, stage="rerank")

    text = obs.render_prometheus()
    assert "# TYPE rag_latency_seconds summary" in text
    assert 'rag_latency_seconds{stage="rerank",quantile="0.5"} 3.0000' in text
    assert 'rag_latency_seconds_count{stage="rerank"} 5' in text
    assert 'rag_latency_seconds_sum{stage="rerank"} 20.0000' in text


def test_every_line_is_valid_exposition_format():
    """每一行要么是注释，要么是 `指标名{标签} 值`。格式错了 Prometheus 会拒绝整份。"""
    obs.inc("rag_tool_calls_total", tool="search_knowledge_base", status="ok")
    obs.observe("rag_stage_seconds", 0.42, stage="coarse")

    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})? -?[0-9.eE+]+$")
    for line in obs.render_prometheus().strip().splitlines():
        if line.startswith("#"):
            assert line.split()[0] in {"#", "# TYPE"}
            continue
        assert pattern.match(line), f"格式不合法: {line}"


def test_label_order_is_stable():
    obs.inc("x_total", a="1", b="2")
    obs.inc("x_total", b="2", a="1")
    text = obs.render_prometheus()
    # 两次传标签的书写顺序不同，但必须归一到同一个序列、只算一条
    assert text.count('x_total{a="1",b="2"}') == 1
    assert 'x_total{a="1",b="2"} 2' in text


def test_unknown_metric_renders_nothing():
    assert obs.render_prometheus() == "\n"


def test_context_filter_injects_ids_into_records():
    obs.set_request_id("rid-abc")
    obs.bind_conversation("conv-42")

    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hi", None, None)
    assert obs.ContextFilter().filter(record) is True
    assert record.rid == "rid-abc"
    assert record.conv == "conv-42"


def test_default_context_is_unknown_not_empty():
    """没有上下文时用 `-`：空值会让日志看起来像是少了一列。"""
    assert obs.current_request_id() == obs.UNKNOWN
    assert obs.current_conversation_id() == obs.UNKNOWN


def test_install_logging_puts_ids_in_the_line(capsys):
    obs.install_logging()
    obs.set_request_id("rid-xyz")
    obs.bind_conversation("conv-9")

    logging.getLogger("rag-kb").info("hello")
    err = capsys.readouterr().err
    assert "rid=rid-xyz" in err
    assert "conv=conv-9" in err


def test_install_logging_replaces_duplicate_handlers():
    """反复调用不应叠加 handler，否则每行日志会打印多遍。"""
    obs.install_logging()
    obs.install_logging()
    assert len(logging.getLogger().handlers) == 1


def test_reset_clears_metrics():
    obs.inc("a_total")
    obs.reset()
    assert obs.render_prometheus() == "\n"
