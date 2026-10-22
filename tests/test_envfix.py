"""NO_PROXY 规范化的单测。

这个修补针对的是"环境导致的启动失败"：httpx 把 `[::1]` 当 URL 解析后抛 InvalidURL，
而它是合法的 no_proxy 写法。测试要保证规范化**只动方括号形式**，其余原样保留。
"""
import pytest

from envfix import _normalize_entry, sanitize_proxy_env


def test_normalize_strips_ipv6_brackets():
    assert _normalize_entry("[::1]") == "::1"
    assert _normalize_entry("[::1]:8080") == "[::1]:8080"   # 带端口不处理，避免歧义
    assert _normalize_entry("127.0.0.1") == "127.0.0.1"
    assert _normalize_entry(" localhost ") == "localhost"


def test_sanitize_fixes_bracket_entries(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1,[::1]")
    changed = sanitize_proxy_env()
    import os
    assert changed, "应当报告被修改的环境变量名"
    assert os.environ["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert "[" not in os.environ["NO_PROXY"]


def test_sanitize_leaves_clean_value_untouched(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1")
    assert sanitize_proxy_env() == []
    import os
    assert os.environ["NO_PROXY"] == "127.0.0.1,localhost,::1"


def test_sanitize_is_noop_without_variable(monkeypatch):
    import os
    for name in [k for k in os.environ if k.upper() == "NO_PROXY"]:
        monkeypatch.delenv(name, raising=False)
    assert sanitize_proxy_env() == []


def test_sanitize_drops_empty_entries(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,,localhost,[::1],")
    sanitize_proxy_env()
    import os
    assert os.environ["NO_PROXY"] == "127.0.0.1,localhost,::1"


def test_sanitize_dedupes_after_normalizing(monkeypatch):
    """`::1` 与 `[::1]` 规范化后是同一个地址，不应保留两份。"""
    monkeypatch.setenv("NO_PROXY", "::1,[::1]")
    sanitize_proxy_env()
    import os
    assert os.environ["NO_PROXY"] == "::1"


def test_sanitize_preserves_other_entries(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*.internal,10.0.0.0/8,[::1],[fe80::1]")
    sanitize_proxy_env()
    import os
    assert os.environ["NO_PROXY"] == "*.internal,10.0.0.0/8,::1,fe80::1"


def test_sanitize_makes_httpx_accept_the_value(monkeypatch):
    """真正的回归点：规范化后 httpx 必须能建客户端。"""
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1,[::1]")
    with pytest.raises(Exception):
        httpx.Client()          # 未规范化时确实会炸

    sanitize_proxy_env()
    client = httpx.Client()
    client.close()
