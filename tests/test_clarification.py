"""反问（澄清）判定单测：纯函数，不加载模型、不调用 LLM。

判定规则就是证据门槛在"反问"这个新出口上的延伸，因此这里的每一条断言
都对应一个安全语义，而不只是分支覆盖。
"""
import json

import pytest

from clarification import (
    CLARIFICATION_KEY,
    MAX_CLARIFICATION_CHARS,
    REFUSAL_TEXT,
    clarify_payload,
    decide,
    extract_clarification,
)


def test_evidence_wins_over_clarification():
    """有证据就必须作答：反问被忽略。否则"反问"会成为绕过证据门槛的出口。"""
    action, text = decide(2, True, [clarify_payload("你指的是哪个？")])
    assert (action, text) == ("answer", "")


def test_no_evidence_with_clarification_asks():
    action, text = decide(0, True, [clarify_payload("你指的是出差住宿还是会议住宿？")])
    assert action == "clarify"
    assert "出差住宿还是会议住宿" in text


def test_no_evidence_without_clarification_refuses():
    """模型既没检索也没反问 -> 拒答，与原有证据门槛行为一致。"""
    action, text = decide(0, False, [])
    assert (action, text) == ("refuse", REFUSAL_TEXT)


def test_failed_clarification_falls_back_to_refusal():
    """反问工具报错（没有可解析的载荷）时必须退化成拒答，不能漏出一个空回答。"""
    action, text = decide(0, True, ["工具执行失败: boom"])
    assert (action, text) == ("refuse", REFUSAL_TEXT)


def test_extract_takes_latest_clarification():
    action, text = decide(0, True, [
        clarify_payload("第一个问题"),
        "无关的工具输出",
        clarify_payload("第二个问题"),
    ])
    assert action == "clarify" and text == "第二个问题"


def test_payload_is_valid_json_with_reserved_key():
    payload = json.loads(clarify_payload("需要澄清什么？"))
    assert payload[CLARIFICATION_KEY] == "需要澄清什么？"


def test_clarification_length_is_capped():
    """防止模型把整段答案伪装成"问题"塞回来。"""
    long_question = "很长的追问" * 200
    text = extract_clarification([clarify_payload(long_question)])
    assert len(text) <= MAX_CLARIFICATION_CHARS


def test_clarification_payload_is_not_treated_as_evidence():
    """反问工具不产生证据：它的载荷里没有证据字段，也不该被当作 hits。

    这里做的是结构性断言——反问即便被调用，decide 在 evidence_count=0 时也只会
    给出 clarify/refuse，永远不可能进入 answer 分支。
    """
    for evidence in (0,):
        action, _ = decide(evidence, True, [clarify_payload("追问")])
        assert action != "answer"


@pytest.mark.parametrize("contents", [[], [""], ["not json"], ["[1,2,3]", '{"other": 1}']])
def test_non_clarification_contents_are_ignored(contents):
    assert extract_clarification(contents) == ""
