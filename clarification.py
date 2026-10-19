"""反问（澄清）判定：把"该答 / 该问 / 该拒"做成纯函数，可离线单测。

为什么反问必须走工具通道而不是让模型自由作答：
如果模型可以随意"反问一下"，它就成了一条绕过证据门槛的路——永远不调检索、
用反问搪塞。因此反问与检索一样必须有结构性依据，并且判定规则写死在这里：

    有证据                    -> 正常作答（反问被忽略，硬门槛优先）
    无证据 + 调了反问工具      -> 反问
    无证据 + 没调反问工具      -> 拒答

`ask_clarification` 的 JSON 结果里带一个保留键 `_clarification`，由工具返回、
本模块只读。这个键同时起标记作用：写在工具结果里，而不是靠工具的副作用，
因此单元测试不需要任何运行时状态就能验证判定结果。
"""
from __future__ import annotations

import json

CLARIFY_TOOL_NAME = "ask_clarification"
CLARIFICATION_KEY = "_clarification"

REFUSAL_TEXT = "无法从现有知识库确认该问题。"
# 追问内容的长度上限：防止模型把整段答案伪装成"问题"塞回来
MAX_CLARIFICATION_CHARS = 300


def clarify_payload(question: str) -> str:
    """工具返回值：结构化的追问载荷（给模型看的是这句话本身）。"""
    return json.dumps(
        {CLARIFICATION_KEY: question.strip()[:MAX_CLARIFICATION_CHARS]},
        ensure_ascii=False,
    )


def extract_clarification(tool_contents: list[str]) -> str:
    """从工具结果里取出上一次反问的内容；没有则返回空串。"""
    for content in reversed(tool_contents):
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and CLARIFICATION_KEY in payload:
            text = str(payload[CLARIFICATION_KEY]).strip()
            if text:
                return text[:MAX_CLARIFICATION_CHARS]
    return ""


def decide(evidence_count: int, ask_succeeded: bool, tool_contents: list[str]) -> tuple[str, str]:
    """返回 (动作, 文本)。动作取值：answer / clarify / refuse。

    - answer：有证据，交给生成节点（反问忽略）
    - clarify：无证据且反问成功，直接返回追问、不调用 LLM
    - refuse：无证据且没反问成功，拒答
    """
    if evidence_count > 0:
        return "answer", ""
    if ask_succeeded:
        text = extract_clarification(tool_contents)
        if text:
            return "clarify", text
    return "refuse", REFUSAL_TEXT
