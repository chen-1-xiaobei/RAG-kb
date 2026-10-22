"""流式回答的单测：SSE 格式、引用解析、生成器行为。

全部离线：用一个假的异步文本块迭代器代替真实模型。
"""
import asyncio
import json

import pytest

from streaming import (
    EVENT_DONE,
    EVENT_SOURCES,
    EVENT_TOKEN,
    StreamResult,
    chunk_text,
    parse_citations,
    source_dict,
    sse,
    stream_answer,
)


class FakeDoc:
    def __init__(self, doc_id, source="报销流程.md"):
        self.metadata = {
            "id": doc_id,
            "source": source,
            "relative_path": f"财务/{source}",
            "folder": "财务",
            "group": "财务",
            "title": "费用报销流程",
            "section": "报销步骤",
            "page": None,
            "para": 1,
        }
        self.page_content = "正文"


async def fake_chunks(pieces):
    for piece in pieces:
        yield piece


def collect(pieces, hits, conversation_id="conv-1"):
    """同步驱动异步生成器，避免为几条用例引入 pytest-asyncio。"""
    result = StreamResult()

    async def run():
        events = []
        async for event in stream_answer(fake_chunks(pieces), hits, conversation_id, result):
            events.append(event)
        return events

    return asyncio.run(run()), result


def parse_event(raw):
    """把一条 SSE 文本解析回 (event, data)。"""
    lines = [line for line in raw.split("\n") if line]
    event = next(line[len("event: "):] for line in lines if line.startswith("event: "))
    data = next(line[len("data: "):] for line in lines if line.startswith("data: "))
    return event, json.loads(data)


# ---------- SSE 格式 ----------

def test_sse_format_and_utf8():
    raw = sse("token", {"text": "报销标准"})
    assert raw.startswith("event: token\n")
    assert raw.endswith("\n\n")
    assert "报销标准" in raw, "中文必须原样输出，不能被转义成 \\uXXXX"
    assert parse_event(raw) == ("token", {"text": "报销标准"})


# ---------- 文本块提取 ----------

def test_chunk_text_handles_str_and_blocks():
    class Str:
        content = "文本"

    class Blocks:
        content = [{"type": "text", "text": "片段一"}, {"type": "text", "text": "片段二"}]

    class Raw:
        content = ["a", "b"]

    assert chunk_text(Str()) == "文本"
    assert chunk_text(Blocks()) == "片段一 片段二".replace(" ", "")
    assert chunk_text(Raw()) == "ab"
    assert chunk_text(object()) == ""


# ---------- 引用解析（服务端校验） ----------

def test_parse_citations_basic():
    assert parse_citations("依据 [1] 和 [2]。", 3) == [1, 2]


def test_parse_citations_drops_out_of_range():
    """越界引用直接丢弃：模型只能引用真实存在的证据。"""
    assert parse_citations("依据 [1] 和 [99] 以及 [0]。", 2) == [1]


def test_parse_citations_dedupes_and_sorts():
    assert parse_citations("[2][1][2][1]", 5) == [1, 2]


def test_parse_citations_without_markers():
    assert parse_citations("没有引用的一段话。", 5) == []


def test_parse_citations_ignores_malformed():
    assert parse_citations("[abc] [1", 5) == []


# ---------- 来源映射 ----------

def test_source_dict_matches_source_model_fields():
    item = source_dict(FakeDoc("d1"), 4.82134, 1)
    assert set(item) == {
        "citation", "id", "source", "relative_path", "folder", "group",
        "title", "section", "page", "para", "score",
    }
    assert item["citation"] == 1
    assert item["score"] == 4.8213          # 四舍五入到 4 位
    assert item["group"] == "财务"


# ---------- 生成器 ----------

def test_stream_emits_tokens_then_sources_then_done():
    hits = [(FakeDoc("d1"), 5.0), (FakeDoc("d2", "报销标准.csv"), 4.0)]
    events, result = collect(["报销", "需在 [1] 三十天内", "提交 [2]。"], hits)

    parsed = [parse_event(raw) for raw in events]
    names = [name for name, _ in parsed]
    assert names[:3] == [EVENT_TOKEN] * 3
    assert names[-2:] == [EVENT_SOURCES, EVENT_DONE]

    tokens = "".join(data["text"] for name, data in parsed if name == EVENT_TOKEN)
    assert tokens == "报销需在 [1] 三十天内提交 [2]。"
    assert result.text == tokens
    assert result.citations == [1, 2]
    assert [item["citation"] for item in result.sources] == [1, 2]


def test_stream_validates_citations_in_sources():
    """模型写了 [99]，sources 里不应出现它——校验发生在服务端。"""
    hits = [(FakeDoc("d1"), 5.0)]
    events, result = collect(["依据 [1] 与 [99]。"], hits)

    sources = parse_event(events[-2])[1]["sources"]
    assert [item["citation"] for item in sources] == [1]
    assert result.citations == [1]


def test_stream_without_citations_yields_empty_sources():
    hits = [(FakeDoc("d1"), 5.0)]
    events, result = collect(["无法确认。"], hits)

    sources = parse_event(events[-2])[1]["sources"]
    done = parse_event(events[-1])[1]
    assert sources == []
    assert done["citations"] == []
    assert done["conversation_id"] == "conv-1"


def test_stream_skips_empty_chunks():
    hits = [(FakeDoc("d1"), 5.0)]
    events, result = collect(["a", "", "b"], hits)
    tokens = [parse_event(raw)[1]["text"] for raw in events
              if parse_event(raw)[0] == EVENT_TOKEN]
    assert tokens == ["a", "b"]
    assert result.text == "ab"
