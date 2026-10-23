"""cassette（录制/回放）的单测：键稳定性、严格回放、顺序消费、流式。

核心安全语义只有一条：**回放时查不到记录必须报错，不能静默返回空**。
"测试全绿但其实什么都没测"是最危险的失败模式。
"""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cassette import (
    CASSETTE_VERSION,
    CassetteMissing,
    CassetteModel,
    CassetteStore,
    cassette_config,
    message_to_dict,
    request_key,
    wrap,
)


class FakeReal:
    """按顺序返回预设响应的假模型。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self._responses.pop(0)

    async def astream(self, messages, **kwargs):
        for piece in self._responses.pop(0):
            yield piece


class FakeTool:
    def __init__(self, name):
        self.name = name


def _store(tmp_path, mode):
    return CassetteStore(tmp_path / "c.json", mode)


def _empty_replay(tmp_path):
    """回放模式要求文件存在（缺文件是配置错误，应当立刻失败），
    所以这里先写一个"有文件但没有记录"的空固件。"""
    import json as _json
    (tmp_path / "c.json").write_text(
        _json.dumps({"version": CASSETTE_VERSION, "entries": []}), encoding="utf-8"
    )
    return CassetteStore(tmp_path / "c.json", "replay")


# ---------- 消息归一化 ----------

def test_message_to_dict_roles():
    assert message_to_dict(SystemMessage("s"))["role"] == "system"
    assert message_to_dict(HumanMessage("h"))["role"] == "human"
    assert message_to_dict(AIMessage("a"))["role"] == "ai"

    tool = ToolMessage(content="结果", tool_call_id="c1")
    payload = message_to_dict(tool)
    assert payload["role"] == "tool" and payload["tool_call_id"] == "c1"


def test_message_to_dict_keeps_tool_calls():
    message = AIMessage(content="", tool_calls=[
        {"name": "search_knowledge_base", "args": {"question": "q"}, "id": "c1"},
    ])
    payload = message_to_dict(message)
    assert payload["tool_calls"][0]["name"] == "search_knowledge_base"
    assert payload["tool_calls"][0]["args"] == {"question": "q"}


def test_key_ignores_volatile_metadata():
    """`additional_kwargs` 这类每次都不一样的字段必须排除，否则键永远对不上。"""
    plain = AIMessage(content="同样内容")
    noisy = AIMessage(content="同样内容", additional_kwargs={"token_usage": 123})
    assert request_key("m", [plain], {"t"}) == request_key("m", [noisy], {"t"})


# ---------- 键的稳定性与敏感度 ----------

def test_key_is_stable_across_calls():
    messages = [HumanMessage("报销要几天内提交？")]
    assert request_key("m", messages, {"a", "b"}) == request_key("m", messages, {"b", "a"})


def test_key_changes_with_tool_set():
    messages = [HumanMessage("q")]
    assert request_key("m", messages, {"search"}) != request_key("m", messages, {"search", "table"})


def test_key_changes_with_content_and_model():
    assert request_key("m", [HumanMessage("a")], set()) != request_key("m", [HumanMessage("b")], set())
    assert request_key("m1", [HumanMessage("a")], set()) != request_key("m2", [HumanMessage("a")], set())


# ---------- 录制 -> 回放 ----------

def test_record_then_replay_roundtrip(tmp_path):
    response = AIMessage(content="", tool_calls=[
        {"name": "search_knowledge_base", "args": {"question": "q"}, "id": "c1"},
    ])
    recording = CassetteModel(_store(tmp_path, "record"), FakeReal([response]), model_name="m")
    recording.bind_tools([FakeTool("search_knowledge_base")]).invoke([HumanMessage("问题")])

    store = _store(tmp_path, "replay")
    replaying = CassetteModel(store, model_name="m")
    replayed = replaying.bind_tools([FakeTool("search_knowledge_base")]).invoke([HumanMessage("问题")])

    assert replayed.content == ""
    assert replayed.tool_calls[0]["name"] == "search_knowledge_base"
    assert replayed.tool_calls[0]["args"] == {"question": "q"}
    assert replayed.tool_calls[0]["id"] == "c1"


def test_replay_missing_record_raises(tmp_path):
    """没有任何记录时回放必须抛错，而不是返回空答案。"""
    replaying = CassetteModel(_empty_replay(tmp_path), model_name="m")
    with pytest.raises(CassetteMissing):
        replaying.invoke([HumanMessage("从没录过的问题")])


def test_replay_exhausted_raises(tmp_path):
    recording = CassetteModel(_store(tmp_path, "record"), FakeReal([AIMessage("一")]), model_name="m")
    recording.invoke([HumanMessage("q")])

    replaying = CassetteModel(_store(tmp_path, "replay"), model_name="m")
    assert replaying.invoke([HumanMessage("q")]).content == "一"
    with pytest.raises(CassetteMissing):
        replaying.invoke([HumanMessage("q")])   # 只录了一次，第二次必须失败


def test_same_key_consumed_in_order(tmp_path):
    """同一输入被调用多次且输出不同时，按到达顺序消费。"""
    recording = CassetteModel(
        _store(tmp_path, "record"), FakeReal([AIMessage("第一次"), AIMessage("第二次")]), model_name="m",
    )
    recording.invoke([HumanMessage("q")])
    recording.invoke([HumanMessage("q")])

    replaying = CassetteModel(_store(tmp_path, "replay"), model_name="m")
    assert replaying.invoke([HumanMessage("q")]).content == "第一次"
    assert replaying.invoke([HumanMessage("q")]).content == "第二次"


def test_record_mode_delegates_to_real_model(tmp_path):
    real = FakeReal([AIMessage("真实回答")])
    model = CassetteModel(_store(tmp_path, "record"), real, model_name="m")
    assert model.invoke([HumanMessage("q")]).content == "真实回答"
    assert real.calls == 1


# ---------- 流式 ----------

def test_stream_record_then_replay(tmp_path):
    real = FakeReal([["员工需在 ", "三十天内 ", "提交报销单。"]])
    recording = CassetteModel(_store(tmp_path, "record"), real, model_name="m")

    async def record_stream():
        return [piece async for piece in recording.astream([HumanMessage("q")])]

    collected = asyncio.run(record_stream())
    assert "".join(collected) == "员工需在 三十天内 提交报销单。"

    replaying = CassetteModel(_store(tmp_path, "replay"), model_name="m", replay_chunk=4)

    async def replay_stream():
        return [piece async for piece in replaying.astream([HumanMessage("q")])]

    replayed = asyncio.run(replay_stream())
    assert "".join(replayed) == "员工需在 三十天内 提交报销单。"
    assert len(replayed) > 1, "回放应分块产出，而不是一次性吐完"


# ---------- 文件格式与配置 ----------

def test_version_mismatch_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": CASSETTE_VERSION + 1, "entries": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        CassetteStore(path, "replay")


def test_replay_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        CassetteStore(tmp_path / "missing.json", "replay")


def test_invalid_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        CassetteStore(tmp_path / "c.json", "whatever")


def test_cassette_file_is_readable_json(tmp_path):
    """固件要能被人读懂和 review，不能是二进制或转义过的中文。"""
    recording = CassetteModel(_store(tmp_path, "record"), FakeReal([AIMessage("中文回答")]), model_name="m")
    recording.invoke([HumanMessage("报销流程")])

    text = (tmp_path / "c.json").read_text(encoding="utf-8")
    assert "中文回答" in text and "报销流程" in text
    assert json.loads(text)["version"] == CASSETTE_VERSION


def test_structured_output_is_explicitly_unsupported(tmp_path):
    """回放不支持结构化输出时要明确失败，而不是返回一个看似能用的对象。"""
    model = CassetteModel(_empty_replay(tmp_path), model_name="m")
    with pytest.raises(NotImplementedError):
        model.with_structured_output(dict)


def test_wrap_is_noop_without_env(monkeypatch):
    monkeypatch.delenv("LLM_CASSETTE", raising=False)
    assert cassette_config() is None
    real = object()
    assert wrap(real) is real


def test_wrap_activates_with_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CASSETTE", str(tmp_path / "c.json"))
    monkeypatch.setenv("LLM_CASSETTE_MODE", "record")
    wrapped = wrap(FakeReal([]), model_name="m")
    assert isinstance(wrapped, CassetteModel)
