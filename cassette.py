"""LLM 交互的录制与回放：让不确定的模型变成可回放的文件。

**解决什么问题**

Agent 的行为依赖模型，而模型每次输出都不一样。这带来两个后果：

1. 端到端测不了——没有密钥不能跑，跑了结果也不稳定，没法进 CI；
2. 改一个提示词，你不知道是变好了还是变坏了，因为"之前的样子"没保存下来。

做法是给模型套一层"录音机"：每次调用的**输入和输出**都存进一个 JSON 文件（cassette），
回放时按输入查表返回当时那一条。于是同一次会话可以**逐字节复现**，
整个 Agent 循环（工具选择、查询改写、证据门槛）都能在 CI 里做确定性回归。

**只在必要时录制**

回放模式下查不到对应记录会直接抛 `CassetteMissing`，**绝不静默返回空答案**——
那种"测试全绿但其实什么都没测"的失败，比测试红更危险。

**当前覆盖范围（重要）**

只覆盖「绑定工具后的决策调用」（`bind_tools(...).invoke/astream`），也就是 Agent 决定
调哪个工具、用什么参数的那一步——多轮改写、工具选择、轮数控制都在这里。
答案生成走的是结构化输出（`with_structured_output`），回放模式不支持，会明确抛错。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

from langchain_core.messages import AIMessage

CASSETTE_VERSION = 1
_ENV_PATH = "LLM_CASSETTE"
_ENV_MODE = "LLM_CASSETTE_MODE"
MODES = ("record", "replay")


class CassetteMissing(LookupError):
    """回放时找不到对应记录。故意用异常而不是返回空，避免静默通过。"""


# ---------------------------------------------------------------------------
# 消息归一化与键计算
# ---------------------------------------------------------------------------

_ROLE_BY_CLASS = {
    "SystemMessage": "system",
    "HumanMessage": "human",
    "AIMessage": "ai",
    "ToolMessage": "tool",
}


def message_to_dict(message: Any) -> dict:
    """把 LangChain 消息压成稳定的 dict，用于计算键与写进 cassette。

    只保留参与决策的字段：角色、正文、工具调用、工具调用 ID。
    像 `additional_kwargs` / `response_metadata` 这类每次都不一样的元数据**必须排除**，
    否则键永远对不上，回放必然失效。
    """
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)

    payload: dict[str, Any] = {
        "role": _ROLE_BY_CLASS.get(type(message).__name__, type(message).__name__),
        "content": content,
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {"name": call.get("name"), "args": call.get("args", {}), "id": call.get("id")}
            for call in tool_calls
        ]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


def request_key(model: str, messages: Iterable[Any], tool_names: Iterable[str]) -> str:
    """一次请求的稳定指纹。

    工具名参与计算是刻意的：**增删工具会让旧 cassette 失效**，这是对的——
    工具集合变了，模型的决策依据也变了，旧记录不能代表新行为。
    """
    payload = {
        "model": model,
        "tools": sorted(tool_names),
        "messages": [message_to_dict(message) for message in messages],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------

@dataclass
class CassetteStore:
    """一个 cassette 文件 = 一次或多次会话的调用记录。

    同一个键可能对应多条记录（模型对同样的输入给过不同输出，或者流程里本来就重复调用），
    因此按**到达顺序**消费：第 n 次调用取第 n 条。用完再取就抛 `CassetteMissing`。
    """
    path: Path
    mode: str
    _entries: dict[str, list[dict]] = field(default_factory=dict)
    _cursor: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，收到 {self.mode!r}")
        self.path = Path(self.path)
        if self.mode == "replay" and not self.path.exists():
            raise FileNotFoundError(f"回放需要 cassette 文件：{self.path}")
        if self.path.exists():
            self._load()

    # ---------- 读写 ----------

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        version = data.get("version")
        if version != CASSETTE_VERSION:
            raise ValueError(f"不支持的 cassette 版本：{version}（期望 {CASSETTE_VERSION}）")
        for entry in data.get("entries", []):
            self._entries.setdefault(entry["key"], []).append(entry)

    def save(self) -> None:
        entries = [
            entry
            for key in self._entries          # 保持插入顺序，diff 才稳定
            for entry in self._entries[key]
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"version": CASSETTE_VERSION, "entries": entries},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # ---------- 记录 / 取用 ----------

    def put(self, key: str, request: dict, response: dict, note: str = "") -> None:
        entry = {"key": key, "request": request, "response": response}
        if note:
            entry["note"] = note
        self._entries.setdefault(key, []).append(entry)

    def take(self, key: str) -> dict:
        entries = self._entries.get(key)
        index = self._cursor.get(key, 0)
        if not entries or index >= len(entries):
            raise CassetteMissing(
                f"没有可用的录制记录（键 {key}，已消费 {index} 条）。"
                "请用 LLM_CASSETTE_MODE=record 重新录制。"
            )
        self._cursor[key] = index + 1
        return entries[index]

    @property
    def recorded_count(self) -> int:
        return sum(len(items) for items in self._entries.values())

    def reset_cursor(self) -> None:
        self._cursor.clear()


# ---------------------------------------------------------------------------
# 包装模型
# ---------------------------------------------------------------------------

def _response_to_dict(message: Any) -> dict:
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return {
        "content": content,
        "tool_calls": [
            {"name": call.get("name"), "args": call.get("args", {}), "id": call.get("id")}
            for call in (getattr(message, "tool_calls", None) or [])
        ],
    }


def _dict_to_message(payload: dict) -> AIMessage:
    return AIMessage(
        content=payload.get("content", ""),
        tool_calls=payload.get("tool_calls") or [],
    )


class CassetteModel:
    """套在真实模型外面的录音机。

    - record 模式：调用真实模型，把输入输出写进 cassette；
    - replay 模式：不碰网络，按输入查表返回记录内容。
    """

    def __init__(self, store: CassetteStore, real: Any = None,
                 tool_names: tuple[str, ...] = (),
                 tool_objects: Any = None, bind_kwargs: dict | None = None,
                 model_name: str = "unknown", replay_chunk: int = 8):
        self._store = store
        self._real = real
        self._tool_names = tool_names
        self._tool_objects = tool_objects
        self._bind_kwargs = bind_kwargs or {}
        self._model_name = model_name
        self._replay_chunk = replay_chunk

    # ---------- 与 LangChain 的接口对齐 ----------

    def bind_tools(self, tools: Any, **kwargs: Any) -> "CassetteModel":
        names = tuple(getattr(tool, "name", str(tool)) for tool in tools)
        return CassetteModel(
            self._store, self._real, names, tools, kwargs, self._model_name, self._replay_chunk
        )

    def with_structured_output(self, *args: Any, **kwargs: Any):
        # 结构化输出的回放没有实现：与其返回一个"看起来能用"的空对象，不如明确失败。
        raise NotImplementedError(
            "cassette 目前只覆盖绑定工具后的决策调用；答案生成走 with_structured_output，"
            "回放模式不支持。请让答案节点使用简单 stub，或在 record 模式下运行。"
        )

    # ---------- 内部 ----------

    def _key(self, messages: Any) -> str:
        return request_key(self._model_name, messages, self._tool_names)

    def _bound_real(self) -> Any:
        if self._real is None:
            return None
        if self._tool_objects is not None:
            return self._real.bind_tools(self._tool_objects, **self._bind_kwargs)
        return self._real

    @staticmethod
    def _request_payload(messages: Any, tool_names: tuple[str, ...]) -> dict:
        return {
            "model": None,
            "tools": list(tool_names),
            "messages": [message_to_dict(message) for message in messages],
        }

    # ---------- 调用 ----------

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        key = self._key(messages)
        if self._store.mode == "record":
            response = self._bound_real().invoke(messages, **kwargs)
            payload = self._request_payload(messages, self._tool_names)
            payload["model"] = self._model_name
            self._store.put(key, payload, _response_to_dict(response))
            self._store.save()
            return response
        return _dict_to_message(self._store.take(key)["response"])

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[str]:
        key = self._key(messages)
        if self._store.mode == "record":
            pieces: list[str] = []
            async for chunk in self._bound_real().astream(messages, **kwargs):
                text = getattr(chunk, "content", chunk)
                if isinstance(text, str) and text:
                    pieces.append(text)
                    yield text
            payload = self._request_payload(messages, self._tool_names)
            payload["model"] = self._model_name
            self._store.put(key, payload, {"content": "".join(pieces), "tool_calls": [],
                                           "streamed": True})
            self._store.save()
            return

        recorded = self._store.take(key)["response"]
        text = recorded.get("content", "")
        for start in range(0, len(text), self._replay_chunk):
            yield text[start:start + self._replay_chunk]


# ---------------------------------------------------------------------------
# 接入
# ---------------------------------------------------------------------------

def cassette_config() -> tuple[Path, str] | None:
    """从环境变量读取配置；未配置时返回 None（生产路径完全不受影响）。"""
    raw_path = os.getenv(_ENV_PATH)
    if not raw_path:
        return None
    mode = os.getenv(_ENV_MODE, "replay")
    return Path(raw_path), mode


def wrap(real: Any, model_name: str = "unknown") -> Any:
    """按需给真实模型套上录音机；没配置环境变量就原样返回。"""
    config = cassette_config()
    if config is None:
        return real
    path, mode = config
    return CassetteModel(CassetteStore(path, mode), real, model_name=model_name)
