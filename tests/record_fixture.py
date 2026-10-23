"""重新生成多轮回放固件（`tests/cassettes/multiturn_rewrite.json`）。

```
python tests/record_fixture.py
```

为什么用"脚本化的假模型"录制，而不是连真实 API：

1. 固件要提交进仓库，内容必须可读、可 review、不含任何真实用户数据；
2. CI 里永远不会重新录制，所以这份固件就是**规格**——它明确写着"第 2 轮模型
   应当把指代替换成完整查询"。

真实录制（跑真实会话）不需要这个脚本，而是让 app 自己录音：

```powershell
$env:LLM_CASSETTE="my_session.json"
$env:LLM_CASSETTE_MODE="record"
.venv\\Scripts\\python.exe -m uvicorn app:app
# 正常对话几轮，每次模型调用都会写进 my_session.json
# 之后把 MODE 改成 replay（或删掉 MODE，默认就是 replay），同一会话可逐字节复现
```
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from langchain_core.messages import AIMessage  # noqa: E402

from cassette import CassetteModel, CassetteStore  # noqa: E402
from cassette_scenario import MODEL_NAME, run_scenario  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "cassettes" / "multiturn_rewrite.json"


class ScriptedModel:
    """按顺序吐出预设响应，冒充真实模型。

    顺序对应场景里的 4 次决策调用：
      第 1 轮：调检索 -> 收手
      第 2 轮：调检索（改写后） -> 收手
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self._responses.pop(0)

    async def astream(self, messages, **kwargs):  # 本场景不走流式
        raise NotImplementedError
        yield ""


def tool_call(name: str, question: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"question": question, "top_k": 4}, "id": call_id}],
    )


def main() -> None:
    responses = [
        tool_call("search_knowledge_base", "报销单提交时限是多久", "c1"),
        AIMessage(content="资料已足够"),
        tool_call("search_knowledge_base", "出差住宿报销标准是多少", "c2"),
        AIMessage(content="资料已足够"),
    ]
    FIXTURE.unlink(missing_ok=True)          # 重新录制前先清空，避免旧记录混入
    store = CassetteStore(FIXTURE, "record")
    model = CassetteModel(store, ScriptedModel(responses), model_name=MODEL_NAME)

    first, second = run_scenario(model)
    print("第 1 轮查询:", first.queries)
    print("第 2 轮查询:", second.queries)
    print("记录条数:", store.recorded_count, "->", FIXTURE)


if __name__ == "__main__":
    main()
