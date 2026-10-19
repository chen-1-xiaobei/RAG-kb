"""工具层单测（stub 检索函数 + 临时 CSV，不加载模型、不调用 LLM）。"""
import json
from pathlib import Path

from langchain_core.messages import AIMessage

from auth import Principal
from clarification import CLARIFICATION_KEY, CLARIFY_TOOL_NAME
from tools import (
    LIST_TOOL_NAME,
    QUERY_TOOL_NAME,
    SEARCH_TOOL_NAME,
    TableRegistry,
    build_tools,
    clamp_top_k,
    execute_tools,
    list_tables_impl,
    make_search_tool,
    merge_hits,
    merge_rows,
    query_table_impl,
    search_impl,
)

FINANCE_CSV = "项目,标准\n住宿-一线城市,500元/晚\n住宿-其他城市,350元/晚\n"


class FakeDoc:
    def __init__(self, doc_id, source="a.md", section="流程", text="正文"):
        self.metadata = {"id": doc_id, "source": source, "section": section}
        self.page_content = text


class StubTool:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def invoke(self, args):
        if self._error:
            raise self._error
        return self._result


def _principal(*groups):
    return Principal("u", frozenset(groups))


def _registry(tmp_path, files):
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    registry = TableRegistry(tmp_path)
    registry.load()
    return registry


# ---------- 检索工具 ----------

def test_search_impl_clamps_top_k():
    seen = {}
    search_impl("q", 100, _principal(), lambda q, k, p: seen.update(k=k) or [], 8, 500)
    assert seen["k"] == 8  # search_impl 自身的兜底夹取（第二道防线）
    search_impl("q", 0, _principal(), lambda q, k, p: seen.update(k=k) or [], 8, 500)
    assert seen["k"] == 1


# ---------- top_k 成本授权 ----------

def test_clamp_top_k_caps_at_default():
    """默认值同时是模型指定值的封顶：调大无效，调小生效。"""
    assert clamp_top_k(4, 4) == 4
    assert clamp_top_k(8, 4) == 4, "模型想放大检索规模应被拒绝"
    assert clamp_top_k(20, 4) == 4
    assert clamp_top_k(2, 4) == 2, "调小应生效（用户想更快）"
    assert clamp_top_k(0, 4) == 1, "下限仍是 1"


def test_clamp_top_k_falls_back_on_bad_input():
    """工具参数由模型生成，格式不对时回落默认值而不是抛异常打断循环。"""
    for bad in (None, "abc", "", [], {}):
        assert clamp_top_k(bad, 4) == 4


def test_search_tool_receives_clamped_top_k():
    seen = {}
    tool = make_search_tool(
        _principal(), lambda q, k, p: seen.update(k=k) or [], max_top_k=8, truncate=500,
        default_top_k=4,
    )
    tool.invoke({"question": "q", "top_k": 8})
    assert seen["k"] == 4, "模型填 8 也必须按服务端默认值 4 执行"
    tool.invoke({"question": "q", "top_k": 1})
    assert seen["k"] == 1


def test_search_impl_zero_hits():
    text, artifact = search_impl("q", 4, _principal(), lambda q, k, p: [], 8, 500)
    assert text == "未检索到相关文档。"
    assert artifact == []


def test_search_impl_text_truncated_and_has_no_citation_ids():
    docs = [(FakeDoc("d1", text="x" * 50), 0.9)]
    text, artifact = search_impl("q", 4, _principal(), lambda q, k, p: docs, 8, truncate=10)
    assert "citation" not in text.lower()      # 编号只在答案节点赋值，工具文本不带
    assert "x" * 10 in text and "x" * 11 not in text
    assert artifact == docs                    # 证据不受截断影响


# ---------- 表格注册表 ----------

def test_registry_scans_groups(tmp_path):
    registry = _registry(tmp_path, {"财务/报销标准.csv": FINANCE_CSV, "公开表.csv": "a,b\n1,2\n"})
    assert set(registry.tables) == {"财务/报销标准", "公开表"}
    assert registry.tables["财务/报销标准"].group == "财务"
    assert registry.tables["公开表"].group == "public"


def test_registry_tolerates_ragged_rows(tmp_path):
    registry = _registry(tmp_path, {"公开表.csv": "a,b\n1\n2,3,4\n"})
    text, docs = query_table_impl("公开表", 'SELECT * FROM "公开表"', _principal(), registry, 50)
    assert len(docs) == 2  # 不足补空、超出截断，不因参差行失败


# ---------- 表格工具 ----------

def test_query_table_markdown_and_document_metadata(tmp_path):
    registry = _registry(tmp_path, {"财务/报销标准.csv": FINANCE_CSV})
    text, docs = query_table_impl(
        "财务/报销标准", 'SELECT * FROM "财务/报销标准"', _principal("财务"), registry, 50
    )
    assert "| 项目 | 标准 |" in text and "命中 2 行" in text
    assert len(docs) == 2
    meta = docs[0].metadata
    assert meta["id"] == "table:财务/报销标准:1"
    assert meta["source"] == "报销标准.csv"
    assert meta["relative_path"] == "财务/报销标准.csv"
    assert meta["group"] == "财务"
    assert meta["section"] == "表格数据"
    assert meta["page"] is None and meta["para"] is None


def test_query_table_unauthorized_same_as_missing(tmp_path):
    registry = _registry(tmp_path, {"财务/报销标准.csv": FINANCE_CSV})
    public = _principal()
    # 无权与不存在返回同一文案：不泄露存在性
    assert query_table_impl("财务/报销标准", 'SELECT * FROM "财务/报销标准"', public, registry, 50) == ("表不存在。", [])
    assert query_table_impl("不存在", "SELECT 1", public, registry, 50) == ("表不存在。", [])


def test_query_table_rejects_writes_and_multi_statements(tmp_path):
    registry = _registry(tmp_path, {"财务/报销标准.csv": FINANCE_CSV})
    principal = _principal("财务")
    for sql in [
        'DROP TABLE "财务/报销标准"',
        'UPDATE "财务/报销标准" SET 标准=1',
        'SELECT * FROM "财务/报销标准"; DROP TABLE "财务/报销标准"',
        'SELECT * FROM "财务/报销标准" -- 注释',
    ]:
        text, docs = query_table_impl("财务/报销标准", sql, principal, registry, 50)
        assert text == "仅支持只读 SELECT 查询该表。" and docs == []


def test_query_table_from_must_match_table(tmp_path):
    registry = _registry(tmp_path, {
        "财务/报销标准.csv": FINANCE_CSV,
        "人事/员工名录.csv": "姓名\n张三\n",
    })
    principal = _principal("财务", "人事")  # 两个组都授权，仍不允许跨表读
    for sql in [
        'SELECT * FROM "人事/员工名录"',
        'SELECT * FROM "财务/报销标准" WHERE 1 IN (SELECT 1 FROM "人事/员工名录")',
    ]:
        text, docs = query_table_impl("财务/报销标准", sql, principal, registry, 50)
        assert text == "仅支持只读 SELECT 查询该表。" and docs == []


def test_query_table_row_limit_truncates(tmp_path):
    rows = "n\n" + "\n".join(str(i) for i in range(60)) + "\n"
    registry = _registry(tmp_path, {"公开表.csv": rows})
    text, docs = query_table_impl("公开表", 'SELECT * FROM "公开表"', _principal(), registry, 50)
    assert "已截断" in text
    assert len(docs) == 50


def test_query_table_empty_result(tmp_path):
    registry = _registry(tmp_path, {"财务/报销标准.csv": FINANCE_CSV})
    text, docs = query_table_impl(
        "财务/报销标准",
        'SELECT * FROM "财务/报销标准" WHERE 项目 = "不存在"',
        _principal("财务"), registry, 50,
    )
    assert text == "表中没有符合条件的行。" and docs == []


def test_list_tables_only_authorized_groups(tmp_path):
    registry = _registry(tmp_path, {
        "财务/报销标准.csv": FINANCE_CSV,
        "人事/员工名录.csv": "姓名\n张三\n",
    })
    text, artifact = list_tables_impl(_principal("财务"), registry)
    assert "财务/报销标准" in text
    assert "人事/员工名录" not in text  # 无权表不出现在清单里
    assert artifact == []


# ---------- 证据累积与工具分发 ----------

def test_merge_dedupes_by_id():
    a, b = FakeDoc("d1"), FakeDoc("d2")
    assert merge_hits([(a, 0.9)], [(a, 0.5), (b, 0.8)]) == [(a, 0.9), (b, 0.8)]
    assert merge_rows([a], [a, b]) == [a, b]


def test_execute_tools_aligns_tool_call_ids_and_accumulates():
    doc = FakeDoc("d1")
    state = {
        "messages": [AIMessage(content="", tool_calls=[
            {"name": SEARCH_TOOL_NAME, "args": {"question": "q"}, "id": "c1"},
            {"name": LIST_TOOL_NAME, "args": {}, "id": "c2"},
        ])],
        "tools": {
            SEARCH_TOOL_NAME: StubTool(("材料", [(doc, 0.7)])),
            LIST_TOOL_NAME: StubTool(("表清单", [])),
        },
        "hits": [], "table_rows": [], "tool_rounds": 0,
    }
    update = execute_tools(state)
    assert [m.tool_call_id for m in update["messages"]] == ["c1", "c2"]
    assert update["hits"] == [(doc, 0.7)]   # 检索 artifact 进 hits
    assert update["table_rows"] == []       # list_tables 不累积证据
    assert update["tool_rounds"] == 1


def test_execute_tools_unknown_and_failing_tools_become_errors():
    state = {
        "messages": [AIMessage(content="", tool_calls=[
            {"name": "nope", "args": {}, "id": "c1"},
            {"name": SEARCH_TOOL_NAME, "args": {}, "id": "c2"},
        ])],
        "tools": {SEARCH_TOOL_NAME: StubTool(error=RuntimeError("boom"))},
        "hits": [], "table_rows": [], "tool_rounds": 0,
    }
    update = execute_tools(state)
    assert all(m.status == "error" for m in update["messages"])
    assert update["hits"] == []  # 失败的工具不产生证据


# ---------- 单请求内检索去重 ----------

class CountingSearchTool:
    """记录被真正执行了几次——去重的意义就是让这个数字不涨。"""

    def __init__(self):
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        doc = FakeDoc(f"d{self.calls}")
        return f"材料{self.calls}", [(doc, 0.9)]


def test_execute_tools_dedupes_repeated_search():
    """同一 query 连调两次只应真正检索一次（agent 重复提问时白付一次精排）。"""
    tool = CountingSearchTool()
    calls = [
        {"name": SEARCH_TOOL_NAME, "args": {"question": "住宿标准", "top_k": 4}, "id": "c1"},
        {"name": SEARCH_TOOL_NAME, "args": {"question": "住宿标准", "top_k": 4}, "id": "c2"},
    ]
    state = {
        "messages": [AIMessage(content="", tool_calls=calls)],
        "tools": {SEARCH_TOOL_NAME: tool},
        "hits": [], "table_rows": [], "tool_rounds": 0,
    }
    update = execute_tools(state)
    assert tool.calls == 1
    assert [m.tool_call_id for m in update["messages"]] == ["c1", "c2"]
    assert update["messages"][0].content == update["messages"][1].content
    assert len(update["hits"]) == 1  # 证据按 id 去重，不会重复计入


def test_execute_tools_dedupes_after_clamping():
    """top_k=8 与 top_k=20 被夹成同一个实际查询，应复用而不是重跑。"""
    tool = CountingSearchTool()
    calls = [
        {"name": SEARCH_TOOL_NAME, "args": {"question": "q", "top_k": 8}, "id": "c1"},
        {"name": SEARCH_TOOL_NAME, "args": {"question": "q", "top_k": 20}, "id": "c2"},
    ]
    state = {
        "messages": [AIMessage(content="", tool_calls=calls)],
        "tools": {SEARCH_TOOL_NAME: tool},
        "hits": [], "table_rows": [], "tool_rounds": 0,
    }
    execute_tools(state)
    assert tool.calls == 1


def test_execute_tools_keeps_distinct_queries():
    """换关键词重试是合法且必要的，绝不能被去重掉。"""
    tool = CountingSearchTool()
    calls = [
        {"name": SEARCH_TOOL_NAME, "args": {"question": "住宿标准", "top_k": 4}, "id": "c1"},
        {"name": SEARCH_TOOL_NAME, "args": {"question": "差旅报销规定", "top_k": 4}, "id": "c2"},
    ]
    state = {
        "messages": [AIMessage(content="", tool_calls=calls)],
        "tools": {SEARCH_TOOL_NAME: tool},
        "hits": [], "table_rows": [], "tool_rounds": 0,
    }
    execute_tools(state)
    assert tool.calls == 2


def test_execute_tools_memo_persists_across_rounds():
    """跨轮也要去重：memo 通过 state 回传，下一轮的同一 query 同样命中。"""
    tool = CountingSearchTool()
    tools = {SEARCH_TOOL_NAME: tool}
    first = execute_tools({
        "messages": [AIMessage(content="", tool_calls=[
            {"name": SEARCH_TOOL_NAME, "args": {"question": "q", "top_k": 4}, "id": "c1"},
        ])],
        "tools": tools, "hits": [], "table_rows": [], "tool_rounds": 0,
    })
    execute_tools({
        "messages": [AIMessage(content="", tool_calls=[
            {"name": SEARCH_TOOL_NAME, "args": {"question": "q", "top_k": 4}, "id": "c2"},
        ])],
        "tools": tools,
        "hits": first["hits"], "table_rows": [], "tool_rounds": first["tool_rounds"],
        "search_memo": first["search_memo"],
    })
    assert tool.calls == 1, "第二轮同一 query 应命中上一轮写入的 memo"


def test_execute_tools_does_not_memo_table_tools(tmp_path):
    """表格工具不参与检索去重：每次查询都重新执行（避免误用陈旧行）。"""
    registry = _registry(tmp_path, {"公开表.csv": "a\n1\n"})
    tool = build_tools(_principal(), lambda q, k, p: [], registry)[QUERY_TOOL_NAME]
    calls = [
        {"name": QUERY_TOOL_NAME, "args": {"table": "公开表", "sql": 'SELECT * FROM "公开表"'}, "id": "c1"},
        {"name": QUERY_TOOL_NAME, "args": {"table": "公开表", "sql": 'SELECT * FROM "公开表"'}, "id": "c2"},
    ]
    state = {
        "messages": [AIMessage(content="", tool_calls=calls)],
        "tools": {QUERY_TOOL_NAME: tool},
        "hits": [], "table_rows": [], "tool_rounds": 0,
    }
    update = execute_tools(state)
    assert update["search_memo"] == {}, "只有检索工具进 memo"
    assert len(update["messages"]) == 2
    assert all(m.status == "success" for m in update["messages"])
    assert len(update["table_rows"]) == 1, "表格证据按 id 去重，两次相同查询不重复计入"


# ---------- 工厂 ----------

def test_build_tools_binds_principal(tmp_path):
    seen = {}

    def fake_search(question, k, principal):
        seen["principal"] = principal
        return []

    registry = _registry(tmp_path, {"财务/报销标准.csv": FINANCE_CSV})
    tools = build_tools(
        _principal("财务"), fake_search, registry,
        max_top_k=8, max_rows=50, truncate=500,
    )
    assert set(tools) == {
        SEARCH_TOOL_NAME, QUERY_TOOL_NAME, LIST_TOOL_NAME, CLARIFY_TOOL_NAME
    }
    text, artifact = tools[SEARCH_TOOL_NAME].invoke({"question": "住宿", "top_k": 2})
    assert text == "未检索到相关文档。"
    assert seen["principal"].allowed_groups == frozenset({"财务", "public"})  # 闭包持有 Principal
    _text, docs = tools[QUERY_TOOL_NAME].invoke(
        {"table": "财务/报销标准", "sql": 'SELECT * FROM "财务/报销标准"'}
    )
    assert len(docs) == 2


def test_clarify_tool_returns_payload_without_evidence():
    """反问工具不产生证据，只返回可解析的追问载荷。"""
    tools = build_tools(_principal(), lambda q, k, p: [], TableRegistry(Path("data")))
    payload = tools[CLARIFY_TOOL_NAME].invoke({"question": "你指的是哪个标准？"})
    assert json.loads(payload)[CLARIFICATION_KEY] == "你指的是哪个标准？"
