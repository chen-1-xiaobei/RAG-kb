"""MCP server 的离线单测：只验证注册与 schema，不真的检索（不加载模型）。

`mcp` 是可选依赖：没装就跳过，不影响主测试套件。
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("mcp", reason="mcp 未安装，跳过 MCP server 测试")

import mcp_server.server as server_module  # noqa: E402


def list_tools() -> list:
    return asyncio.run(server_module.server.list_tools())


def test_expected_tools_are_registered():
    names = {tool.name for tool in list_tools()}
    assert names == {"search_knowledge_base", "list_tables", "query_table"}


def test_write_tools_are_deliberately_not_exposed():
    """报销提交依赖用户确认，MCP 的 tool call 没有确认通道，因此不能暴露。"""
    names = {tool.name for tool in list_tools()}
    assert "commit_expense_claim" not in names
    assert "propose_expense_claim" not in names


def test_tool_schemas_expose_expected_parameters():
    by_name = {tool.name: tool for tool in list_tools()}
    search = by_name["search_knowledge_base"]
    # mcp 2.x 的字段名是 input_schema（v1 叫 inputSchema）
    schema = search.input_schema
    assert set(schema["properties"]) == {"question", "top_k"}
    assert schema["required"] == ["question"]

    query = by_name["query_table"]
    assert set(query.input_schema["properties"]) == {"table", "sql"}


def test_tool_descriptions_are_present():
    """docstring 就是给模型看的说明——为空等于工具没定义。"""
    for tool in list_tools():
        assert tool.description and len(tool.description) > 10, f"{tool.name} 缺少说明"


def test_principal_comes_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_KB_USER", "alice")
    monkeypatch.setenv("RAG_KB_GROUPS", "财务,人事")
    principal = server_module.current_principal()
    assert principal.user_id == "alice"
    # Principal 会自动并入 public（public 恒为可见组，与主应用一致）
    assert {"财务", "人事"} <= principal.allowed_groups
    assert "public" in principal.allowed_groups


def test_principal_defaults_to_public(monkeypatch):
    monkeypatch.delenv("RAG_KB_USER", raising=False)
    monkeypatch.delenv("RAG_KB_GROUPS", raising=False)
    principal = server_module.current_principal()
    assert principal.allowed_groups == frozenset({"public"})


def test_index_metadata_resource_is_json():
    text = server_module.index_metadata()
    # 索引不可用时返回提示语，可用时返回 JSON——两种都不该抛异常
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        assert "索引不可用" in text
    else:
        assert "record_count" in payload or "embedding_model" in payload


def test_prompt_contains_citation_requirement():
    text = server_module.grounded_answer("报销多久内提交")
    assert "[1]" in text and "不要推测" in text and "报销多久内提交" in text
