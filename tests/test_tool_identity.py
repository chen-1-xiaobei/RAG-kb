"""身份绑定方式的回归测试。

守的是项目最核心的那条结构性保证：**身份在闭包里，不在工具参数里**。

为什么值得单独一个文件：这条性质此前只在注释和 README 里写着，没有任何测试直接断言。
以后有人加工具时手滑写了 `user_id` 参数，或者把 `principal` 改成从参数取，
测试不会红——那就等于这条保证悄悄失效了。
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auth import Principal  # noqa: E402
from tools import (  # noqa: E402
    CLARIFY_TOOL_NAME,
    COMMIT_CLAIM_TOOL_NAME,
    LIST_TOOL_NAME,
    PROPOSE_CLAIM_TOOL_NAME,
    QUERY_TOOL_NAME,
    SEARCH_TOOL_NAME,
    build_tools,
)

ALICE = Principal("alice", frozenset({"财务"}))
BOB = Principal("bob", frozenset({"人事"}))

# 任何"模型的参数里不应出现"的身份相关键
IDENTITY_KEYS = {
    "user_id", "userid", "user", "principal", "groups", "allowed_groups",
    "session_id", "conversation_id", "tenant", "tenant_id", "role", "roles",
}

# 构造期不会真正用到 registry / claim_store，传 None 即可
def build(principal: Principal, conversation_id: str = "conv-test") -> dict:
    return build_tools(principal, lambda *a: ([], []), None, None,
                       conversation_id=conversation_id)


def free_vars(tool) -> tuple[str, ...]:
    """工具包装器里原函数的自由变量名（闭包捕获了哪些名字）。"""
    function = getattr(tool, "func", None)
    return function.__code__.co_freevars if function else ()


def closure_values(tool) -> dict:
    function = tool.func
    return {
        name: cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__)
    }


# ---------------------------------------------------------------------------
# 1. 模型的参数面上不能出现身份
# ---------------------------------------------------------------------------

def test_no_tool_exposes_identity_parameter():
    """任何工具的 schema 里都不允许出现身份相关参数。

    这是"权限不在模型可写的通道上"的直接断言：模型的入参里没有身份这个位置，
    它就无法构造"以别人身份调用"的请求。
    """
    for name, tool in build(ALICE).items():
        leaked = IDENTITY_KEYS & set(tool.args)
        assert not leaked, f"工具 {name} 暴露了身份参数 {sorted(leaked)}"


def test_tool_signatures_have_no_identity_parameter():
    """schema 之外的二次确认：原函数的签名里也没有。"""
    for name, tool in build(ALICE).items():
        function = getattr(tool, "func", None)
        if function is None:
            continue
        leaked = IDENTITY_KEYS & set(inspect.signature(function).parameters)
        assert not leaked, f"{name} 的签名里有身份参数 {sorted(leaked)}"


# ---------------------------------------------------------------------------
# 2. 身份必须真的被闭包捕获
# ---------------------------------------------------------------------------

def test_principal_is_captured_by_closure():
    """需要身份的工具，其 `principal` 必须是自由变量（闭包捕获）。"""
    expected = {
        SEARCH_TOOL_NAME, QUERY_TOOL_NAME, LIST_TOOL_NAME,
        PROPOSE_CLAIM_TOOL_NAME, COMMIT_CLAIM_TOOL_NAME,
    }
    captured = {name for name, tool in build(ALICE).items() if "principal" in free_vars(tool)}
    assert captured == expected, f"闭包捕获身份的工具不一致：{captured ^ expected}"


def test_clarify_tool_does_not_need_identity():
    """反问工具不查数据、不动数据，因此不该持有身份——持有越多，泄露面越大。"""
    assert "principal" not in free_vars(build(ALICE)[CLARIFY_TOOL_NAME])


def test_closure_holds_the_right_principal():
    """闭包里存的必须是本次请求的那个 Principal，而不是某个全局默认值。"""
    values = closure_values(build(ALICE)[SEARCH_TOOL_NAME])
    assert values["principal"] is ALICE
    assert values["principal"].user_id == "alice"


def test_two_principals_get_separate_tool_objects():
    """同一段函数体、两份闭包：alice 与 bob 的工具不是同一个对象。"""
    alice_tool = build(ALICE)[SEARCH_TOOL_NAME]
    bob_tool = build(BOB)[SEARCH_TOOL_NAME]
    assert alice_tool is not bob_tool
    assert closure_values(alice_tool)["principal"] is not closure_values(bob_tool)["principal"]


# ---------------------------------------------------------------------------
# 3. 写操作还要绑定会话
# ---------------------------------------------------------------------------

def test_write_tools_capture_conversation_id():
    """提案与提交必须发生在某个会话里——没有会话就没有"确认"这个上下文。"""
    for name in (PROPOSE_CLAIM_TOOL_NAME, COMMIT_CLAIM_TOOL_NAME):
        tool = build(ALICE, conversation_id="conv-42")[name]
        assert "conversation_id" in free_vars(tool), f"{name} 没有绑定会话"
        assert closure_values(tool)["conversation_id"] == "conv-42"


def test_conversation_id_is_not_a_model_parameter():
    """会话同理：模型不能自己指定往哪个会话里写。"""
    for name in (PROPOSE_CLAIM_TOOL_NAME, COMMIT_CLAIM_TOOL_NAME):
        assert "conversation_id" not in set(build(ALICE)[name].args)
