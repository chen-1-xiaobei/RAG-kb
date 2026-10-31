"""路由表断言：用 AST 解析 app.py，**不导入应用**（导入会拉起模型栈）。

这条测试是被一次真实事故逼出来的：

插入新端点时误改了 `@app.post("/ask")` 那两行，`ask()` 的函数体因此变成
`conversation_trace()` 里 `return` 之后的不可达代码。结果——
- `ast.parse` 通过（文件语法合法）；
- 255 个单测全绿（**没有一个测试 import app**）；
- CI 的导入冒烟也通过（它只检查 import 成功与路由数量不为零）；
- 直到并发压测时 `/ask` 返回 404 才暴露。

所以这里用静态解析把期望的端点集合钉死：不加载模型、毫秒级、能进最快的那个 CI 任务。
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

HTTP_METHODS = {"get", "post", "delete", "put", "patch"}

# 对外契约：路径与方法的组合就是 API 面，删掉或改名都必须是有意识的改动
EXPECTED_ROUTES = {
    ("GET", "/"),
    ("GET", "/metrics"),
    ("GET", "/health"),
    ("DELETE", "/conversations/{conversation_id}"),
    ("GET", "/conversations/{conversation_id}/trace"),
    ("POST", "/ask"),
    ("POST", "/agent/ask"),
    ("POST", "/agent/ask/stream"),
}


def collect_routes() -> dict[tuple[str, str], ast.AST]:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    routes: dict[tuple[str, str], ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
                continue
            owner = getattr(decorator.func.value, "id", "")
            method = decorator.func.attr
            if owner != "app" or method not in HTTP_METHODS:
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                routes[(method.upper(), decorator.args[0].value)] = node
    return routes


def test_all_expected_routes_are_registered():
    routes = collect_routes()
    missing = EXPECTED_ROUTES - set(routes)
    assert not missing, f"端点缺失（被误删或改名？）: {sorted(missing)}"


def test_no_route_was_added_silently():
    """新增端点时必须同步更新这份清单——它是对外契约，不该悄悄变化。"""
    extra = set(collect_routes()) - EXPECTED_ROUTES
    assert not extra, f"出现未登记的新端点，请更新 EXPECTED_ROUTES: {sorted(extra)}"


def test_route_handlers_are_not_empty():
    """处理器必须有实际语句（排除"只剩 docstring"的残缺函数）。"""
    for (method, path), node in collect_routes().items():
        body = [stmt for stmt in node.body if not isinstance(stmt, ast.Expr)
                or not isinstance(getattr(stmt, "value", None), ast.Constant)]
        assert body, f"{method} {path} 的处理器只有 docstring"


def test_handlers_do_not_return_inside_another_handler():
    """崩溃现场的特征：函数体被挪进了上一个函数、位于 return 之后。

    这条检查不可达语句：如果某个函数在 return 之后还有语句，几乎一定是误编辑。
    """
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        returned = False
        for stmt in node.body:
            if returned:
                offenders.append(f"{node.name}:{stmt.lineno}")
                break
            if isinstance(stmt, ast.Return):
                returned = True
    assert not offenders, f"return 之后仍有不可达语句，疑似误编辑: {offenders}"
