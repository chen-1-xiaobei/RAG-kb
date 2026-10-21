"""所有模块都必须能通过编译。

为什么需要这个文件：**没有任何单测 import `app.py`**——它会在 import 期构造
embedding 与 FastAPI 应用，成本太高，所以测试刻意绕开它。代价是 `app.py` 里的
**语法错误不会被任何测试发现**，CI 会静默放行。

这个教训是真实踩到的：一次编辑把 `generate_answer` 的 `def` 行重复了，`pytest`
依然是绿的，直到手工用 `ast.parse` 检查才发现。

`compile()` 只做语法与字节码生成，不执行模块顶层代码，因此不会加载模型。
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULES = sorted(ROOT.glob("*.py"))


def test_modules_are_discovered():
    """自检：至少要扫到主要模块，避免 glob 路径写错导致本文件形同虚设。"""
    names = {path.name for path in MODULES}
    assert {"app.py", "tools.py", "retrieval.py", "claims.py", "conversation.py"} <= names


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_module_compiles(path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
