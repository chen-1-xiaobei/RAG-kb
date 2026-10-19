"""Agent 工具层：检索工具 + 结构化表格查询工具（只读 sqlite）。

三个设计点：
- 工具身份闭包注入：每个工具由工厂函数构造并绑定 Principal，模型只能提供
  question/top_k/表名/SQL 等查询参数——权限不在模型可写的通道上。
- 模型可见材料与结构化证据解耦：工具返回 (text, artifact)。text 是截断后的
  材料文本（给模型看，不带 citation 编号，避免与答案节点的编号混淆）；
  artifact 是完整证据（给答案节点），由 execute_tools 按工具名累积进 state。
- 表格只读双保险：SELECT-only 语法校验 + sqlite set_authorizer 数据库层授权
  （只放行对白名单表的 READ）；未授权组查表与表不存在返回同一文案，
  不泄露存在性（与检索层预过滤哲学一致）。
"""
import csv
import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from auth import Principal
from clarification import CLARIFY_TOOL_NAME, clarify_payload
from config import settings
from ingest import group_for  # 复用权限约定：data/ 根目录=public，data/<文件夹>/=组名

logger = logging.getLogger("rag-kb")

ROOT = Path(__file__).parent
DATA = ROOT / "data"

SEARCH_TOOL_NAME = "search_knowledge_base"
QUERY_TOOL_NAME = "query_table"
LIST_TOOL_NAME = "list_tables"

_FROM_RE = re.compile(r"\bFROM\s+([`\"\w./\u4e00-\u9fff]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 检索工具：复用两段式检索管线（函数注入，避免 tools <-> app 循环导入）
# ---------------------------------------------------------------------------

def search_impl(
    question: str,
    top_k: int,
    principal: Principal,
    search_fn: Callable,
    max_top_k: int,
    truncate: int,
) -> tuple[str, list]:
    """检索核心：纯函数可单测。search_fn 生产环境传 app.retrieve_ranked。

    返回 (给模型看的材料文本, 完整 hits)。top_k 在服务端夹取——模型参数不可信。
    """
    k = max(1, min(int(top_k), max_top_k))
    hits = search_fn(question, k, principal)  # 内部已按 principal 预过滤
    if not hits:
        return "未检索到相关文档。", []
    parts = [f"命中 {len(hits)} 条资料："]
    for doc, _score in hits:
        source = doc.metadata.get("source", "")
        section = doc.metadata.get("section", "")
        parts.append(f"来源: {source}（章节: {section}）\n{doc.page_content[:truncate]}\n---")
    return "\n".join(parts), hits


def clamp_top_k(top_k: Any, default_top_k: int) -> int:
    """把模型给的 top_k 夹到服务端认可的范围。

    默认值同时是"模型指定值"的封顶：只有调小才能改，调大一律按默认值处理。
    精排耗时与候选数线性相关，检索成本不该由模型随手填的参数决定。
    非整数输入（字符串等）不报错、直接回落默认值——工具参数是模型生成的，
    因参数格式把整个循环打断不值得。
    """
    try:
        requested = int(top_k)
    except (TypeError, ValueError):
        requested = default_top_k
    return max(1, min(requested, default_top_k))


def make_search_tool(
    principal: Principal,
    search_fn: Callable,
    max_top_k: int = 8,
    truncate: int = 500,
    default_top_k: int = 4,
):
    @tool
    def search_knowledge_base(question: str, top_k: int = default_top_k) -> tuple:
        """在知识库文档中检索资料，用于企业制度、流程、手册类问题。

        参数 question 为检索查询（可重述用户问题以提高召回）。
        参数 top_k 为返回条数：**默认即可，调大不会有更多结果，调小可加快响应**。
        """
        return search_impl(
            question, clamp_top_k(top_k, default_top_k), principal, search_fn, max_top_k, truncate
        )

    return search_knowledge_base


# ---------------------------------------------------------------------------
# 表格注册表：data/**/*.csv -> sqlite :memory:（只读）
# ---------------------------------------------------------------------------

_NON_IDENT_RE = re.compile(r"[^\w]")  # \w 在 Python 3 已包含中日韩文字


class TableInfo:
    def __init__(
        self,
        key: str,
        columns: list[str],
        source: str,
        relative_path: str,
        group: str,
        row_count: int,
    ):
        self.key = key                  # 表名（模型照抄）：相对路径去扩展名，如 "财务/报销标准"
        self.columns = columns          # CSV 表头（已 sanitize）
        self.source = source            # 文件名（报销标准.csv）
        self.relative_path = relative_path
        self.group = group              # 权限组，与文档分组同一约定
        self.row_count = row_count


class TableRegistry:
    """启动时把 data/**/*.csv 载入一个 :memory: 连接；查询加锁串行（只读 SELECT 很快）。

    sqlite 表名直接使用相对路径去扩展名（quoted identifier），模型 SQL 里
    FROM "财务/报销标准" 即可命中白名单表，无需名字映射。
    CSV 全部按 TEXT 列存储；数值比较请用 CAST（工具 docstring 已提示模型）。
    """

    def __init__(self, data_dir: Path = DATA):
        self.data_dir = data_dir
        self._tables: dict[str, TableInfo] = {}
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def tables(self) -> dict[str, TableInfo]:
        return self._tables

    def load(self) -> None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)  # 跨线程共享，用锁串行
        tables: dict[str, TableInfo] = {}
        for path in sorted(self.data_dir.rglob("*.csv")):
            rel = path.relative_to(self.data_dir)
            group = group_for(rel)
            key = rel.with_suffix("").as_posix()  # "财务/报销标准"
            with path.open(encoding="utf-8-sig", newline="") as fh:
                raw = list(csv.reader(fh))
            if not raw:
                continue
            header = self._sanitize_columns(raw[0])
            width = len(header)
            # 参差行容错：不足补空、超出截断，保证 INSERT 列数一致
            rows = [
                [(row[i] if i < len(row) else "") for i in range(width)]
                for row in raw[1:]
            ]
            conn.execute(
                f'CREATE TABLE "{key}" ({", ".join(f'"{c}" TEXT' for c in header)})'
            )
            if rows:
                conn.executemany(
                    f'INSERT INTO "{key}" VALUES ({", ".join("?" * width)})', rows
                )
            tables[key] = TableInfo(key, header, path.name, rel.as_posix(), group, len(rows))
        self._tables, self._conn = tables, conn

    @staticmethod
    def _sanitize_columns(header: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        columns = []
        for col in header:
            cleaned = _NON_IDENT_RE.sub("_", col.strip()) or "col"
            seen[cleaned] = seen.get(cleaned, 0) + 1
            columns.append(cleaned if seen[cleaned] == 1 else f"{cleaned}_{seen[cleaned]}")
        return columns

    def lookup(self, key: str, principal: Principal) -> TableInfo | None:
        info = self._tables.get(key)
        if info is None or info.group not in principal.allowed_groups:
            return None  # 不存在与无权返回同一结果，不泄露存在性
        return info

    def list_for(self, principal: Principal) -> list[TableInfo]:
        return [t for t in self._tables.values() if t.group in principal.allowed_groups]


# ---------------------------------------------------------------------------
# SQL 安全：先语法白名单校验，再由 sqlite authorizer 在数据库层兜底
# ---------------------------------------------------------------------------

_FORBIDDEN_SQL = re.compile(
    r";|--|/\*|\*/"
    r"|\b(insert|update|delete|drop|alter|create|attach|detach|pragma"
    r"|replace|vacuum|grant|begin|commit|rollback)\b",
    re.IGNORECASE,
)


def _is_safe_select(sql: str, table_key: str) -> bool:
    """SELECT-only 校验：单语句、无注释、无写关键字，且 FROM 只引用白名单表。"""
    if not sql.upper().lstrip().startswith("SELECT"):
        return False
    if _FORBIDDEN_SQL.search(sql):
        return False
    froms = _FROM_RE.findall(sql)
    return not froms or all(ref.strip('`"') == table_key for ref in froms)


def _query_rows(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    sql: str,
    table_key: str,
    max_rows: int,
) -> tuple[list[str], list[tuple], bool]:
    """执行查询，返回 (列名, 行, 是否被截断)。

    set_authorizer 做数据库层兜底：即使语法校验被绕过，也只允许 SELECT/内置函数
    与对白名单表的 READ，其他动作一律 DENY。行数用 fetchmany 截断。
    """
    def authorizer(action, arg1, arg2, dbname, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1 == table_key:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    with lock:
        conn.set_authorizer(authorizer)
        try:
            cursor = conn.execute(sql)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchmany(max_rows + 1)
        finally:
            conn.set_authorizer(None)
    return columns, list(rows[:max_rows]), len(rows) > max_rows


# ---------------------------------------------------------------------------
# 表格工具：查询 + 列表
# ---------------------------------------------------------------------------

def row_to_document(info: TableInfo, columns: list[str], row: tuple, index: int) -> Document:
    """表格行 -> 伪 Document：metadata 对齐检索文档，Source 模型/citation 校验全部复用。"""
    cells = "；".join(f"{col}={value}" for col, value in zip(columns, row))
    return Document(
        page_content=f"《{info.source}》表格第 {index} 行：{cells}",
        metadata={
            "id": f"table:{info.key}:{index}",
            "source": info.source,
            "relative_path": info.relative_path,
            "folder": info.relative_path.rpartition("/")[0],
            "group": info.group,
            "title": info.key,
            "section": "表格数据",
            "page": None,
            "para": None,
        },
    )


def _markdown_table(columns: list[str], rows: list[tuple]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return lines


def query_table_impl(
    table: str,
    sql: str,
    principal: Principal,
    registry: TableRegistry,
    max_rows: int,
) -> tuple[str, list[Document]]:
    info = registry.lookup(table, principal)
    if info is None:
        return "表不存在。", []  # 与"无权访问"同文案：不泄露存在性
    if not _is_safe_select(sql, info.key):
        return "仅支持只读 SELECT 查询该表。", []
    try:
        columns, rows, truncated = _query_rows(
            registry._conn, registry._lock, sql, info.key, max_rows
        )
    except sqlite3.Error as exc:
        return f"SQL 执行失败：{exc}", []
    if not rows:
        return "表中没有符合条件的行。", []
    lines = [f"表 {info.key} 命中 {len(rows)} 行：", ""]
    lines.extend(_markdown_table(columns, rows))
    if truncated:
        lines.append(f"\n（结果超过 {max_rows} 行，已截断）")
    documents = [row_to_document(info, columns, row, i + 1) for i, row in enumerate(rows)]
    return "\n".join(lines), documents


def make_query_tool(principal: Principal, registry: TableRegistry, max_rows: int = 50):
    @tool
    def query_table(table: str, sql: str) -> tuple:
        """查询结构化表格数据（只读 SELECT）。参数 table 为表名（用 list_tables 查看可用表名），
        sql 的 FROM 后跟该表名。表格列均为文本，数值比较请用 CAST(x AS REAL)。
        例如: query_table(table="财务/报销标准",
        sql='SELECT * FROM "财务/报销标准" WHERE 项目 LIKE "%住宿%"')"""
        return query_table_impl(table, sql, principal, registry, max_rows)

    return query_table


def list_tables_impl(principal: Principal, registry: TableRegistry) -> tuple[str, list]:
    infos = registry.list_for(principal)
    if not infos:
        return "没有可访问的表格。", []
    lines = ["你有权访问以下表格："]
    for info in infos:
        lines.append(f"- {info.key}（{info.row_count} 行；列: {', '.join(info.columns)}）")
    return "\n".join(lines), []


def make_list_tool(principal: Principal, registry: TableRegistry):
    @tool
    def list_tables() -> tuple:
        """列出你有权访问的表格及列名。查询表格数据前先调用本工具了解可用表。"""
        return list_tables_impl(principal, registry)

    return list_tables


def make_clarify_tool():
    @tool
    def ask_clarification(question: str) -> str:
        """当用户问题缺少必要指代、无法确定检索目标时，向用户追问一句。

        只在无法构造出有效检索查询时使用；只要能检索就不要用它。
        """
        return clarify_payload(question)

    return ask_clarification


def build_tools(
    principal: Principal,
    search_fn: Callable,
    registry: TableRegistry,
    *,
    max_top_k: int = 8,
    max_rows: int = 50,
    truncate: int = 500,
    default_top_k: int = 4,
) -> dict[str, Any]:
    """per-request 构造工具闭包表。search_fn 生产环境传 app.retrieve_ranked。"""
    return {
        SEARCH_TOOL_NAME: make_search_tool(
            principal, search_fn, max_top_k, truncate, default_top_k
        ),
        QUERY_TOOL_NAME: make_query_tool(principal, registry, max_rows),
        LIST_TOOL_NAME: make_list_tool(principal, registry),
        CLARIFY_TOOL_NAME: make_clarify_tool(),
    }


# ---------------------------------------------------------------------------
# 证据累积与工具分发节点
# ---------------------------------------------------------------------------

def merge_hits(old: list, new: list) -> list:
    """跨轮证据累积：按文档 id 去重（首轮优先，保持精排分数序）。"""
    seen = {doc.metadata["id"] for doc, _ in old}
    merged = list(old)
    for doc, score in new:
        if doc.metadata["id"] not in seen:
            seen.add(doc.metadata["id"])
            merged.append((doc, score))
    return merged


def merge_rows(old: list, new: list) -> list:
    seen = {doc.metadata["id"] for doc in old}
    return old + [doc for doc in new if doc.metadata["id"] not in seen]


def memo_key_for(name: str, args: dict) -> str | None:
    """单请求内检索去重的键；非检索工具返回 None（不参与去重）。

    键基于**夹取后**的参数：模型写 top_k=8 或 top_k=20 都会被服务端夹成同一个
    实际查询，按原始参数建键会漏掉本可复用的那次检索。排序键保证参数顺序无关。
    """
    if name != SEARCH_TOOL_NAME:
        return None
    normalized = dict(args)
    normalized["top_k"] = clamp_top_k(args.get("top_k"), settings.tool_default_top_k)
    try:
        return f"{name}\x1f{json.dumps(normalized, ensure_ascii=False, sort_keys=True)}"
    except TypeError:  # 无法序列化的参数：放弃去重，不影响正确性
        return None


def execute_tools(state: dict) -> dict:
    """工具节点：执行最后一条 AIMessage 的全部 tool_calls。

    关键约定：
    - ToolMessage.tool_call_id 必须与 AIMessage.tool_calls[].id 一一对应，
      模型下一轮才能把工具结果与调用对上号。
    - 工具返回 (text, artifact)：text 进 ToolMessage（模型可见的截断材料，
      不含 citation 编号）；artifact 按工具名累积进 state 证据字段（不被截断）。
    - 未知工具/工具异常 -> error ToolMessage：把错误交给模型决定下一步，
      循环不中断、证据不累积。
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    round_no = state.get("tool_rounds", 0) + 1
    outputs: list[ToolMessage] = []
    hits = list(state.get("hits") or [])
    rows = list(state.get("table_rows") or [])
    # 单请求内的检索去重：同一 query 重复调用直接复用结果。
    # agent 得到空结果后常会"换个说法再试"，但也可能重复同一个 query——
    # 那就是白付一次精排（本机约 5s）。memo 只在单次请求的 state 里，
    # 不跨用户共享，因此不引入权限泄露面。
    memo: dict[str, tuple] = dict(state.get("search_memo") or {})
    clarified = False
    for call in calls:
        logger.info(
            "tool_call round=%d name=%s args=%s",
            round_no, call["name"], json.dumps(call["args"], ensure_ascii=False)[:300],
        )
        tool_fn = (state.get("tools") or {}).get(call["name"])
        if tool_fn is None:
            logger.warning("tool_unknown name=%s", call["name"])
            outputs.append(ToolMessage(
                content=f"未知工具: {call['name']}",
                tool_call_id=call["id"], name=call["name"], status="error",
            ))
            continue
        # 去重键必须基于"夹取后"的参数：top_k=8 与 top_k=20 会被服务端夹成同一个
        # 实际查询，若按原始参数建键就会白跑一次。
        memo_key = memo_key_for(call["name"], call["args"])
        if memo_key is not None and memo_key in memo:
            text, artifact = memo[memo_key]
            logger.info("tool_deduped name=%s saved_ms=rerank", call["name"])
            outputs.append(ToolMessage(content=text, tool_call_id=call["id"], name=call["name"]))
            hits = merge_hits(hits, artifact)
            continue
        started = perf_counter()
        try:
            result = tool_fn.invoke(call["args"])
            if call["name"] == CLARIFY_TOOL_NAME:
                # 反问工具只返回一段文本，不产生证据
                text, artifact = result, []
                clarified = True
            else:
                text, artifact = result
        except Exception as exc:  # 参数校验失败等：错误文本给模型看，不阻断循环
            logger.warning(
                "tool_failed name=%s ms=%d error=%s",
                call["name"], (perf_counter() - started) * 1000, exc,
            )
            outputs.append(ToolMessage(
                content=f"工具执行失败: {exc}",
                tool_call_id=call["id"], name=call["name"], status="error",
            ))
            continue
        logger.info(
            "tool_result name=%s chars=%d evidence=%d ms=%d",
            call["name"], len(text), len(artifact), (perf_counter() - started) * 1000,
        )
        outputs.append(ToolMessage(content=text, tool_call_id=call["id"], name=call["name"]))
        if call["name"] == SEARCH_TOOL_NAME:
            hits = merge_hits(hits, artifact)
            if memo_key is not None:
                memo[memo_key] = (text, artifact)
        elif call["name"] == QUERY_TOOL_NAME:
            rows = merge_rows(rows, artifact)
        # list_tables 的 artifact 为空，不累积证据
    return {
        "messages": outputs,
        "hits": hits,
        "table_rows": rows,
        "search_memo": memo,
        "clarification_requested": clarified or state.get("clarification_requested", False),
        "tool_rounds": state.get("tool_rounds", 0) + 1,
    }
