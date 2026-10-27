"""可观测性：请求上下文 + 极简指标。

两块内容，对应之前缺的两样东西。

**一、请求上下文（rid / conv）**

之前的问题：一次请求会打出十几行日志，但这些行之间没有任何共同标识。两个用户
同时提问时，日志是交错的一团，无法回答"这个用户那 9 秒到底花在哪"。

做法是 `contextvars`：中间件为每个请求分配 `rid`，端点拿到会话后再补上 `conv`，
日志过滤器把它们注进每条记录。`contextvars` 会随 anyio 的线程池一起复制，
所以同步端点（走 `run_in_threadpool`）里打的日志同样带得上。

**二、指标**

只做四件事：请求结果、各阶段耗时、工具调用、精排缓存。故意**不引入
prometheus_client**——四个指标不值得多一个依赖，而 Prometheus 文本格式本身
很简单（summary 用 quantile 行 + `_count` / `_sum`）。

注意它不是完整的指标系统：延迟分位数是从一个**固定长度滑动窗口**算出来的近似值，
不是 Prometheus 那种真直方图。对单实例、低频的服务够用，对高并发有偏。
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from contextvars import ContextVar
from typing import Deque, Iterable

# ---------------------------------------------------------------------------
# 请求上下文
# ---------------------------------------------------------------------------

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_conversation_id: ContextVar[str] = ContextVar("conversation_id", default="-")

UNKNOWN = "-"


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def set_request_id(rid: str) -> None:
    """由中间件调用。设成空串时保持 UNKNOWN，避免日志里出现 rid= 空值。"""
    _request_id.set(rid or UNKNOWN)


def bind_conversation(conversation_id: str | None) -> None:
    """会话 ID 只有解析完请求体才知道，所以在端点里补绑。"""
    _conversation_id.set(conversation_id or UNKNOWN)


def current_request_id() -> str:
    return _request_id.get()


def current_conversation_id() -> str:
    return _conversation_id.get()


def reset_context() -> None:
    """测试用：清空上下文。"""
    _request_id.set(UNKNOWN)
    _conversation_id.set(UNKNOWN)


class ContextFilter(logging.Filter):
    """把 rid / conv 注入日志记录，供 format 里的 %(rid)s / %(conv)s 使用。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = _request_id.get()
        record.conv = _conversation_id.get()
        return True


LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)s %(name)s rid=%(rid)s conv=%(conv)s: %(message)s"
)


def install_logging(level: int = logging.INFO) -> None:
    """配置根日志：时间精确到毫秒 + 请求关联键。

    把 filter 装在 handler 上而不是 root logger 上：logger 级别的 filter 只对
    该 logger 直接产生的记录生效，子 logger 传播上来的记录不会经过它。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

LabelKey = tuple[tuple[str, str], ...]

_lock = threading.Lock()
_counters: dict[tuple[str, LabelKey], float] = {}
_summaries: dict[tuple[str, LabelKey], Deque[float]] = {}
_summary_totals: dict[tuple[str, LabelKey], list[float]] = {}   # [count, sum]

WINDOW = 512          # 分位数只在最近这么多次观测上计算
QUANTILES = (0.5, 0.95, 0.99)


def _label_key(labels: dict[str, str]) -> LabelKey:
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def inc(name: str, amount: float = 1, **labels: str) -> None:
    key = (name, _label_key(labels))
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + amount


def observe(name: str, value: float, **labels: str) -> None:
    key = (name, _label_key(labels))
    with _lock:
        window = _summaries.setdefault(key, deque(maxlen=WINDOW))
        window.append(float(value))
        totals = _summary_totals.setdefault(key, [0.0, 0.0])
        totals[0] += 1
        totals[1] += float(value)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _render_labels(labels: LabelKey, extra: Iterable[tuple[str, str]] = ()) -> str:
    parts = [f'{key}="{value}"' for key, value in labels]
    parts.extend(f'{key}="{value}"' for key, value in extra)
    return "{" + ",".join(parts) + "}" if parts else ""


def render_prometheus() -> str:
    """按 Prometheus 文本格式输出。"""
    lines: list[str] = []
    with _lock:
        counters = dict(_counters)
        summaries = {key: list(window) for key, window in _summaries.items()}
        totals = {key: list(total) for key, total in _summary_totals.items()}

    emitted: set[str] = set()
    for (name, labels), value in sorted(counters.items()):
        if name not in emitted:
            lines.append(f"# TYPE {name} counter")
            emitted.add(name)
        lines.append(f"{name}{_render_labels(labels)} {value:g}")

    emitted.clear()
    for (name, labels), values in sorted(summaries.items()):
        if name not in emitted:
            lines.append(f"# TYPE {name} summary")
            emitted.add(name)
        ordered = sorted(values)
        for quantile in QUANTILES:
            # Prometheus 里分位数是 summary 的一个标签，不是单独的指标名
            with_quantile = labels + (("quantile", f"{quantile:g}"),)
            lines.append(f"{name}{_render_labels(with_quantile)} {_quantile(ordered, quantile):.4f}")
        count, total = totals.get((name, labels), [0.0, 0.0])
        lines.append(f"{name}_count{_render_labels(labels)} {count:g}")
        lines.append(f"{name}_sum{_render_labels(labels)} {total:.4f}")

    return "\n".join(lines) + "\n"


def reset() -> None:
    """测试用：清空所有指标。"""
    with _lock:
        _counters.clear()
        _summaries.clear()
        _summary_totals.clear()
