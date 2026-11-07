"""准入控制：令牌桶限流 + 有界排队 + 并发闸门。

## 为什么需要它

实测（1 进程 × 8 线程）：冷路径单请求 p50 约 9.5 秒，吞吐天花板约 0.38 req/s。
在这台机器上，**并发不是产能，是排队**——并发 16 时如果全放进来，p95 会到 60 秒以上，
连 `/health` 都可能被拖死。

所以这里的目标不是"提高吞吐"，而是**过载时行为可预测**：

    ┌─ 令牌桶（可选，默认关闭）── 超过速率 → 429
    ├─ 队列位（有界）───────────── 队列满   → 503 + Retry-After
    ├─ 排队超时 ────────────────── 等太久   → 503 + Retry-After
    └─ 并发闸门（信号量）───────── 真正执行

## 三个刻意的设计决定

**一、闸门只包住"贵的那一段"（精排），不包住整个请求。**
零候选的拒答路径耗时约 20ms，如果它也排队，就会被一个 10 秒的精排卡住——
那会把快路径的 25 QPS 直接赔掉。所以 `retrieve_ranked` 在候选为空时**不取闸门**。

**二、令牌桶默认关闭（rate=0），闸门默认开启。**
限流速率必须从目标机器的实测容量倒推：本机 0.38 req/s，服务器可能是 3 req/s。
写死一个"看起来合理"的数（比如 100/s）等于没做限流——它永远不会触发。

**三、拒绝是显式的，带原因码与 Retry-After。**
"排队 60 秒后超时"和"立刻告诉我稍后重试"是完全不同的两种用户体验，
而且后者不会把上游的连接池占满。

## 时钟可注入

TokenBucket 接受 `clock` 参数，测试用假时钟断言"恰好放行 N 次"，
不需要 sleep——这让限流逻辑可以离线、确定性地验证。
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import obs


class AdmissionRejected(RuntimeError):
    """请求被准入控制拒绝。端点是唯一把它翻译成 HTTP 状态码的地方。"""

    def __init__(self, reason: str, retry_after: int = 5, detail: str = ""):
        self.reason = reason
        self.retry_after = max(1, int(retry_after))
        self.detail = detail
        super().__init__(f"admission rejected: {reason}")

    @property
    def status_code(self) -> int:
        # 速率超限是"你太快了"（429），容量不足是"现在处理不了"（503）
        return 429 if self.reason == "rate_limited" else 503


class TokenBucket:
    """经典令牌桶。线程安全，时钟可注入。

    容量 = burst（允许的瞬时突发），每秒补充 rate 个令牌。
    rate <= 0 表示不限流（直接放行），用于"限流关闭"的默认状态。
    """

    def __init__(self, rate: float, burst: int, clock: Callable[[], float] = time.monotonic):
        if burst < 1:
            raise ValueError("burst 必须 >= 1")
        self.rate = float(rate)
        self.burst = float(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._updated = clock()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        if self.enabled:
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)

    def try_acquire(self, amount: float = 1.0) -> tuple[bool, float]:
        """尝试取令牌。返回 (是否成功, 建议等待秒数)。"""
        if not self.enabled:
            return True, 0.0
        with self._lock:
            self._refill()
            if self._tokens >= amount:
                self._tokens -= amount
                return True, 0.0
            missing = amount - self._tokens
            return False, missing / self.rate if self.rate else 0.0

    def tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class AdmissionController:
    """限流 + 有界排队 + 并发闸门。"""

    def __init__(
        self,
        *,
        workers: int = 1,
        queue_max: int = 4,
        queue_timeout: float = 30.0,
        rate: float = 0.0,
        burst: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ):
        if workers < 1:
            raise ValueError("workers 必须 >= 1")
        if queue_max < 0:
            raise ValueError("queue_max 不能为负")
        self.workers = workers
        self.queue_max = queue_max
        self.queue_timeout = float(queue_timeout)
        self.bucket = TokenBucket(rate, burst, clock)
        self._gate = threading.BoundedSemaphore(workers)
        self._lock = threading.Lock()
        self._waiting = 0          # 已进入队列、还没开始执行
        self._in_flight = 0        # 已拿到闸门、正在执行
        self._clock = clock
        self._publish()

    # ---------- 指标 ----------

    def _publish(self) -> None:
        obs.set_gauge("admission_queue_depth", self._waiting)
        obs.set_gauge("admission_in_flight", self._in_flight)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "workers": self.workers,
                "queue_max": self.queue_max,
                "queue_depth": self._waiting,
                "in_flight": self._in_flight,
                "tokens": round(self.bucket.tokens(), 3),
                "rate": self.bucket.rate,
                "queue_timeout": self.queue_timeout,
            }

    # ---------- 主流程 ----------

    def _enter(self) -> None:
        with self._lock:
            self._in_flight += 1
            self._publish_locked()

    def _leave(self) -> None:
        self._gate.release()
        with self._lock:
            self._in_flight -= 1
            self._publish_locked()

    @contextmanager
    def slot(self) -> Iterator[None]:
        """申请一个执行位。可能抛 AdmissionRejected。

        顺序：限流 → **先试着立刻拿到闸门** → 拿不到才进队列（受 queue_max 限制）。
        不能"先看队列是否已满再拿闸门"：那样 queue_max=0 时 `0 >= 0` 恒真，
        连空闲机器上的请求都会被拒——队列上限说的是"**排队**人数"，不是"并发数"。
        """
        allowed, wait_hint = self.bucket.try_acquire()
        if not allowed:
            obs.inc("admission_rejected_total", reason="rate_limited")
            raise AdmissionRejected("rate_limited", retry_after=max(1, round(wait_hint)))

        # ① 有空位：直接执行，不占队列额度
        if self._gate.acquire(blocking=False):
            self._enter()
            try:
                yield
            finally:
                self._leave()
            return

        # ② 需要排队：先占一个队列位，再等闸门
        with self._lock:
            queue_full = self._waiting >= self.queue_max
            if not queue_full:
                self._waiting += 1
            self._publish_locked()
        if queue_full:
            obs.inc("admission_rejected_total", reason="queue_full")
            raise AdmissionRejected("queue_full", retry_after=max(1, int(self.queue_timeout)))

        started = self._clock()
        acquired = self._gate.acquire(timeout=self.queue_timeout)
        waited = max(0.0, self._clock() - started)
        with self._lock:
            self._waiting -= 1
            self._publish_locked()

        if not acquired:
            obs.inc("admission_rejected_total", reason="queue_timeout")
            obs.observe("admission_wait_seconds", waited)
            raise AdmissionRejected(
                "queue_timeout", retry_after=max(1, int(self.queue_timeout / 2))
            )

        obs.observe("admission_wait_seconds", waited)
        self._enter()
        try:
            yield
        finally:
            self._leave()

    def _publish_locked(self) -> None:
        """在持锁状态下更新 gauge（obs 自己也加锁，这里只是保持快照一致）。"""
        obs.set_gauge("admission_queue_depth", self._waiting)
        obs.set_gauge("admission_in_flight", self._in_flight)
