"""准入控制的离线测试：令牌桶、并发闸门、有界队列、拒绝语义。

全部用**假时钟**与**线程同步**验证，不依赖真实耗时、不需要压测——
这是"在弱机器上也能证明行为正确"的做法：验证的是"会不会按要求拒绝"，
而不是"这台机器多快"。
"""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import obs  # noqa: E402
from admission import (  # noqa: E402
    AdmissionController,
    AdmissionRejected,
    TokenBucket,
)


class FakeClock:
    """可手动推进的单调时钟。"""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _clean_metrics():
    obs.reset()
    yield
    obs.reset()


# ---------------------------------------------------------------------------
# 令牌桶
# ---------------------------------------------------------------------------

def test_bucket_disabled_when_rate_is_zero():
    """rate=0 表示不限流：这正是默认值，避免"看起来合理"的限流永远不触发。"""
    bucket = TokenBucket(rate=0, burst=4, clock=FakeClock())
    assert not bucket.enabled
    for _ in range(100):
        assert bucket.try_acquire() == (True, 0.0)


def test_bucket_allows_burst_then_blocks():
    clock = FakeClock()
    bucket = TokenBucket(rate=2, burst=2, clock=clock)

    assert bucket.try_acquire()[0] is True       # 第 1 个
    assert bucket.try_acquire()[0] is True       # 第 2 个（突发额度用完）
    allowed, wait = bucket.try_acquire()
    assert allowed is False
    assert wait == pytest.approx(0.5, abs=1e-6)  # 缺 1 个令牌 / 每秒 2 个


def test_bucket_refills_over_time():
    clock = FakeClock()
    bucket = TokenBucket(rate=2, burst=2, clock=clock)
    bucket.try_acquire()
    bucket.try_acquire()
    clock.advance(1.0)                            # 补 2 个令牌
    assert bucket.try_acquire()[0] is True
    assert bucket.try_acquire()[0] is True
    assert bucket.try_acquire()[0] is False


def test_bucket_never_exceeds_burst():
    clock = FakeClock()
    bucket = TokenBucket(rate=10, burst=2, clock=clock)
    clock.advance(100)                            # 放很久也只补到 burst
    assert bucket.tokens() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 并发闸门
# ---------------------------------------------------------------------------

def test_gate_limits_concurrent_execution():
    """同时在"执行中"的请求数不得超过 workers——这是闸门存在的唯一理由。"""
    controller = AdmissionController(workers=2, queue_max=8, queue_timeout=5.0)
    peak = 0
    current = 0
    lock = threading.Lock()

    def worker():
        nonlocal peak, current
        with controller.slot():
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.02)
            with lock:
                current -= 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert peak <= 2, f"并发执行数超过闸门：{peak}"
    assert peak >= 2, "闸门没有真正并行（可能是串行执行）"


def test_snapshot_reports_state():
    controller = AdmissionController(workers=1, queue_max=3, queue_timeout=1.0)
    snapshot = controller.snapshot()
    assert snapshot["workers"] == 1
    assert snapshot["queue_max"] == 3
    assert snapshot["queue_depth"] == 0
    assert snapshot["in_flight"] == 0


# ---------------------------------------------------------------------------
# 有界队列与拒绝
# ---------------------------------------------------------------------------

def test_queue_full_rejects_immediately():
    """队列满时必须**立刻**拒绝，而不是排进去一起等。"""
    controller = AdmissionController(workers=1, queue_max=0, queue_timeout=5.0)
    started = threading.Event()
    release = threading.Event()

    def holder():
        with controller.slot():
            started.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=holder)
    thread.start()
    started.wait(timeout=5)

    with pytest.raises(AdmissionRejected) as excinfo:
        with controller.slot():
            pass
    assert excinfo.value.reason == "queue_full"
    assert excinfo.value.status_code == 503

    release.set()
    thread.join(timeout=5)


def test_queue_timeout_rejects_when_waiting_too_long():
    controller = AdmissionController(workers=1, queue_max=2, queue_timeout=0.05)
    started = threading.Event()
    release = threading.Event()

    def holder():
        with controller.slot():
            started.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=holder)
    thread.start()
    started.wait(timeout=5)

    with pytest.raises(AdmissionRejected) as excinfo:
        with controller.slot():
            pass
    assert excinfo.value.reason == "queue_timeout"
    assert excinfo.value.status_code == 503

    release.set()
    thread.join(timeout=5)


def test_rate_limited_is_429_not_503():
    """速率超限是"你太快了"（429），容量不足才是"我处理不了"（503）。"""
    controller = AdmissionController(workers=1, queue_max=0, queue_timeout=1.0,
                                     rate=1, burst=1)
    with controller.slot():
        pass                                   # 用掉唯一令牌
    with pytest.raises(AdmissionRejected) as excinfo:
        with controller.slot():
            pass
    assert excinfo.value.reason == "rate_limited"
    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after >= 1


def test_rejections_are_counted_with_reason():
    controller = AdmissionController(workers=1, queue_max=0, queue_timeout=1.0)
    started = threading.Event()
    release = threading.Event()

    def holder():
        with controller.slot():
            started.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=holder)
    thread.start()
    started.wait(timeout=5)
    for _ in range(3):
        with pytest.raises(AdmissionRejected):
            with controller.slot():
                pass
    release.set()
    thread.join(timeout=5)

    rendered = obs.render_prometheus()
    assert 'admission_rejected_total{reason="queue_full"}' in rendered
    assert "admission_queue_depth" in rendered


def test_slot_releases_gate_on_exception():
    """执行中抛异常也必须归还闸门，否则一次失败会永久占掉一个执行位。"""
    controller = AdmissionController(workers=1, queue_max=0, queue_timeout=1.0)
    with pytest.raises(ValueError):
        with controller.slot():
            raise ValueError("boom")
    # 还能再拿到（说明已归还）
    with controller.slot():
        pass
    assert controller.snapshot()["in_flight"] == 0
