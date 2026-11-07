"""Phase 1 验收：过载时是"快速失败"还是"一起超时"。

配置故意收紧（workers=1, queue=2, timeout=5s），然后一次打 8 个并发请求：
预期 1 个执行、2 个排队（可能超时）、5 个被立刻拒绝并带 Retry-After。
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from envfix import sanitize_proxy_env

sanitize_proxy_env()

import httpx

PORT = 8041
HEADERS = {"X-Dev-User": "loadtest", "X-Dev-Groups": "%E8%B4%A2%E5%8A%A1"}


def wait_health(timeout=420) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


async def fire(questions: list[str]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        await client.post(f"http://127.0.0.1:{PORT}/ask",
                          json={"question": questions[0], "top_k": 4},
                          headers=HEADERS, timeout=120)   # 预热

        async def one(question: str) -> dict:
            started = time.perf_counter()
            try:
                response = await client.post(
                    f"http://127.0.0.1:{PORT}/ask",
                    json={"question": question, "top_k": 4},
                    headers=HEADERS, timeout=120,
                )
                return {
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - started) * 1000),
                    "retry_after": response.headers.get("retry-after"),
                    "reason": (response.json().get("detail") or {}).get("reason")
                    if response.status_code != 200 else None,
                }
            except Exception as exc:
                return {"status": "timeout", "ms": round((time.perf_counter() - started) * 1000),
                        "error": type(exc).__name__}

        started = time.perf_counter()
        results = await asyncio.gather(*[one(q) for q in questions[1:9]])
        wall = round((time.perf_counter() - started) * 1000)
        return results, wall


def main() -> int:
    payload = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))
    questions = [
        case["question"] for case in payload["cases"]
        if case["kind"] == "retrieval" and case.get("required_sources")
        and set(case.get("groups") or []) <= {"财务", "人事"}
    ]

    env = dict(os.environ)
    env.update({
        "PYTHONUTF8": "1",
        "ADMISSION_WORKERS": "1",
        "ADMISSION_QUEUE_MAX": "2",
        "ADMISSION_QUEUE_TIMEOUT": "5",
        "VECTOR_BACKEND": "json",
    })
    env.pop("NO_PROXY", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
         "--port", str(PORT), "--log-level", "warning"],
        cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_health():
            print("服务未就绪")
            return 1
        results, wall = asyncio.run(fire(questions))

        print("\n=== 8 个并发请求的结果（workers=1, queue=2, timeout=5s）")
        for index, row in enumerate(results, 1):
            detail = f"reason={row.get('reason')}" if row.get("reason") else ""
            retry = f"Retry-After={row['retry_after']}" if row.get("retry_after") else ""
            print(f"  #{index} status={row['status']:<7} {row['ms']:>6}ms  {retry} {detail}")
        print(f"\n墙钟总时长: {wall} ms")

        codes = [row["status"] for row in results]
        ok = codes.count(200)
        rejected = sum(1 for code in codes if code in (429, 503))
        print(f"\n成功 {ok} 个 / 被拒 {rejected} 个 / 超时或异常 {len(codes) - ok - rejected} 个")

        metrics = httpx.get(f"http://127.0.0.1:{PORT}/metrics", timeout=20).text
        print("\n=== 相关指标")
        for line in metrics.splitlines():
            if line.startswith("admission") and not line.startswith("#"):
                print("  " + line)
        return 0
    finally:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    raise SystemExit(main())
