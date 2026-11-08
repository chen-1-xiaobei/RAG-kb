"""推理设备解析：把 `auto` / `cpu` / `cuda` 变成具体设备。

## 为什么要单独一个模块

三处地方要用同一个判断（精排、embedding、以后可能的其它模型），而它们的导入顺序
各不相同（有的在 threading 之前，有的在模型加载时）。把判断收敛到一处，可以保证：

1. **`auto` 的行为只有一份**：有 CUDA 用 CUDA，没有就 CPU。CI 是无 GPU 的
   GitHub runner，与笔记本共用同一份代码——它必须自动退化。
2. **显式要求 CUDA 却不可用时要立刻报错**，而不是静默跑回 CPU。否则
   "我明明设了 GPU"会变成一个查不到的谜题（性能看着不对，但没有任何报错）。

设备探测会 import torch（约 1 秒），因此结果缓存到模块级。
"""
from __future__ import annotations

import logging

logger = logging.getLogger("rag-kb")

_resolved: dict[str, str] = {}


def resolve(choice: str) -> str:
    """把配置值解析成实际设备字符串（"cpu" / "cuda"）。"""
    if choice == "cpu":
        return "cpu"

    if choice in _resolved:
        return _resolved[choice]

    try:
        import torch

        available = torch.cuda.is_available()
    except Exception:  # torch 未装/构建异常：当作没有 GPU
        available = False

    if choice == "cuda" and not available:
        raise RuntimeError(
            "配置要求 cuda，但 torch.cuda.is_available() 为 False："
            "请确认装的是 CUDA 版 torch（pip install torch --index-url "
            "https://download.pytorch.org/whl/cu126）且驱动正常"
        )

    device = "cuda" if available else "cpu"
    _resolved[choice] = device
    logger.info("device_resolved choice=%s -> %s", choice, device)
    return device


def describe(choice: str) -> dict:
    """给健康检查/日志用的设备描述（不抛异常）。"""
    try:
        device = resolve(choice)
    except RuntimeError as exc:
        return {"requested": choice, "device": "unknown", "error": str(exc)}
    info = {"requested": choice, "device": device}
    if device == "cuda":
        try:
            import torch

            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )
        except Exception:
            pass
    return info
