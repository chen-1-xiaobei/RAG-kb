"""从模型响应里提取 token 用量，并换算成本记账。

## 为什么单独一个模块

用量信息在 LangChain 里有两个来源，而且**不同 provider / 不同调用方式给的不一样**：

- `message.usage_metadata`：LangChain 标准字段（`input_tokens` / `output_tokens`）
- `message.response_metadata["token_usage"]`：OpenAI 风格（`prompt_tokens` / `completion_tokens`）

把它们收敛到一处，调用点只写 `record_usage(message)`，不用关心是哪一种。
拿不到用量时**显式记一个缺失计数**，而不是静默跳过——"成本统计一直是 0"
和"这个 provider 没回用量"是两件事，混起来会让人误以为成本为零。

## 价格必须配置，否则不记成本

默认价格为 0 时**不输出成本指标**：输出一个恒为 0 的成本比不输出更危险，
它会让人以为"这个系统不花钱"。价格写在环境变量里，因为不同模型/不同时间的
单价不同，硬编码在代码里一定会过时。
"""
from __future__ import annotations

import logging
from typing import Any

import obs
from config import settings

logger = logging.getLogger("rag-kb")


def extract_usage(message: Any) -> dict[str, int] | None:
    """返回 {"prompt": n, "completion": m}；拿不到返回 None。"""
    if message is None:
        return None

    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt = usage_metadata.get("input_tokens")
        completion = usage_metadata.get("output_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            return {"prompt": prompt, "completion": completion}

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, dict):
            prompt = token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
            completion = token_usage.get(
                "completion_tokens", token_usage.get("output_tokens")
            )
            if isinstance(prompt, int) and isinstance(completion, int):
                return {"prompt": prompt, "completion": completion}
    return None


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float | None:
    """按配置的单价换算成本；未配置价格时返回 None（不记指标）。"""
    price_in = settings.llm_price_input_per_million
    price_out = settings.llm_price_output_per_million
    if price_in <= 0 and price_out <= 0:
        return None
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def record_usage(message: Any, *, stage: str = "agent") -> dict[str, int] | None:
    """记录一次模型调用的 token 与成本。

    `stage` 用来区分是"决策"还是"生成"——两段的输入输出比例差别很大
    （生成把整段证据塞进上下文，prompt token 通常是决策的几倍）。
    """
    usage = extract_usage(message)
    if usage is None:
        obs.inc("rag_llm_usage_missing_total", stage=stage)
        logger.debug("llm_usage_missing stage=%s model=%s", stage, settings.chat_model)
        return None

    obs.inc("rag_llm_tokens_total", amount=usage["prompt"],
            direction="prompt", stage=stage, model=settings.chat_model)
    obs.inc("rag_llm_tokens_total", amount=usage["completion"],
            direction="completion", stage=stage, model=settings.chat_model)

    cost = estimate_cost(usage["prompt"], usage["completion"])
    if cost is not None:
        obs.inc("rag_llm_cost_total", amount=cost,
                currency=settings.llm_price_currency, stage=stage)
    logger.info(
        "llm_usage stage=%s prompt=%d completion=%d cost=%s",
        stage, usage["prompt"], usage["completion"],
        f"{cost:.6f}" if cost is not None else "n/a",
    )
    return usage
