"""查询侧的实体选择：用 LLM 从候选实体里挑真正相关的，替代字面匹配。

为什么改：原实现用「二元组重叠」把问题和实体对上，结果问题里的「制度」二字
把「社會保險制度」「退休年金制度」「师徒制度」全匹配进来了——候选池被噪声灌满，
这是图检索输给向量检索的直接原因。

现在的两步：
1. 便宜的字面/二元组匹配先粗筛出候选实体（可能几十个）
2. 让 LLM 从这份候选里挑出与问题真正相关的

第 2 步只花 1 次 LLM 调用/查询，而且是在**封闭集合里做选择**（不是自由生成），
比让模型凭空抽关键词更可控——抽取出的实体一定在索引里。

结果缓存到 `query_keywords.json` 并提交，让复现不需要密钥。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
CACHE = HERE / "query_keywords.json"

PROMPT = """下面是一个用户问题，以及从知识库实体表里粗筛出的候选实体。

问题：{question}

候选实体：
{candidates}

请挑出**真正能帮助回答这个问题**的实体，只输出 JSON 数组，不要解释。
规则：
- 只从候选里挑，不要新增
- 宁可少挑，不要挑只有一两个字重合的
- 最多 8 个
"""


def candidates_for(question: str, entity_to_chunks: dict, limit: int = 40) -> list[str]:
    """粗筛：字面包含或二元组重叠。这一步不追求准，只求别漏。"""
    from run_experiment import _bigrams

    question_bigrams = _bigrams(question)
    matched = [
        entity for entity in entity_to_chunks
        if entity in question or _bigrams(entity) & question_bigrams
    ]
    # 长实体（信息量更大）优先，避免短词占满名额
    matched.sort(key=len, reverse=True)
    return matched[:limit]


def select(question: str, candidates: list[str], llm) -> list[str]:
    """让 LLM 从候选里挑相关实体。失败时退化为原候选（不阻断实验）。"""
    if not candidates:
        return []
    message = llm.invoke(
        PROMPT.format(question=question, candidates="\n".join(f"- {item}" for item in candidates))
    )
    raw = message.content if isinstance(message.content, str) else str(message.content)
    start, end = raw.find("["), raw.rfind("]")
    if start < 0:
        return candidates
    try:
        picked = json.loads(raw[start:end + 1])
    except ValueError:
        return candidates
    # 只接受确实在候选集合里的，防止模型编造
    allowed = set(candidates)
    return [item for item in picked if isinstance(item, str) and item in allowed][:8]


def build_cache(questions: list[dict], entity_to_chunks: dict) -> dict:
    """对每个问题跑一次 LLM 选择，结果落盘。"""
    from app import get_llm

    llm = get_llm()
    payload = {}
    for case in questions:
        candidates = candidates_for(case["question"], entity_to_chunks)
        picked = select(case["question"], candidates, llm)
        payload[case["id"]] = {"question": case["question"], "candidates": len(candidates),
                               "selected": picked}
        print(f"[{case['id']}] 候选 {len(candidates)} -> 选中 {len(picked)}: {picked}")

    CACHE.write_text(
        json.dumps({
            "note": "查询侧实体选择结果（LLM 在候选集合内挑选）。已入库，复现不需要密钥。",
            "questions": payload,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {CACHE.name}")
    return payload


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    return json.loads(CACHE.read_text(encoding="utf-8"))["questions"]
