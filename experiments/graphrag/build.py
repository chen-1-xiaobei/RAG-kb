"""构建实体图：每个 chunk 抽实体，建 chunk–实体、实体–chunk 的双向索引。

    python experiments/graphrag/build.py           # 首次构建（需要 DEEPSEEK_API_KEY）
    python experiments/graphrag/build.py --force   # 强制重抽

产物 `entities.json` 会提交进仓库：图只建一次，之后查询与实验都不需要密钥。

**为什么抽实体而不是直接用全文检索**：聚合型问题的关键是"找出散落在所有文档里的
同一类东西"（比如所有出现「工作日」的条款）。向量检索给的是"和问题最像的 4 段"，
天然做不到覆盖；实体索引给的是"所有提到 X 的段"，再交给精排决定怎么用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
INDEX = ROOT / "index.json"
OUTPUT = HERE / "entities.json"

PROMPT = """从下面的企业制度文本中抽取关键实体。只输出 JSON 数组，不要解释。

要求：
- 每个实体 2-8 个字，是文本里出现的**名词性短语**
- 优先抽取：制度/流程名称、部门与角色、金额、时限、关键业务术语
- 最多 8 个，宁缺毋滥；不要抽虚词、不要造词
- 实体的字面必须能在原文中找到

示例输出：["报销流程", "财务部", "五个工作日", "纸质发票"]

文本：
{text}
"""


def load_chunks() -> list[dict]:
    payload = json.loads(INDEX.read_text(encoding="utf-8"))
    return payload["records"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略缓存重新抽取")
    args = parser.parse_args(argv)

    if OUTPUT.exists() and not args.force:
        print(f"{OUTPUT.name} 已存在，跳过（用 --force 重抽）")
        return 0

    # 延迟导入：只有构建阶段需要密钥与网络
    from app import get_llm

    llm = get_llm()
    chunks = load_chunks()
    results = []
    failures = 0

    for position, chunk in enumerate(chunks, 1):
        text = chunk["text"]
        try:
            message = llm.invoke(PROMPT.format(text=text[:1200]))
            raw = message.content if isinstance(message.content, str) else str(message.content)
            start, end = raw.find("["), raw.rfind("]")
            entities = json.loads(raw[start:end + 1]) if start >= 0 else []
        except Exception as exc:                       # 单条失败不中断整体构建
            failures += 1
            print(f"  [{position}] 抽取失败: {exc}", file=sys.stderr)
            entities = []

        # 只保留确实出现在原文里的实体，防止模型造词污染索引
        kept = []
        for entity in entities:
            if not isinstance(entity, str):
                continue
            entity = entity.strip()
            if 2 <= len(entity) <= 8 and entity in text and entity not in kept:
                kept.append(entity)

        results.append({
            "id": chunk["id"],
            "source": chunk.get("source", ""),
            "entities": kept,
        })
        if position % 20 == 0:
            print(f"  已处理 {position}/{len(chunks)}")

    OUTPUT.write_text(
        json.dumps(
            {
                "note": "实体抽取结果（一次性 LLM 产物，已提交入库）。"
                        "只保留能在原文中找到字面的实体，防止模型造词。",
                "chunk_count": len(results),
                "failures": failures,
                "chunks": results,
            },
            ensure_ascii=False,
            indent=1,
        ) + "\n",
        encoding="utf-8",
    )
    total_entities = sum(len(item["entities"]) for item in results)
    print(f"已写入 {OUTPUT.name}: {len(results)} 个 chunk, {total_entities} 个实体, 失败 {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
