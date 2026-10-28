"""抓取中文维基百科条目，转成 markdown 放进 data/，用于把语料扩到"召回会失败"的规模。

为什么需要它：演示语料原本只有 5 篇文档、34 个 chunk，而精排候选数是 16 ——
top-16 已经覆盖语料的一半，**召回率不可能失败**，评测指标于是全部饱和在 100%。
一个召回率恒为 100% 的检索评测测不出任何东西。

选材原则：
- 与内部制度**同领域**（报销、差旅、发票、考勤、年假、采购、招聘……），
  词汇重叠度高，才构成真实的干扰；
- 但**不与之矛盾**：百科条目讲的是概念与通行做法，不给出"本公司三十天"这类具体值；
- 不含法条文本（法律与内部制度的具体数值容易互相冲突，会把检索题变成歧义题）。

数据来源：中文维基百科（zh.wikipedia.org），条目文本依 CC BY-SA 4.0 授权，
此处保留来源说明，仅做格式整理。

    python scripts/fetch_public_docs.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# NO_PROXY 里的方括号写法会让 httpx/urllib 解析失败，先规范化；随后按环境代理访问。
from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

DATA = ROOT / "data"
API = "https://zh.wikipedia.org/w/api.php"

TITLES = [
    "报销",
    "差旅费",
    "发票",
    "考勤",
    "年假",
    "绩效管理",
    "采购",
    "招聘",
    "社会保险",
    "商务旅行",
    "工资",
    "加班",
    "预算",
    "审计",
    "内部控制",
    "供应链管理",
    "人力资源管理",
    "员工福利",
    "培训",
    "知识管理",
    "会计",
    "增值税",
    "企业所得税",
    "成本控制",
    "劳动合同",
]


def fetch_extract(title: str, attempts: int = 4) -> str | None:
    """取条目的纯文本摘要（explaintext=1 已去掉维基标记）。

    经代理访问，偶发 `SSL: UNEXPECTED_EOF_WHILE_READING`——这不是条目不存在，
    重试即可；不重试会把一半文档静默漏掉。
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "redirects": "1",
        "titles": title,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "rag-kb-doc-fetch/1.0"})

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            # 走环境代理：本机对外访问需要经过 HTTP(S)_PROXY，直连会超时
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:                  # noqa: BLE001 - 网络异常统一重试
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"抓取失败（重试 {attempts} 次）：{last_error}")

    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract") or ""
        if len(text) < 200:            # 条目不存在或只有一句话
            return None
        return text
    return None


def to_markdown(title: str, text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    body = "\n".join(lines)
    import re
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    header = (
        f"# {title}\n\n"
        f"> 来源：中文维基百科条目「{title}」，依 CC BY-SA 4.0 授权，"
        f"内容经格式整理。\n\n"
    )
    return header + body + "\n"


def main() -> int:
    DATA.mkdir(exist_ok=True)
    written = 0
    for title in TITLES:
        target = DATA / f"{title}.md"
        try:
            text = fetch_extract(title)
        except Exception as exc:                  # 网络问题不应打断其他条目
            print(f"跳过 {title}: {exc}", file=sys.stderr)
            continue
        if not text:
            print(f"跳过 {title}: 条目过短或不存在", file=sys.stderr)
            continue
        markdown = to_markdown(title, text)
        target.write_text(markdown, encoding="utf-8")
        written += 1
        print(f"已写入 {target.relative_to(ROOT)}  ({len(markdown)} 字)")
    print(f"共写入 {written} 篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
