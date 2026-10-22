"""启动期环境修补：让被 httpx 解析不了的 NO_PROXY 条目变成合法写法。

为什么需要这个模块：

httpx 会把 `NO_PROXY` 里的**每个条目当作 URL 解析**。Windows 上常见的写法
`NO_PROXY=127.0.0.1,localhost,::1,[::1]` 里，`[::1]` 会被解析成 host=`[`、port=`:1]`，
直接抛 `InvalidURL: Invalid port: ':1]'`。而 `[::1]` 本身是合法的 no_proxy 写法
（RFC 里 IPv6 字面量用方括号），所以这是 httpx 的解析限制，不是使用者的错。

后果很隐蔽：`huggingface_hub` / `sentence_transformers` 一创建 HTTP 客户端就崩，
表现为**服务启动时报一大坨与业务无关的 traceback**。实测 `HF_HUB_OFFLINE=1` 也挡不住
（新版 sentence-transformers 仍会为查模型元数据发 HEAD 请求）。

处理方式只做**最小规范化**：把 `[::1]` 这类方括号形式去掉方括号（`::1`），
语义不变（回环地址依然被排除在代理之外），且 httpx 能正常解析。
其它条目原样保留，绝不擅自删改。

用法：必须在**导入 huggingface_hub / sentence_transformers 之前**调用。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("rag-kb")


def _normalize_entry(entry: str) -> str:
    """`[::1]` -> `::1`；其它形式原样返回。"""
    stripped = entry.strip()
    if len(stripped) >= 2 and stripped.startswith("[") and stripped.endswith("]"):
        return stripped[1:-1]
    return stripped


def sanitize_proxy_env() -> list[str]:
    """规范化 NO_PROXY/no_proxy，返回被修改的环境变量名（未修改则为空）。

    遍历 `os.environ` 里实际存在的键，而不是硬编码大小写：Windows 的环境变量名
    大小写不敏感，硬写两个名字容易覆盖掉同一个条目。
    """
    changed: list[str] = []
    names = [name for name in os.environ if name.upper() == "NO_PROXY"]
    for name in names:
        raw = os.environ.get(name) or ""
        if "[" not in raw:
            continue
        entries: list[str] = []
        for part in raw.split(","):
            item = _normalize_entry(part)
            # 去重：`::1` 与 `[::1]` 规范化后是同一个地址，重复条目没有意义
            if item and item not in entries:
                entries.append(item)
        cleaned = ",".join(entries)
        if cleaned != raw:
            os.environ[name] = cleaned
            changed.append(name)

    if changed:
        logger.warning(
            "NO_PROXY 含 httpx 无法解析的方括号条目，已规范化：%s -> %s",
            ",".join(changed), os.environ.get(changed[0]),
        )
    return changed
