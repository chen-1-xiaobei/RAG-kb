"""文档标准化层：把 .md/.txt/.pdf/.docx 统一转换为 markdown 文本再进入切分管道。

设计：
- 每个格式一个可插拔的 loader（LOADERS 注册表），新增格式只加注册项
- PDF 用 pymupdf4llm 版面分析转 markdown（标题/表格/阅读顺序保真，
  布局模型打包在 wheel 内、无 HF 下载），失败降级 pypdf 纯文本
- 转换产物按源文件内容哈希缓存到 .cache/convert/，源文件变更自动失效

统一契约：load_document(path) -> list[[文本, 页码|None]]，按页切分以保留页码元数据。
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / ".cache" / "convert"
CACHEABLE = {".pdf", ".docx"}
# 转换缓存版本：loader 实现或转换依赖（pymupdf4llm/mammoth）升级后递增，旧缓存自动失效
CACHE_VERSION = 1


def _text_pages(path: Path) -> list[list[Any]]:
    return [[path.read_text(encoding="utf-8"), None]]


def _pdf_markdown_pages(path: Path) -> list[list[Any]]:
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(str(path))
    try:
        return [
            [pymupdf4llm.to_markdown(doc, pages=[page.number]).strip(), page.number + 1]
            for page in doc
        ]
    finally:
        doc.close()


def _pdf_text_pages(path: Path) -> list[list[Any]]:
    from pypdf import PdfReader

    return [
        [page.extract_text() or "", page_number]
        for page_number, page in enumerate(PdfReader(str(path)).pages, 1)
    ]


def _pdf_pages(path: Path) -> list[list[Any]]:
    """PDF：优先版面分析转 markdown，失败降级 pypdf 纯文本（不阻断索引）。"""
    try:
        return _pdf_markdown_pages(path)
    except Exception:
        return _pdf_text_pages(path)


def _docx_markdown_pages(path: Path) -> list[list[Any]]:
    import mammoth

    with path.open("rb") as fh:
        result = mammoth.convert_to_markdown(fh)
    return [[result.value, None]]


def _html_markdown_pages(path: Path) -> list[list[Any]]:
    from markdownify import markdownify as md

    html_content = path.read_text(encoding="utf-8")
    markdown_text = md(html_content, heading_style="ATX").strip()
    return [[markdown_text, None]]   # HTML 没有页码概念，页码设为 None


LOADERS: dict[str, Callable[[Path], list[list[Any]]]] = {
    ".md": _text_pages,
    ".txt": _text_pages,
    ".pdf": _pdf_pages,
    ".docx": _docx_markdown_pages,
    ".html": _html_markdown_pages,
}


def _cache_path(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return CACHE_DIR / f"v{CACHE_VERSION}-{digest}.json"


def _load_cached(path: Path) -> list[list[Any]] | None:
    cached = _cache_path(path)
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    return None


def _store_cache(path: Path, pages: list[list[Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(path).write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")


def load_document(path: Path) -> list[list[Any]]:
    """统一入口：任意支持格式 -> [[markdown 文本, 页码|None], ...]。"""
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix in CACHEABLE:
        cached = _load_cached(path)
        if cached is not None:
            return cached
    pages = loader(path)
    if suffix in CACHEABLE:
        _store_cache(path, pages)
    return pages
