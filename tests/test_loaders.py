import pytest

import loaders
from loaders import LOADERS, _pdf_pages, load_document


def test_text_passthrough(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# 标题\n\n正文", encoding="utf-8")
    assert load_document(path) == [["# 标题\n\n正文", None]]


def test_cache_hit_and_source_change_invalidation(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_pdf(path):
        calls["n"] += 1
        return [["第一页", 1], ["第二页", 2]]

    monkeypatch.setitem(LOADERS, ".pdf", fake_pdf)
    monkeypatch.setattr(loaders, "CACHE_DIR", tmp_path / "cache")
    path = tmp_path / "a.pdf"
    path.write_bytes(b"pdf-bytes")
    assert load_document(path) == [["第一页", 1], ["第二页", 2]]
    assert load_document(path) == [["第一页", 1], ["第二页", 2]]  # 缓存命中
    assert calls["n"] == 1
    path.write_bytes(b"changed")  # 源文件变更 -> 内容哈希变化 -> 缓存失效
    assert load_document(path) == [["第一页", 1], ["第二页", 2]]
    assert calls["n"] == 2


def test_cache_version_invalidates_old_cache(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_pdf(path):
        calls["n"] += 1
        return [["内容", 1]]

    monkeypatch.setitem(LOADERS, ".pdf", fake_pdf)
    monkeypatch.setattr(loaders, "CACHE_DIR", tmp_path / "cache")
    path = tmp_path / "a.pdf"
    path.write_bytes(b"pdf-bytes")
    load_document(path)
    load_document(path)
    assert calls["n"] == 1  # 缓存命中
    monkeypatch.setattr(loaders, "CACHE_VERSION", loaders.CACHE_VERSION + 1)  # 转换逻辑升级
    load_document(path)
    assert calls["n"] == 2  # 旧版本缓存不再命中


def test_pdf_falls_back_to_pypdf_on_converter_failure(tmp_path, monkeypatch):
    def boom(path):
        raise RuntimeError("converter broken")

    def fake_pypdf(path):
        return [["降级文本", 1]]

    monkeypatch.setattr(loaders, "_pdf_markdown_pages", boom)
    monkeypatch.setattr(loaders, "_pdf_text_pages", fake_pypdf)
    assert _pdf_pages(tmp_path / "x.pdf") == [["降级文本", 1]]


def test_html_to_markdown(tmp_path):
    path = tmp_path / "a.html"
    path.write_text("<h1>标题</h1><p>正文</p>", encoding="utf-8")
    assert load_document(path) == [["# 标题\n\n正文", None]]

def test_unsupported_suffix_raises(tmp_path):
    with pytest.raises(ValueError):
        load_document(tmp_path / "a.exe")
