from pathlib import Path

from ingest import group_for, paragraph_chunks


def test_group_comes_from_first_folder():
    assert group_for(Path("notice.md")) == "public"
    assert group_for(Path("finance/policy.md")) == "finance"
    assert group_for(Path("finance/2026/policy.md")) == "finance"


def test_paragraph_chunks_keep_section_and_size():
    text = "# 报销流程\n\n第一段。\n\n第二段。\n\n" + "很长。" * 100
    chunks = paragraph_chunks(text, chunk_size=40, overlap=5)
    assert chunks
    assert all(section == "报销流程" for _, section in chunks)
    assert all(len(chunk) <= 40 for chunk, _ in chunks)

