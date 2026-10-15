"""Read local documents, split into paragraph-based chunks with metadata, persist index.json.

权限约定（约定优于配置）：
- data/ 根目录文件 -> group="public"，所有人可见
- data/<文件夹>/ 内文件 -> group=文件夹名，仅该组成员可见
"""
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
DATA = ROOT / "data"
INDEX = ROOT / "index.json"
# 必须先于 huggingface_hub 相关导入执行：HF_HUB_OFFLINE 在 huggingface_hub 导入时读取一次
load_dotenv(ROOT / ".env")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from kb_store import JSONVectorStore
from loaders import load_document,LOADERS

EMBEDDING_MODEL = settings.embedding_model
CHUNK_SIZE = 700
OVERLAP = 100
PUBLIC_GROUP = "public"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# 超长单段落兜底切分用的中文分隔符
CN_SEPARATORS = ["。", "！", "？", "；", "，", " ", ""]


def paragraph_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> list[tuple[str, str]]:
    """按段落切分并合并成块，返回 [(块文本, 所属章节标题)]。

    段落边界 = 语义边界：短段落连续合并到 chunk_size 为止，绝不跨段截断，
    因此天然不需要字符级 overlap；超长单段落交给 RecursiveCharacterTextSplitter 兜底。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=CN_SEPARATORS
    )
    chunks: list[tuple[str, str]] = []
    buf, buf_section = "", ""
    for para in paragraphs:
        heading = HEADING_RE.match(para)
        if heading and len(para) <= 60:  # 标题行：更新章节上下文，不参与正文缓冲
            if buf:
                chunks.append((buf, buf_section))
                buf = ""
            buf_section = heading.group(2)
            continue
        if len(para) > chunk_size:  # 超长段落兜底切分
            if buf:
                chunks.append((buf, buf_section))
                buf = ""
            chunks.extend((piece, buf_section) for piece in splitter.split_text(para))
            continue
        if buf and len(buf) + len(para) + 1 > chunk_size:
            chunks.append((buf, buf_section))
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append((buf, buf_section))
    return chunks


def group_for(rel_path: Path) -> str:
    """data/ 根目录 -> public；data/<文件夹>/ -> 一级文件夹名。"""
    parts = rel_path.parts[:-1]  # 去掉文件名
    return PUBLIC_GROUP if not parts else parts[0]


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    documents = []
    for path in DATA.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue
        rel = path.relative_to(DATA)
        folder = str(rel.parent) if str(rel.parent) != "." else ""
        group = group_for(rel)
        pages = load_document(path)  # 文档标准化层：统一转 markdown（PDF 版面分析 + 缓存 + 降级）
        text = "\n".join(page_text for page_text, _ in pages)
        # 标题：文档首个 # 标题，否则用文件名
        title = m.group(1) if (m := re.search(r"^#\s+(.+)$", text, flags=re.M)) else path.stem
        n = 0
        for page_text, page_number in pages:
            for chunk_text, section in paragraph_chunks(page_text):
                documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "source": path.name,
                            "relative_path": rel.as_posix(),
                            "folder": folder,
                            "group": group,
                            "title": title,
                            "section": section,
                            "page": page_number,
                            "para": n,
                        },
                    )
                )
                n += 1
    if not documents:
        raise SystemExit(f"data/ 中没有可索引的 {', '.join(LOADERS)} 文件")
    store = JSONVectorStore.from_documents(documents, embeddings, path=INDEX)
    groups = sorted({doc.metadata["group"] for doc in documents})
    print(f"indexed {len(documents)} chunks (groups: {', '.join(groups)}) -> {INDEX}")


if __name__ == "__main__":
    main()
