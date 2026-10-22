"""两段式检索管线：向量粗排 + cross-encoder 精排。

单独成模块的原因（不只是"拆文件"）：
- `app.py` 在 import 期构造 LLM 客户端（需要 DEEPSEEK_API_KEY），而检索与评测
  **完全不需要 LLM**。检索留在这里，评测与任何离线脚本就能独立于 LLM 运行，
  CI 也不必配置任何密钥——这是让评测能作为硬门禁的前提。
- 索引加载、embedding 模型、rerank 都只被这一条链路依赖，聚在一处职责更清楚。

依赖方向：retrieval 不 import app（app 单向 import 它），因此不存在循环导入。
"""
import logging
from pathlib import Path
from threading import RLock
from time import perf_counter

# 必须早于 langchain_huggingface 导入：它会拉起 huggingface_hub，
# 后者一创建 HTTP 客户端就会因 NO_PROXY 里的方括号条目抛 InvalidURL。
from envfix import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from auth import Principal  # noqa: E402
from config import settings  # noqa: E402
from kb_store import JSONVectorStore  # noqa: E402
from rerank import rerank  # noqa: E402

ROOT = Path(__file__).parent
INDEX = ROOT / "index.json"

logger = logging.getLogger("rag-kb")
# bge 系列检索指令：查询侧加指令、文档侧不加（非对称检索）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_CANDIDATES = 16  # 精排候选封顶，控制 CPU 耗时


class IndexConfigError(RuntimeError):
    """索引文件与当前配置/代码不兼容（schema 版本、向量维度或 embedding 模型不一致）。"""


embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={"prompt": QUERY_INSTRUCTION},
)


class IndexManager:
    def __init__(self, path: Path):
        self.path = path
        self._store: JSONVectorStore | None = None
        self._mtime_ns: int | None = None
        self._lock = RLock()

    def get(self) -> JSONVectorStore:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        mtime_ns = self.path.stat().st_mtime_ns
        with self._lock:
            if self._store is None or self._mtime_ns != mtime_ns:
                try:
                    store = JSONVectorStore.load(self.path, embeddings)
                except ValueError as exc:  # schema 版本/向量维度等索引自身问题
                    raise IndexConfigError(str(exc)) from exc
                indexed_model = store.index_metadata.get("embedding_model")
                if indexed_model and indexed_model != settings.embedding_model:
                    raise IndexConfigError(
                        f"Index uses {indexed_model}, configured model is {settings.embedding_model}"
                    )
                self._store, self._mtime_ns = store, mtime_ns
                logger.info("Loaded index with %s records", len(store._records))
            return self._store


index_manager = IndexManager(INDEX)


def retrieve(question: str, top_k: int, principal: Principal):
    return index_manager.get().similarity_search_with_relevance_scores(
        question,
        k=top_k,
        score_threshold=settings.score_threshold,
        filter=lambda record: record.get("group") in principal.allowed_groups,
    )


def candidate_k(top_k: int) -> int:
    """粗排召回候选数：精排需要比最终 top_k 更多的候选。"""
    return min(top_k * settings.rerank_candidate_factor, MAX_CANDIDATES)


def retrieve_ranked(question: str, top_k: int, principal: Principal):
    """两段式检索：向量粗排召回候选 -> cross-encoder 精排过滤。"""
    started = perf_counter()
    candidates = retrieve(question, candidate_k(top_k), principal)
    coarse_ms = (perf_counter() - started) * 1000
    rerank_started = perf_counter()
    hits = rerank(question, candidates, top_n=top_k, threshold=settings.rerank_threshold)
    logger.info(
        "retrieve top_k=%d candidates=%d hits=%d coarse_ms=%d rerank_ms=%d",
        top_k, len(candidates), len(hits), coarse_ms, (perf_counter() - rerank_started) * 1000,
    )
    return hits
