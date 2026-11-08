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
from hybrid import bm25_search, rrf_fuse  # noqa: E402
import obs  # noqa: E402
from admission import AdmissionController
from device import resolve  # noqa: E402
from rerank import rerank  # noqa: E402
from vector_backend import (  # noqa: E402
    BackendConfigError,
    VectorBackend,
    open_backend,
)

ROOT = Path(__file__).parent
INDEX = ROOT / "index.json"

logger = logging.getLogger("rag-kb")
# bge 系列检索指令：查询侧加指令、文档侧不加（非对称检索）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_CANDIDATES = 16  # 精排候选封顶，控制 CPU 耗时


class IndexConfigError(RuntimeError):
    """索引文件与当前配置/代码不兼容（schema 版本、向量维度或 embedding 模型不一致）。"""


_EMBEDDING_DEVICE = resolve(settings.embedding_device)
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={"prompt": QUERY_INSTRUCTION},
    model_kwargs={"device": _EMBEDDING_DEVICE},
)


class IndexManager:
    """按配置打开向量后端，并在 JSON 后端下支持按 mtime 热更新。

    Milvus 不做 mtime 热更新：数据在库里而不是文件里，重建索引由 ingest 完成。
    """

    def __init__(
        self,
        *,
        backend: str,
        index_path: Path,
        milvus_uri: str,
        milvus_collection: str,
        milvus_metric: str = "COSINE",
    ):
        self.backend = backend
        self.path = index_path
        self.milvus_uri = milvus_uri
        self.milvus_collection = milvus_collection
        self.milvus_metric = milvus_metric
        self._store: VectorBackend | None = None
        self._mtime_ns: int | None = None
        self._lock = RLock()

    def get(self) -> VectorBackend:
        with self._lock:
            if self.backend == "json":
                return self._get_json()
            if self._store is None:
                self._store = self._open()
            return self._store

    def _open(self) -> VectorBackend:
        try:
            store = open_backend(
                self.backend,
                embedding=embeddings,
                index_path=self.path,
                milvus_uri=self.milvus_uri,
                milvus_collection=self.milvus_collection,
                milvus_metric=self.milvus_metric,
            )
        except BackendConfigError as exc:  # 模型/维度不一致
            raise IndexConfigError(str(exc)) from exc
        indexed_model = store.index_metadata.get("embedding_model")
        if indexed_model and indexed_model != settings.embedding_model:
            raise IndexConfigError(
                f"Index uses {indexed_model}, configured model is {settings.embedding_model}"
            )
        logger.info(
            "Loaded index backend=%s records=%d", self.backend, store.count()
        )
        return store

    def _get_json(self) -> VectorBackend:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        mtime_ns = self.path.stat().st_mtime_ns
        if self._store is None or self._mtime_ns != mtime_ns:
            self._store, self._mtime_ns = self._open(), mtime_ns
        return self._store


index_manager = IndexManager(
    backend=settings.vector_backend,
    index_path=INDEX,
    milvus_uri=settings.milvus_uri,
    milvus_collection=settings.milvus_collection,
    milvus_metric=settings.milvus_metric,
)

# 准入控制：只包住精排（见 retrieve_ranked 的说明）。
# workers=1 是实测最优（一个请求吃满 8 线程），queue_max 与 timeout 决定过载时
# 是"快速失败"还是"一起超时"。
admission_controller = AdmissionController(
    workers=settings.admission_workers,
    queue_max=settings.admission_queue_max,
    queue_timeout=settings.admission_queue_timeout,
    rate=settings.admission_rate,
    burst=settings.admission_burst,
)


def admission_enabled() -> bool:
    return settings.admission_enabled


class _NullSlot:
    """准入关闭时的空实现，避免调用点写 if/else。"""

    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


def admission_slot():
    """取一个执行位：开启时排队，关闭时直接放行。

    用一个函数而不是在调用点判断开关，是为了让"关闭准入"这条路径也走同一个
    代码形状——否则"关掉准入"的行为从来没有被测试覆盖过。
    """
    return admission_controller.slot() if settings.admission_enabled else _NullSlot()


def retrieve(question: str, top_k: int, principal: Principal):
    """粗排召回。

    默认走**混合检索**：向量召回 + BM25 关键词召回，用 RRF 融合。
    纯向量的弱点是精确词面（「折现」「双签」这类制度用语容易被泛化到相邻话题），
    BM25 对此敏感，两者互补。详见 hybrid.py 的说明与实测数据。
    """
    store = index_manager.get()

    groups = principal.allowed_groups
    dense = store.search(
        question, k=top_k, groups=groups, score_threshold=settings.score_threshold
    )
    if not settings.hybrid_enabled:
        return dense

    sparse = bm25_search(store, question, k=top_k, groups=groups)
    if not sparse:
        return dense

    fused = rrf_fuse([dense, sparse], top_k)
    logger.info("retrieve_fuse dense=%d bm25=%d fused=%d", len(dense), len(sparse), len(fused))
    return fused


def candidate_k(top_k: int) -> int:
    """粗排召回候选数：精排需要比最终 top_k 更多的候选。"""
    return min(top_k * settings.rerank_candidate_factor, MAX_CANDIDATES)


def retrieve_ranked(question: str, top_k: int, principal: Principal):
    """两段式检索：向量粗排召回候选 -> cross-encoder 精排过滤。

    精排是整条链路里唯一"贵"的阶段，因此只有它在准入控制的闸门后面。
    **候选为空时直接返回，不取闸门**——否则一次 20ms 的拒答会被前面正在跑的
    精排卡住，把快路径的吞吐全部赔掉。

    计时分两段：`admission_wait` 是排队等待，`stage=rerank` 才是真正的计算。
    早先把两者算在一起，导致换了 GPU 之后指标仍报"精排 3.3 秒"——而实测计算
    只要 0.2 秒，其余全是排队。**把排队算进计算，会让"优化有没有效"完全不可判断。**
    """
    started = perf_counter()
    candidates = retrieve(question, candidate_k(top_k), principal)
    coarse_ms = (perf_counter() - started) * 1000

    rerank_ms = 0.0
    wait_ms = 0.0
    if candidates:
        entered = perf_counter()
        with admission_slot():
            wait_ms = (perf_counter() - entered) * 1000
            rerank_started = perf_counter()
            hits = rerank(question, candidates, top_n=top_k, threshold=settings.rerank_threshold)
            rerank_ms = (perf_counter() - rerank_started) * 1000
    else:
        hits = []

    obs.observe("rag_stage_seconds", coarse_ms / 1000, stage="coarse")
    obs.observe("rag_stage_seconds", rerank_ms / 1000, stage="rerank")
    if wait_ms:
        obs.observe("rag_stage_seconds", wait_ms / 1000, stage="admission_wait")
    obs.inc("rag_retrieval_hits_total", amount=len(hits))
    logger.info(
        "retrieve top_k=%d candidates=%d hits=%d coarse_ms=%d rerank_ms=%d",
        top_k, len(candidates), len(hits), coarse_ms, rerank_ms,
    )
    return hits
