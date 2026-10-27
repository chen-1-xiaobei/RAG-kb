"""Cross-encoder 精排：对向量粗排的候选按语义相关性重排并过滤。

与 embedding（bi-encoder）不同，cross-encoder 把问题和文档拼接成单输入做全注意力
交互，精度高但不能预计算，因此只用于粗排后的少量候选（两段式检索）。
分数保留模型的原始 logit（无界；0 约为中性/无关，越大越相关）。
标定实测：sigmoid 会把接近中性的分数压缩到 0.5 附近，噪声与相关块的
判别度只剩千分位；原始 logit 的判别余量是几十倍，因此不用 sigmoid。

**逐对结果缓存**：精排是整条链路最贵的步骤（本机 568M 模型纯 CPU 约 0.6s/对），
而 agent 循环里模型会「换个关键词再试」——同一批候选块会被反复重排。
缓存键为 (模型, question, doc_id, 文档内容哈希)，内容哈希用于在索引重建后自动失效；
只缓存 logit 这一纯计算结果，不缓存 top-N/阈值等策略，因此策略参数调整无需清缓存。
"""
import logging
from collections import OrderedDict
from threading import Lock

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings
import obs

logger = logging.getLogger("rag-kb")

# 缓存规模：键约 200 字符（Python 字符串开销约 400B）+ 分数，2 万条约 10MB 量级
_MAX_CACHE_ENTRIES = 20000

_model = None
_model_lock = Lock()
_cache: OrderedDict[str, float] = OrderedDict()
_cache_lock = Lock()


def get_model() -> CrossEncoder:
    """懒加载单例：模型约 2GB，只在首次精排时加载。

    加锁是必要的：FastAPI 把同步端点丢进线程池，首个并发请求会同时进入这里，
    无锁时两个线程各加载一份 2GB 模型，显存/内存直接翻倍。
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = CrossEncoder(settings.rerank_model)
    return _model


def _key(question: str, doc: Document) -> str:
    """缓存键：模型 + 问题 + 文档标识 + 内容哈希。

    内容哈希是必需的——文档被重新摄入后 doc_id 可能不变而内容已变，
    只按 id 缓存会返回与当前内容不符的陈旧分数。
    """
    content_hash = hash(doc.page_content)
    doc_id = doc.metadata.get("id", "")
    return f"{settings.rerank_model}\x1f{question}\x1f{doc_id}\x1f{content_hash}"


def _cache_get(keys: list[str]) -> tuple[list[float | None], int]:
    """按 key 取缓存；返回等长分数列表（未命中为 None）与命中数。"""
    scores: list[float | None] = []
    with _cache_lock:
        for key in keys:
            value = _cache.get(key)
            if value is not None:
                _cache.move_to_end(key)  # LRU：命中即刷新
            scores.append(value)
    return scores, sum(score is not None for score in scores)


def _cache_put(keys: list[str], scores: list[float]) -> None:
    with _cache_lock:
        for key, score in zip(keys, scores):
            _cache[key] = score
            _cache.move_to_end(key)
        while len(_cache) > _MAX_CACHE_ENTRIES:
            _cache.popitem(last=False)  # 淘汰最久未使用


def clear_cache() -> None:
    """清空精排缓存（测试与换模型时用）。"""
    with _cache_lock:
        _cache.clear()


def cache_stats() -> dict[str, int]:
    with _cache_lock:
        return {"entries": len(_cache)}


def rerank(
    question: str,
    hits: list[tuple[Document, float]],
    top_n: int,
    threshold: float,
) -> list[tuple[Document, float]]:
    """对粗排候选精排，返回 [(Document, 原始 logit 分数)]，已按分数降序。

    低于 threshold 的候选直接丢弃（语义不相关）；全部被丢弃时返回 []，
    调用方据此走拒答分支。已缓存的对不会重复送进模型。
    """
    if not hits:
        return []
    # 模型要的是 (str, str)；缓存键要 Document（拿 id 与内容哈希）。两者不能混用——
    # 把 Document 直接喂给 CrossEncoder 会抛 ValueError，而调用方会把它降级成
    # "未检索到相关文档"，表现为静默的检索失效。
    model_pairs = [(question, doc.page_content) for doc, _ in hits]
    keys = [_key(question, doc) for doc, _ in hits]
    cached, cache_hits = _cache_get(keys)
    missing = [index for index, score in enumerate(cached) if score is None]

    if missing:
        fresh = get_model().predict(
            [model_pairs[index] for index in missing],
            batch_size=settings.rerank_batch_size,
        )
        fresh = [float(score) for score in fresh]
        for index, score in zip(missing, fresh):
            cached[index] = score
        _cache_put([keys[index] for index in missing], fresh)

    scored = [(doc, float(score)) for (doc, _), score in zip(hits, cached)]
    if cache_hits:
        logger.info(
            "rerank_cache pairs=%d hits=%d misses=%d", len(model_pairs), cache_hits, len(missing)
        )
    if cache_hits:
        obs.inc("rag_rerank_cache_total", amount=cache_hits, result="hit")
    if missing:
        obs.inc("rag_rerank_cache_total", amount=len(missing), result="miss")
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(doc, score) for doc, score in scored[:top_n] if score >= threshold]


def warmup() -> None:
    """加载并预热模型（lifespan 调用，避免首个请求慢）。"""
    get_model().predict([("预热", "预热")], batch_size=1)
