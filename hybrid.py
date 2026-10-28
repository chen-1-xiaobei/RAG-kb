"""混合检索：BM25 关键词召回与向量召回融合（RRF）。

为什么需要它：实测出来的一个具体失败——把精排候选数从 8 降到 4 之后，
「没休完能折现吗」这一条的正确来源《员工手册》被《加班》《考勤与请假细则》
挤出了候选集，精排再准也救不回来（**精排只能重排候选，不能凭空召回**）。

纯向量召回的弱点是**精确词面**：制度里的「折现」「双签」「代开发票」这类词，
语义模型可能把它泛化到相邻话题；而 BM25 恰恰对稀有词敏感。两者互补。

融合用 RRF（Reciprocal Rank Fusion）而不是加权求和：两路的分数尺度完全不同
（余弦在 [0,1]，BM25 无上界），归一化再相加需要额外调参，RRF 只看名次，
天然可比，也不需要标定。

中文分词没有引入第三方库：用**字符二元组**（bigram）即可——中文没有空格，
bigram 在这个规模下的检索效果与词级分词接近，且不引入词典依赖。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable

from langchain_core.documents import Document

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60          # RRF 的平滑常数，取文献常用值

_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中文按字符二元组，英文/数字按整词。

    单字也保留：像「税」「假」这种能在制度文本里单独成义的词，
    丢掉会让短查询（用户只打一个词）召回变差。
    """
    lowered = text.lower()
    tokens: list[str] = _ASCII_WORD.findall(lowered)
    chinese = [ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"]
    tokens.extend(chinese)
    tokens.extend(
        chinese[index] + chinese[index + 1] for index in range(len(chinese) - 1)
    )
    return tokens


class BM25Index:
    """内存 BM25。语料是万级 chunk 以内，全量打分足够快（毫秒级）。"""

    def __init__(self, texts: Iterable[str]):
        self._term_freqs: list[Counter] = []
        self._lengths: list[int] = []
        document_freq: Counter = Counter()

        for text in texts:
            tokens = tokenize(text)
            freq = Counter(tokens)
            self._term_freqs.append(freq)
            self._lengths.append(len(tokens))
            document_freq.update(freq.keys())

        self._document_freq = document_freq
        self._total = len(self._lengths)
        self._avg_length = (sum(self._lengths) / self._total) if self._total else 0.0

    def scores(self, query: str) -> list[float]:
        tokens = tokenize(query)
        if not tokens or not self._total:
            return [0.0] * self._total

        import math

        results = [0.0] * self._total
        for token in set(tokens):
            df = self._document_freq.get(token, 0)
            if df == 0:
                continue
            # 标准 BM25 IDF；加 1 防止负值（df 很小时 log 为负）
            idf = math.log(1 + (self._total - df + 0.5) / (df + 0.5))
            for index, freq in enumerate(self._term_freqs):
                count = freq.get(token, 0)
                if not count:
                    continue
                length_norm = 1 - BM25_B + BM25_B * (self._lengths[index] / self._avg_length)
                results[index] += idf * (count * (BM25_K1 + 1)) / (count + BM25_K1 * length_norm)
        return results


def rrf_fuse(
    channels: list[list[tuple[Document, float]]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """按名次融合多路召回结果。

    同一文档出现在多路里会累加分值——这正是我们要的："向量觉得相关"且
    "词面也命中"的块，应该排在只有一路命中的块前面。
    """
    fused: dict[str, float] = {}
    by_id: dict[str, tuple[Document, float]] = {}

    for channel in channels:
        for rank, (doc, score) in enumerate(channel):
            doc_id = doc.metadata.get("id") or f"{doc.metadata.get('source', '')}|{doc.page_content[:40]}"
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            # 保留原始分数最高的那次，便于排查
            if doc_id not in by_id or score > by_id[doc_id][1]:
                by_id[doc_id] = (doc, score)

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(by_id[doc_id][0], fused_score) for doc_id, fused_score in ranked]


def bm25_search(
    store,
    query: str,
    k: int,
    filter_fn: Callable[[dict], bool] | None = None,
) -> list[tuple[Document, float]]:
    """在允许的组内做 BM25 检索。

    IDF 用**全语料**统计（不是只统计允许的组）：否则不同用户的 IDF 会不同，
    同一个查询在不同权限下打分不一致，排查问题时会很困惑。
    """
    index = _index_for(store)
    scores = index.scores(query)

    allowed = [
        i for i, record in enumerate(store._records)
        if filter_fn is None or filter_fn(record)
    ]
    allowed.sort(key=lambda i: scores[i], reverse=True)

    results: list[tuple[Document, float]] = []
    for i in allowed[:k]:
        if scores[i] <= 0:
            continue
        results.append((store._to_document(i), float(scores[i])))
    return results


# 索引按 (id(store), 记录数) 缓存：索引重建后记录数变化即失效
_CACHE: dict[tuple[int, int], BM25Index] = {}


def _index_for(store) -> BM25Index:
    key = (id(store), len(store._records))
    cached = _CACHE.get(key)
    if cached is None:
        cached = BM25Index(record["text"] for record in store._records)
        _CACHE.clear()          # 只保留最新一份，避免旧索引占内存
        _CACHE[key] = cached
    return cached
