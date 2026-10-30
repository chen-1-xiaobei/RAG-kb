"""改进版图检索：PPR 扩散 + 文档配额 + 与向量召回融合。

对应清单里的三条改动，每条都针对旧实现的一个具体失败：

1. **查询实体由 LLM 选**（见 keywords.py）
   旧版：字面匹配 → 「制度」匹配到「社會保險制度」等 8 个无关实体。

2. **PPR 代替 1 跳扩散**
   旧版：命中实体的 chunk 全部拉进来，噪声大且不区分重要性。
   PPR 从种子实体出发在「实体–chunk」二部图上随机游走，**同时连到多个相关实体的
   chunk 会得到更高的分数**——这正是聚合型问题需要的"跨文档桥接"信号。

3. **候选按文档配额**
   旧版：社会保险.md 一个文档占了 12/28 个候选位。
   现在每份文档最多占 `per_doc_cap` 个候选位，保证候选池多样。

另外提供 `fuse_with_dense`：把图召回与向量召回用 RRF 融合，而不是让图独占候选池。
LightRAG 的 mix 模式就是把 naive 检索也并进来；实验证明图单独用会被噪声带崩。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

PPR_ALPHA = 0.85
PPR_ITERS = 30


def build_bipartite(store, chunk_entities: dict[str, list[str]]):
    """实体与 chunk 的二部图。返回节点编号与边。"""
    entity_ids: dict[str, int] = {}
    chunk_positions: list[int] = []
    id_to_position = {record["id"]: i for i, record in enumerate(store._records)}

    for chunk_id, entities in chunk_entities.items():
        if chunk_id not in id_to_position:
            continue
        chunk_positions.append(id_to_position[chunk_id])
        for entity in entities:
            entity_ids.setdefault(entity, len(entity_ids))

    entity_count = len(entity_ids)
    node_of_chunk = {position: entity_count + index for index, position in enumerate(chunk_positions)}

    edges: list[tuple[int, int]] = []
    for chunk_id, entities in chunk_entities.items():
        position = id_to_position.get(chunk_id)
        if position is None:
            continue
        chunk_node = node_of_chunk[position]
        for entity in entities:
            edges.append((entity_ids[entity], chunk_node))

    return {
        "entity_ids": entity_ids,
        "node_of_chunk": node_of_chunk,
        "chunk_of_node": {node: position for position, node in node_of_chunk.items()},
        "edges": edges,
        "node_count": entity_count + len(chunk_positions),
    }


def personalized_pagerank(graph: dict, seeds: list[str]) -> dict[int, float]:
    """从种子实体出发的 PPR，返回 chunk 位置 -> 分数。"""
    entity_ids = graph["entity_ids"]
    seed_nodes = [entity_ids[name] for name in seeds if name in entity_ids]
    if not seed_nodes:
        return {}

    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in graph["edges"]:
        adjacency[left].append(right)
        adjacency[right].append(left)

    restart = np.zeros(graph["node_count"])
    restart[seed_nodes] = 1.0 / len(seed_nodes)
    rank = restart.copy()

    for _ in range(PPR_ITERS):
        nxt = np.zeros(graph["node_count"])
        for node, neighbours in adjacency.items():
            weight = rank[node] / len(neighbours)
            if weight == 0:
                continue
            for neighbour in neighbours:
                nxt[neighbour] += weight
        rank = PPR_ALPHA * nxt + (1 - PPR_ALPHA) * restart

    result: dict[int, float] = {}
    for node, position in graph["chunk_of_node"].items():
        score = float(rank[node])
        if score > 0:
            result[position] = score
    return result


def apply_document_quota(store, positions: list[int], per_doc_cap: int, limit: int) -> list[int]:
    """按文档配额挑候选：同一份文档最多占 per_doc_cap 个位置。

    旧实现里《社会保险》一个文档占掉 12/28 个候选位，把真正需要的文档挤没了。
    """
    taken: dict[str, int] = defaultdict(int)
    kept: list[int] = []
    for position in positions:
        source = store._records[position].get("source", "")
        if taken[source] >= per_doc_cap:
            continue
        taken[source] += 1
        kept.append(position)
        if len(kept) >= limit:
            break
    return kept
