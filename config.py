import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    chat_model: str = os.getenv("CHAT_MODEL", "deepseek-chat")
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.25"))
    auth_mode: str = os.getenv("AUTH_MODE", "dev")
    jwt_secret: str | None = os.getenv("JWT_SECRET")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    enable_agent: bool = _bool_env("ENABLE_AGENT", True)
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    # 精排门槛（cross-encoder 原始 logit；0 为中性/无关，评测集标定值见 rerank.py 说明）
    rerank_threshold: float = float(os.getenv("RERANK_THRESHOLD", "0.05"))
    # 粗排召回候选数 = top_k x 倍数（封顶 16，控制 CPU 精排耗时）
    # 精排是整条链路的耗时大头，倍数越大线性越慢。
    # 实测（126 chunk 语料）：倍数 2（8 候选）source_recall 100%；
    # 降到 1（4 候选）时「没休完能折现吗」被同领域干扰文档挤出候选集，掉到 97.4%——
    # 精排只能重排候选、不能凭空召回，所以候选数是**召回上限**，不要轻易调小。
    rerank_candidate_factor: int = int(os.getenv("RERANK_CANDIDATE_FACTOR", "2"))
    # 混合检索：向量召回 + BM25 关键词召回，RRF 融合（实现见 hybrid.py）。
    # 实测结论（126 chunk、49 条用例）：开启与关闭的 source_recall / nDCG 完全相同（都是 100%）；
    # 代价是粗排 +1.25ms（21.4 -> 22.6ms，仅为精排 3.5s 的 0.04%）。
    # **没有可测量的收益，因此默认关闭**，代码保留为开关：
    # 语料显著变大（数千 chunk）或换更弱的精排模型时，值得用同一套评测重测一次。
    hybrid_enabled: bool = os.getenv("HYBRID_ENABLED", "0") not in {"0", "false", "False"}
    # 精排批大小：实测本机（i5-13420H 纯 CPU）批量越大反而越慢——每对耗时
    # batch=1/2 约 0.6s、8 约 0.85s、16 约 1.4s；故默认小批量。
    # 换硬件/换模型后建议重测（方法见 README「精排延迟」段）。
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "2"))
    # Agent 工具循环：最多执行工具的轮数，超过强制进答案节点（无证据则拒答）
    max_tool_rounds: int = int(os.getenv("MAX_TOOL_ROUNDS", "3"))
    # 表格工具单次返回行数上限（超出截断并提示）
    table_max_rows: int = int(os.getenv("TABLE_MAX_ROWS", "50"))
    # 检索工具的 top_k 服务端夹取上限（与 AskRequest.top_k 的 le=8 一致）
    tool_top_k_max: int = int(os.getenv("TOOL_TOP_K_MAX", "8"))
    # 检索工具默认条数：模型不显式指定时用它，同时作为"模型指定值"的封顶。
    # 理由：精排耗时与候选数线性相关（候选 = top_k x 倍数），若放任模型填 8，
    # 单次检索成本就是默认值的两倍；成本不该由模型随手填的参数决定。
    tool_default_top_k: int = int(os.getenv("TOOL_DEFAULT_TOP_K", "4"))
    # 工具给模型看的每条资料截断字符数（答案节点的证据不受影响）
    tool_text_truncate: int = int(os.getenv("TOOL_TEXT_TRUNCATE", "500"))

    def validate(self) -> None:
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("SCORE_THRESHOLD must be between 0 and 1")
        if self.auth_mode not in {"dev", "jwt"}:
            raise ValueError("AUTH_MODE must be 'dev' or 'jwt'")
        if self.auth_mode == "jwt" and not self.jwt_secret:
            raise ValueError("JWT_SECRET is required when AUTH_MODE=jwt")
        for name in ("max_tool_rounds", "table_max_rows", "tool_top_k_max", "tool_text_truncate",
                     "rerank_candidate_factor", "rerank_batch_size", "tool_default_top_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.upper()} must be positive")
        if self.tool_default_top_k > self.tool_top_k_max:
            raise ValueError("TOOL_DEFAULT_TOP_K must not exceed TOOL_TOP_K_MAX")


settings = Settings()
settings.validate()

