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
    # 精排批大小：0 = 按设备自动（CPU 用 2、GPU 用 8）。
    # **结论在两种设备上是相反的**：
    #   CPU（i5-13420H）：批量越大越慢——每对 batch=1/2 约 0.6s、8 约 0.85s、16 约 1.4s
    #   GPU（RTX 3050）：批量越大越快——8 对 batch=8 约 0.20s；
    #     实测并发 4 时 batch=8 达 0.90 QPS，batch=2 只有 0.41 QPS
    # 所以默认值不能是一个常数，见 rerank.effective_batch_size()。
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "0"))
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
    # 向量后端：json（默认，全量点积，零依赖）| milvus（可选，见 README 实测对比）
    # 实测（scripts/bench_backends.py）：126 chunk 下两者都是毫秒级，
    # 而查询编码占 22~25ms、精排占 12~28s——存储不是瓶颈；
    # 2,000/20,000 条时 milvus-lite 的检索开销约为全量点积的 48 倍（无 ANN 索引）。
    # 因此默认保持 json，Milvus 作为可切换后端保留，理由见 README「向量后端」一节。
    vector_backend: str = os.getenv("VECTOR_BACKEND", "json")
    # Milvus 连接：本地文件即 Milvus Lite；填 http://host:19530 则连独立服务
    milvus_uri: str = os.getenv("MILVUS_URI", "milvus.db")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "kb_chunks")
    milvus_metric: str = os.getenv("MILVUS_METRIC", "COSINE")
    # ---- 准入控制（admission.py）----
    # 实测：1 进程 × 8 线程冷路径吞吐约 0.38 req/s、单请求 p50 约 9.5s。
    # 并发在这里不是产能而是排队，所以默认只放 workers=1 个请求进入"贵的那一段"（精排），
    # 其余在有界队列里等；队列满或等太久直接拒绝，而不是让所有人一起超时。
    admission_enabled: bool = _bool_env("ADMISSION_ENABLED", True)
    admission_workers: int = int(os.getenv("ADMISSION_WORKERS", "1"))
    admission_queue_max: int = int(os.getenv("ADMISSION_QUEUE_MAX", "4"))
    admission_queue_timeout: float = float(os.getenv("ADMISSION_QUEUE_TIMEOUT", "30"))
    # 令牌桶默认关闭：速率必须按目标机器实测容量倒推（本机 0.38 req/s、服务器可能 3 req/s），
    # 写死一个"看起来合理"的值（如 100/s）等于没做限流——它永远不会触发。
    admission_rate: float = float(os.getenv("ADMISSION_RATE", "0"))
    admission_burst: int = int(os.getenv("ADMISSION_BURST", "4"))
    # ---- 推理设备 ----
    # auto = 有 CUDA 就用 GPU，否则 CPU。CI（无 GPU）与笔记本共用同一份代码。
    # 注意：换设备后 logit 有 1e-3 级差异，而阈值 0.05 踩在边界上——必须重跑评测。
    rerank_device: str = os.getenv("RERANK_DEVICE", "auto")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "auto")
    # 精排用 fp16（仅 GPU 有效）：显存减半、通常更快，数值差异更大，需评测确认
    rerank_fp16: bool = _bool_env("RERANK_FP16", False)
    # ---- 成本记账 ----
    # 单价必须自己配：不同模型/不同时间价格不同，硬编码一定过时。
    # 默认 0 = 不输出成本指标——一个恒为 0 的成本比没有更危险，会让人以为不花钱。
    # 例：若输入 1 元/百万 token、输出 2 元/百万 token：
    #   LLM_PRICE_INPUT_PER_MILLION=1  LLM_PRICE_OUTPUT_PER_MILLION=2
    llm_price_input_per_million: float = float(os.getenv("LLM_PRICE_INPUT_PER_MILLION", "0"))
    llm_price_output_per_million: float = float(os.getenv("LLM_PRICE_OUTPUT_PER_MILLION", "0"))
    llm_price_currency: str = os.getenv("LLM_PRICE_CURRENCY", "CNY")
    # ---- 报销提案存储 ----
    # 空 = 用本地 SQLite（默认，行为与迁移前一致）；填 mysql://user:pass@host:port/db 则走 MySQL。
    # 迁移分期进行：第一期只切这一层，会话与 checkpoint 仍是 SQLite。
    claim_store_url: str = os.getenv("CLAIM_STORE_URL", "")

    def validate(self) -> None:
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("SCORE_THRESHOLD must be between 0 and 1")
        if self.auth_mode not in {"dev", "jwt"}:
            raise ValueError("AUTH_MODE must be 'dev' or 'jwt'")
        if self.vector_backend not in {"json", "milvus"}:
            raise ValueError("VECTOR_BACKEND must be 'json' or 'milvus'")
        for name in ("rerank_device", "embedding_device"):
            if getattr(self, name) not in {"auto", "cpu", "cuda"}:
                raise ValueError(f"{name.upper()} must be 'auto', 'cpu' or 'cuda'")
        if self.auth_mode == "jwt" and not self.jwt_secret:
            raise ValueError("JWT_SECRET is required when AUTH_MODE=jwt")
        if self.admission_workers < 1:
            raise ValueError("ADMISSION_WORKERS must be >= 1")
        if self.admission_queue_max < 0:
            raise ValueError("ADMISSION_QUEUE_MAX must be >= 0")
        if self.admission_queue_timeout <= 0:
            raise ValueError("ADMISSION_QUEUE_TIMEOUT must be positive")
        if self.admission_rate < 0:
            raise ValueError("ADMISSION_RATE must be >= 0（0 表示不限流）")
        if self.admission_burst < 1:
            raise ValueError("ADMISSION_BURST must be >= 1")
        if self.claim_store_url and not self.claim_store_url.startswith(
            ("mysql://", "mysql+pymysql://")
        ):
            raise ValueError("CLAIM_STORE_URL 只支持 mysql:// 或留空（留空=SQLite）")
        for name in ("max_tool_rounds", "table_max_rows", "tool_top_k_max", "tool_text_truncate",
                     "rerank_candidate_factor", "tool_default_top_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.upper()} must be positive")
        if self.rerank_batch_size < 0:
            raise ValueError("RERANK_BATCH_SIZE must be >= 0（0 表示按设备自动选择）")
        if self.tool_default_top_k > self.tool_top_k_max:
            raise ValueError("TOOL_DEFAULT_TOP_K must not exceed TOOL_TOP_K_MAX")


settings = Settings()
settings.validate()

