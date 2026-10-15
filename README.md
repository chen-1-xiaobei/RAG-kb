# 企业知识库问答系统（RAG + LangGraph Agent）

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/tests-74%20passed-brightgreen)](#测试与评测)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> 面向企业内部制度与流程问答的 Agentic RAG 系统。RAG 只是它可调用的工具之一：模型自主决定
> **检索文档** 还是 **查询结构化表格**，而答案在任何情况下都只能来自真实工具证据。

**三个设计取向**

- **拒答是控制流，不是提示词。** 零工具证据一律拒答且不调用 LLM，模型自编的答案结构上无处安放。
- **权限不在模型可写的通道上。** 工具按请求构造并绑定身份闭包，模型只能提供查询参数；检索层按组预过滤。
- **性能靠测量，不靠直觉。** 精排 2.2x 与缓存 8517ms→0.3ms 都来自实测，且包含一个反直觉结论（见下）。

## 架构

```mermaid
flowchart TB
    C[客户端] -->|X-Dev-User / X-Dev-Groups 或 JWT| P[身份解析 Principal]
    P --> A1["POST /ask<br/>固定管线"]
    P --> A2["POST /agent/ask<br/>Agentic RAG"]

    subgraph RAG["两段式检索（/ask 与检索工具共用）"]
        direction LR
        V["向量粗排<br/>group 预过滤<br/>20~60ms"] --> RR["cross-encoder 精排<br/>原始 logit + 阈值<br/>约 4.5s"]
        RC[("逐对 logit 缓存<br/>LRU 2 万条")] -.->|热命中 0.3ms| RR
    end

    A1 --> V
    A2 --> AG["agent 节点<br/>bind_tools 决策"]
    AG -->|tool_calls 且未超轮数| TL["tools 节点"]
    AG -->|无调用 / 轮数耗尽| ANS
    TL --> T1["search_knowledge_base<br/>身份闭包注入"]
    TL --> T2["list_tables / query_table<br/>三层只读 SQL"]
    T1 --> V
    T2 --> DB[("内存 sqlite<br/>SELECT-only + authorizer")]
    TL -->|ToolMessage| AG
    ANS["answer 节点<br/>证据门槛"] -->|证据为空| REF["拒答（不调用 LLM）"]
    ANS -->|有证据| GEN["生成答案<br/>citation 服务端校验"]
    GEN --> C
    REF --> C

    classDef gate fill:#ffe8e8,stroke:#c00,stroke-width:2px
    classDef cache fill:#e8f4ff,stroke:#06c,stroke-dasharray: 4 3
    class ANS,REF gate
    class RC cache
```

## 运行

需要 **Python 3.12+**（开发环境为 3.14.6）。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 版 torch，避免数 GB CUDA 依赖
pip install -r requirements.txt
Copy-Item .env.example .env    # .env 已存在则跳过
# 编辑 .env，填入 DEEPSEEK_API_KEY（仅对话需要；embedding 用本地模型，无需 key）
python ingest.py               # 首次运行会下载 bge-small-zh-v1.5 模型，建立 index.json
uvicorn app:app --reload
```

打开 <http://127.0.0.1:8000/docs> 交互调试。两个端点都返回 `answer` / `sources` / `insufficient_context`：

```powershell
# 固定管线：retrieve -> rerank -> answer
curl.exe -s http://127.0.0.1:8000/ask -H "Content-Type: application/json" `
  -H "X-Dev-Groups: %E8%B4%A2%E5%8A%A1" `
  -d '{\"question\":\"报销单的审批流程是什么？\",\"top_k\":4}'

# Agentic RAG：模型自行决定检索文档还是查表（同步接口，纯 CPU 精排约 5s/次，请耐心等待）
curl.exe -s http://127.0.0.1:8000/agent/ask -H "Content-Type: application/json" `
  -H "X-Dev-Groups: %E8%B4%A2%E5%8A%A1" `
  -d '{\"question\":\"出差住宿一晚最多能报多少钱？\"}'
```

`X-Dev-Groups` 只能传 ASCII，中文组名需 URL 编码（`财务` = `%E8%B4%A2%E5%8A%A1`）；开发模式通过
`X-Dev-User` / `X-Dev-Groups` 请求头模拟已验证身份，生产环境设置 `AUTH_MODE=jwt`，权限从 JWT 的
`sub/groups` claims 获取，**客户端不能在请求体中声明权限**。

示例响应（截断）：

```json
{
  "answer": "员工需在费用发生后三十天内在 OA 系统填写报销单并上传发票原件照片；报销单经直属主管审批后流转至财务部审核，审核通过后三个工作日内打款至员工工资卡。",
  "sources": [
    {
      "citation": 1,
      "source": "报销流程.md",
      "relative_path": "财务/报销流程.md",
      "folder": "财务",
      "group": "财务",
      "title": "费用报销流程（财务部）",
      "section": "报销步骤",
      "page": null,
      "para": 1,
      "score": 4.8213
    }
  ],
  "insufficient_context": false
}
```

> 提示：无法访问 huggingface.co 时，在 `.env` 设置 `HF_HUB_OFFLINE=1` 使用本地缓存的模型（首次下载模型时需要先注释掉该行）。
> 提示：`CHAT_MODEL` 必须用支持 tool calling 的模型（如 `deepseek-chat`）；thinking 模型（如 `deepseek-v4-pro`）不支持结构化输出，会报 400。

`POST /ask` 是固定管线（retrieve → rerank → answer），`POST /agent/ask` 是 Agentic RAG：LangGraph 循环 `agent -> tools -> agent ... -> answer`，模型自主决定调用 `search_knowledge_base` 检索或 `list_tables`/`query_table` 查询 CSV 表格。**答案节点的证据门槛**保证零工具证据一律拒答——模型不调工具直接作答、工具零命中、工具报错、轮数耗尽（`MAX_TOOL_ROUNDS`）都会被拦截，模型自编的答案不会被采纳。

## 文档与权限

- 把 `.txt`、`.md`、`.pdf` 或 `.docx` 放入 `data/`，运行 `python ingest.py` 建索引（无需重启服务）。
- **文档标准化层**（`loaders.py`）：所有格式先统一转为 markdown 再进切分管道——PDF 走 pymupdf4llm 版面分析（标题/表格/阅读顺序保真，布局模型随 wheel 打包、无需联网），docx 走 mammoth；转换产物按源文件内容哈希 + 转换版本号缓存于 `.cache/`，源文件变更或 loader 升级自动失效；转换失败降级 pypdf 纯文本。
- 文档按段落切分（短段落合并至 700 字，绝不跨段截断），每块带 `title`（文档标题）、`section`（所属章节）、`para`（段块序号）、`folder`、`group` 等标识字段。
- 权限约定：`data/` 根目录文件 → `group=public`；子目录名为权限组。服务端身份解析后按组**预过滤**，未授权内容不参与 top-k 竞争。
- **结构化表格数据**：把 `.csv` 放入 `data/`（组权限约定与文档相同），服务启动时载入内存 sqlite（只读），`/agent/ask` 的模型通过 `list_tables`/`query_table` 工具查询——表格与数值查询是向量检索的短板，交给 SQL 补。查询白名单校验 + SELECT-only + sqlite authorizer 三层只读；未授权表与不存在的表返回同一文案（不泄露存在性）。改 CSV 后重启服务生效。

## 面试可讲的工程点

- **两段式检索**：bi-encoder 向量粗排（高召回，O(N) 点积毫秒级）→ cross-encoder reranker 精排（高精度，query 与文档全注意力交互，仅跑在少量候选上）；精排分数为原始 logit（0≈中性，阈值 0.05 由评测集标定——sigmoid 会把中性分数压缩到 0.5 附近丢失判别度），低于阈值直接丢弃并拒答。
- **精排延迟优化（实测驱动）**：链路日志显示精排占单次请求 70%+ 耗时（粗排 20~60ms vs 精排 10s+），做了三处改动——① 候选数 `top_k×4`（封顶 16）→ `top_k×2`，每减一条候选线性省时间，`evaluate.py` 验证 source_recall / answerability 仍 100%；② 批大小改为可配（`RERANK_BATCH_SIZE`，默认 2）。**反直觉发现：CPU 上批量越大反而越慢**（本机 i5-13420H 实测 16 对：batch=1/2 约 9.6s、8 约 13.6s、16 约 22.5s，用交替测量排除热漂移，反向顺序复现同一单调曲线）——大批量触发填充与降频。合计精排 10.0s → 4.5s（**2.2x**，其中候选数贡献 2.1x、批大小约 5%~10%）；剩余耗时是 568M 模型在纯 CPU 上的物理成本，再往下要动量化或换更小的 reranker（需重新标定阈值）。
- **精排逐对缓存（针对 agent 重试路径）**：agent 循环里"换个关键词再试"是提示词明确鼓励的行为，但每一轮都重排一批高度重叠的候选块——同一 `(question, doc)` 对会被反复送进模型。改为按 `(模型, question, doc_id, 文档内容哈希)` 缓存原始 logit，LRU 上限 2 万条（约 10MB）：**冷 8517ms → 热 0.3ms**（本机 8 对，`rerank_cache pairs=8 hits=8 misses=0`）；离线评测中命中率约 3/8（`hits=5 misses=3`），真实多轮重试场景更高。内容哈希参与键，索引重建后自动失效；只缓存 logit 这一纯计算结果、不缓存 top-N/阈值等策略，因此调阈值无需清缓存。
- **检索成本不交给模型决定**：精排耗时与候选数线性相关（候选 = `top_k × RERANK_CANDIDATE_FACTOR`），而 `top_k` 原本是模型可自由填写的工具参数——填 8 就是默认值 4 的两倍开销。现改为服务端封顶（`TOOL_DEFAULT_TOP_K`，默认 4）：模型**只能调小、不能调大**，工具描述也明说"调大不会有更多结果，调小可加快响应"。同时对参数格式容错（模型给出非整数时回落默认值，不因参数格式打断循环）。
- **单请求内检索去重**：同一 query 重复调用（模型可能原地重试）直接复用结果，键基于**夹取后**的参数（`top_k=8` 与 `top_k=20` 夹成同一查询应视为同一键）。memo 只存在单次请求的 state 里、不跨用户共享，因此不引入权限泄露面；表格工具刻意不参与去重，避免误用陈旧行数据。
- **文档标准化中间层**：所有格式先归一化为 markdown 再进切分管道，下游只懂一种格式；loader 按格式可插拔、按内容哈希 + 转换版本缓存（源文件变更或 loader 升级自动失效）、转换失败降级纯文本提取。
- LangGraph Agentic RAG：`agent（bind_tools 决策）-> tools -> agent ... -> answer` 循环，条件边 + 轮数上限强制收敛；节点/路由全部注入式，单测注入 stub 不碰模型。配套两个结构性保证：**证据门槛**（零工具证据直接拒答且不调用 LLM，闭卷不依赖提示词自觉）与**工具身份闭包注入**（工具由工厂按请求构造并绑定 Principal，模型只能提供 query/top_k/表名/SQL 等查询参数，权限不在模型可写的通道上；top_k 服务端夹取）。
- **模型可见材料与答案证据解耦**：工具返回 `(text, artifact)`——给模型的是截断后的材料文本、不带 citation 编号；给答案节点的是完整证据。citation 编号只在答案节点组装时统一赋值，检索命中与表格行混合编号不冲突；表格行转伪 Document，Source 模型/citation 校验/拒答逻辑零改动复用。
- 表格工具安全三层：表名白名单 + SELECT-only 语法校验（单语句、禁注释与写关键字、FROM 必须等于白名单表）+ sqlite `set_authorizer` 数据库层授权（只放行对白名单表的 READ，语法被绕过也读不到其他表）；行数上限截断；工具异常/未知工具降级为 error ToolMessage 回给模型，循环不中断。
- 提示词分离：agent 节点管"调什么工具收集资料"（规划），答案节点管"只能依据已收集证据作答"（生成），两条提示词各司其职。
- **链路可观测**：两个端点各自输出结构化链路日志（`rag_query_start/done`、`agent_query_start/done`、`agent_decision`、`tool_call/tool_result`、`answer_gate`、`retrieve`、`answer_generate`），带毫秒时间戳与每步耗时（粗排/精排/LLM 决策/工具执行/生成分别计时），单看日志即可复盘一次请求的决策路径与耗时分布；问题文本只记前缀（`LOG_QUESTION_CHARS`，折叠空白防多行污染），不把用户原文写入日志。
- 基于 LangChain 1.0 组件化搭建：HuggingFaceEmbeddings（bge 本地模型，文档/查询非对称检索指令）、自实现 JSONVectorStore（实现 LangChain VectorStore 接口，归一化向量余弦排序 + score_threshold 过滤 + filter 预过滤）。
- 段落级切分：段落边界 = 语义边界，短段落合并至目标长度、绝不跨段截断（因此天然无需字符 overlap）；超长单段落用 RecursiveCharacterTextSplitter 兜底；每块携带 title/section/para 等标识字段。
- 权限控制：文件夹名 = 权限组（约定优于配置），检索层按 group **预过滤**（未授权切片不参与 top-k 竞争，比检索后过滤更正确），无权限内容不泄露存在性。
- 查询链路用 LCEL 组合（ChatPromptTemplate | ChatDeepSeek with_structured_output），system/user 分离 + prompt 注入防御；服务端校验 citation，返回 source/score/folder/group/page 支持引用追溯。
- 启动时 lifespan 预热 embedding 与 reranker 模型，避免首个请求慢且分数不稳。
- 索引采用 schema 版本、稳定内容哈希 ID、PDF 页码和原子替换；启动加载并按 mtime 热更新。
- `tests/` 提供离线单元测试（含 agent 图路由、工具白名单/组过滤/注入防护），`evals/dataset.json` 与 `evaluate.py` 提供检索、权限负例、拒答与注入对抗的评测闭环（含精排管线）。

项目指标必须在本地评测后填写，不要直接虚构。

## 测试与评测

```powershell
python -m pytest -q
python evaluate.py                  # 评测不达标或相对基线回退时 exit 1
python evaluate.py --update-baseline  # 人工确认指标后刷新 evals/baseline.json
```

### 评测集构成（39 条，`evals/dataset.json`）

| 类别 | 条数 | 检验什么 |
| --- | --- | --- |
| 常规检索（含口语化改写） | 28 | 粗排 + 精排能否召回正确来源；改写题刻意避开文档原词，避免把评测做成关键词匹配 |
| 表格问答（工具层） | 6 | 结构化 CSV 不进向量索引，由 `query_table` 的 SQL 通路回答，并断言结果行确实含目标单元格 |
| 权限负例 | 2 | 无权限组一条都不能召回，也不能因拒答文案泄露受限文档的存在性 |
| 无关问题 | 2 | 知识库确无答案时必须零召回（精排阈值真的在起作用） |
| 注入对抗 | 2 | 检索结果正文不得携带注入标记；跨权限注入不得因注入而放行受控内容 |

> 表格用例是这次评测扩展中**被实测纠正**的一处口径错误：最初 6 条表格问题被写成了向量检索用例，跑出来 `hits=0`、`source_recall` 只有 82%。根因不是检索退步，而是 CSV 只以 sqlite 表形式存在、对向量索引根本不可见——用检索指标评它等于出题超出能力边界。现改为走 `query_table` 工具层，单列 `table_tool_accuracy`。

### 实测指标（39 条，本机 CPU）

| 指标 | 值 |
| --- | --- |
| source_recall / hit_rate | 100% |
| mrr / ndcg@10 | 100% |
| table_tool_accuracy | 100% |
| answerability_accuracy | 100% |
| permission_denial_accuracy | 100% |
| injection_resistance | 100% |

**必须说明这个 100% 不能当成泛化能力**：知识库只有 5 个来源、语料极小，且多轮迭代后评测集与文档高度贴合，指标已饱和、失去区分度（改动前后都是满分，看不出退步）。它目前的价值是**回归门禁**（`evals/baseline.json` + 2% 容差，能立刻发现召回塌陷、权限漏放、阈值失效），而不是能力度量。要作为能力证明，需要外部真实语料 + 人工标注相关性、并补一层答案级评测（citation 是否指向支撑句、拒答率）。这一点没有藏起来，正是下一步该做的事。

数据集自带**来源注册表**（文件名 -> 权限组/类型/入选理由），`tests/test_eval_dataset.py` 会断言：注册表与 `data/` 实际文件双向一致、**必须来源与分级来源都必须对用例声明的权限组可见**、负例不得声明必须来源、每个被索引来源都被至少一条用例覆盖。文档改权限组或改名而忘了改用例，测试直接红——评测集是会长毛的资产，这类断言比指标本身更值钱。

### 指标口径

- `source_recall` / `hit_rate`：来源级集合召回，由 `required_sources` 驱动。单条用例要求召回**全部**必须来源，避免多来源题被"召回到一个就算过"稀释。
- `mrr`：第一个相关来源排名的倒数——检索结果排第 1 还是第 4，对下游生成的影响完全不同，集合指标看不见。
- `ndcg@10`：排序质量，由 `relevant_sources` 驱动（主来源 2 / 次来源 1），单来源用例退化为二值。**排序相关性不等于召回门槛**：次要来源可以确实相关、但缺了它答案依然完整，所以 `relevant_sources` 允许包含不在 `required_sources` 里的来源（例如"住宿限额"以标准表为准、流程文档只是背景）。排第 1 得 1.0，排第 2 约 0.63，掉出前 10 得 0。
- `answerability_accuracy`：该答的答、该拒的拒。
- `permission_denial_accuracy` / `injection_resistance`：硬保证，门槛设为 100%，不接受"大多数情况下安全"。

评测只跑粗排 + 精排，不调用 LLM，因此可在 CI 中离线重复运行；`--update-baseline` 生成的 `evals/baseline.json` 带 2% 容差，用于捕捉"某次改动让 MRR 悄悄掉了"这类回归。

`index.json` 是生成物，已加入 `.gitignore`。更换 embedding 模型后必须重新执行 `python ingest.py`；跑评测前同样需要先建好索引。

## 已知边界与下一步

这个项目是**单实例、单轮对话、只读**的形态，边界是主动划定的，不是遗漏：

- **对话是单轮的**：`AskRequest` 不含会话状态，每个请求独立。多轮对话需要引入会话状态模型，并解决三个新问题——会话归属校验（会话 ID 是客户端传入的新权限入口）、每轮重新检索以免权限判定退化为"首次检索时判定一次"、以及避免检索到的注入内容随历史跨轮驻留。
- **知识库规模上限**：`JSONVectorStore` 每次查询做全量点积，面向万级 chunk 以内；再往上需要 FAISS/pgvector 之类带索引的实现。
- **纯 CPU 精排约 4.5s/次**：这是 568M cross-encoder 的物理成本。要降只能量化或换更小的 reranker，且**必须重新标定阈值**（换模型后旧阈值不再有效）。
- **缺少部署与协作配套**：无 Docker、无 CI。测试与评测都是纯 CPU 离线可跑的，接 CI 的成本很低。
- **评测集偏小**：39 条用例、5 个来源，指标已饱和。详见下方「实测指标」中的说明。

## 许可

[MIT](LICENSE)

