# 企业知识库 Agentic RAG 服务镜像（CPU 版）
#
# 构建：docker build -t rag-kb .
# 运行：见 docker-compose.yml，或
#       docker run --rm -p 8000:8000 -e DEEPSEEK_API_KEY=... -v rag-kb-cache:/app/.cache rag-kb
#
# 注意：embedding 与 reranker 模型（合计约 2GB+）在首次运行时下载到 /app/.cache，
# 该目录已声明为 volume，重建容器不会丢失；数据目录 data/ 与索引建议用挂载覆盖。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=0

WORKDIR /app

# 先只拷贝依赖清单：代码改动不会让依赖层缓存失效
COPY requirements.txt ./

# CPU 版 torch 单独装，避免拖入数 GB 的 CUDA 依赖；再装其余依赖
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/.cache

EXPOSE 8000

# 以非 root 运行；挂载目录需可写，故显式赋权
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查：索引缺失时 /health 返回 ok=false，容器会被标记为 unhealthy，
# 这比"进程还活着"更能反映真实可用性
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); raise SystemExit(0 if d.get('ok') else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
