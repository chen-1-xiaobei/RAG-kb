"""测试会话的全局设置。

**把默认向量后端固定为 json**，原因有两条：

1. 单测必须**离线且确定**。Milvus 后端会在磁盘上起一个本地服务进程、
   并依赖 `milvus.db` 已经建好；让单测依赖它，等于把"有没有跑过 ingest"
   变成测试能否通过的前提。
2. CI 的测试任务**只装依赖、不建索引**。

即使将来默认值再变，这里也显式钉住，避免单测行为随环境漂移。

Milvus 后端本身由 `tests/test_vector_backend.py` 显式测试：它自己建临时索引，
所以在没装 pymilvus 的环境里自动跳过，不影响主套件。

注意：必须在这里设置，因为 `config.settings` 在**导入期**就读取了环境变量。
"""
import os

os.environ.setdefault("VECTOR_BACKEND", "json")
