把 `.md`、`.txt`、`.pdf`、`.docx` 或 `.html` 文件放到此目录，然后运行 `python ingest.py`。

权限约定：
- 放在 `data/` 根目录的文件对所有人可见（group=public）
- 放在子文件夹（如 `data/财务/`）的文件仅对声明了该组的用户可见
- 调用 API 时通过 `X-Dev-User` / `X-Dev-Groups` 请求头声明身份（`AUTH_MODE=dev`）；生产环境设 `AUTH_MODE=jwt`，权限从 token 的 `sub/groups` claims 读取
- 本目录的 `README.md` 不会被索引（不在 loader 支持格式内）

新增或改名文档后，记得同步 `evals/dataset.json` 的来源注册表——`tests/test_eval_dataset.py` 会检查两者一致。
