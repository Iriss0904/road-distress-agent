# 公开发布审计报告

审计日期：2026-07-27

## 1. 公开内容

保留内容均按白名单复制，并经过路径、内容和运行依赖审阅：

- `src/road_distress_agent/`：诊断状态机、LangGraph 编排、混合检索、重排序、证据门控、引用、HITL、API 与本地持久化实现；这些是工作台运行所需的应用代码。
- `src/road_distress_agent/prompts/`：通用节点提示模板。已去除对特定资料、特定地区和演示身份的引用。
- `frontend/`：Vue 工作台源码、依赖锁文件与一个不含业务数据的 UI smoke test。
- `scripts/`：Markdown/文本导入、PDF 解析、上下文处理与 Qdrant 索引构建脚本。`import_documents.py` 是新建的用户资料导入入口。
- `data/`：仅包含 README 和 `.gitkeep`，用于指示本地未跟踪数据目录。
- `.env.example`、`docker-compose.yml`、`pyproject.toml`、`.gitignore`、README 与 SECURITY 文档：安装、配置和部署所需的公开配置模板。

入口命令：

```bash
python -m pip install -e ".[web,qdrant,dev]"
docker compose up -d qdrant
cd frontend && npm ci && npm run build && cd ..
road-distress-web
```

导入入口：`python scripts/import_documents.py data/raw --out data/processed`。PDF 入口：`python scripts/ingest_pdf_full.py --pdf <user-file> --out data/processed`。

## 2. 排除内容

以下类别未复制到公开目录：

- 数据：原始规范、PDF、文本、解析结果、chunks、JSONL、metadata dump、Qdrant collection、embedding cache、BM25 索引和 SQLite 数据库。
- 文档：内部设计、实现计划、审计、项目复盘、实验报告、优化资料和访谈记录。
- 评测：gold set、评测样本、人工标注、RAGAS/DeepEval 类结果、baseline 与运行结果。
- 日志：运行 trace、日志、截图、录屏、导出 UI 数据和本地 PID 文件。
- Secrets：`.env`、真实 API Key、Token、密码、Cookie、连接串和私有地址。
- 构建产物：虚拟环境、`node_modules`、前端 `dist`、Python 缓存、测试/静态分析缓存。
- 其他：Qdrant 二进制及配置、内部演示身份/记忆、特定来源文档名和未确认可公开的素材。

排除的路径模式包括 `data/**`（仅保留三处 `.gitkeep` 与 README）、`docs/**`、`eval/**`、`eval_runs/**`、`logs/**`、`storage/**`、`snapshots/**`、`interview/**`、`.env*`（仅允许 `.env.example`）及所有数据库和原始文档扩展名。

## 3. 代码改造

- 移除了固定的开发机数据库路径、固定 Web 端口、打包 Qdrant 二进制启动逻辑、特定 collection 名称、特定资料名和内部演示用户/记忆。
- 新增或统一使用 `DATA_DIR`、`ROAD_DISTRESS_DOC_DIR`、`ROAD_DISTRESS_WEB_DB`、`ROAD_DISTRESS_MEMORY_DB`、`ROAD_DISTRESS_PROJECT_DB`、`ROAD_DISTRESS_DELIVERY_INDEX_DB`、`ROAD_DISTRESS_DELIVERY_DIR`、`ROAD_DISTRESS_NORM_DB`、`QDRANT_API_KEY` 与 Web host/port 配置。
- 默认 collection 改为通用的 `road_distress_documents`。无资料的开发模式可以启动，live 模式仍会对缺失的必要配置显式报错。
- 新增 `scripts/import_documents.py`，将用户自己的 Markdown/文本转换为 `rag_chunks.jsonl`；PDF 继续通过显式的 MinerU 导入链路处理。
- 造价定额数据库不再有内置默认文件；调用造价功能却未设置 `ROAD_DISTRESS_NORM_DB` 时会明确报错。

## 4. Git 历史检查

结论：原开发仓库历史不能公开。

- 历史中发现过原始 PDF、Qdrant collection/WAL/向量段、Qdrant 二进制、Web 数据库备份、评测/设计资产和 `.env*` 提交。
- 对历史 Git blob 进行了长度受限的常见 Key 模式扫描，未发现匹配 `sk-`、`lsv2_` 或 `tvly-` 非占位符字面量的结果；该扫描不能替代发布者的密钥轮换和人工复核。
- 本目录由 `git init -b main` 在独立目录创建，未复制原仓库 `.git`；提交 `35a6a47` 是无父提交的全新历史。
- 因此公开版本不会携带原开发仓库的提交对象、文件内容或历史引用。

## 5. 测试结果

已实际执行并通过：

```bash
python -m compileall -q src scripts tests
timeout 60s python -m pytest                 # 2 passed
ruff check .                                  # All checks passed
cd frontend && npm ci && npm run test         # 1 passed
cd frontend && npm run build                  # passed
docker compose config                         # passed
python scripts/import_documents.py <synthetic-temp-input> --out <synthetic-temp-output>
python scripts/build_qdrant_index.py --help
python scripts/ingest_pdf_full.py --help
PYTHONPATH=src ROAD_DISTRESS_RUN_MODE=dev DATA_DIR=<temp> \
  ROAD_DISTRESS_WEB_PORT=18010 python -m road_distress_agent.api.server
curl --fail http://127.0.0.1:18010/api/profile # dev 空知识库启动通过
```

静态扫描结果：未发现公开目录中的凭证模式、个人绝对路径、PDF/DOCX/Parquet/Pickle/SQLite 文件或大于 1 MiB 的文件；`.gitignore` 已覆盖本地数据、构建产物和密钥文件。

未验证项：未在干净新虚拟环境中重新下载 Python 依赖；未运行真实 PDF/MinerU 解析、BGE-M3 模型下载、真实 Qdrant upsert 或 live 模型调用，以避免引入用户资料、下载缓存或供应商费用。前端 `npm ci` 报告 7 个依赖审计问题（3 moderate、3 high、1 critical）；未执行自动升级，需发布者审查锁文件与升级策略。

## 6. 发布前人工检查项

- 选择并加入正式开源 `LICENSE`；当前只有 `LICENSE_PENDING.md`。
- 确认项目名称、GitHub 描述、README 截图和是否公开具体架构/性能指标。
- 审核源代码及贡献者对代码的版权授权，并完成 Python/Node 第三方依赖许可证与安全审查。
- 选择并验证模型供应商、模型名称、数据保留政策与费用控制。
- 用自有且具有合法授权的资料执行一次真实导入、索引和检索验收。
- 为生产部署配置身份认证、授权、TLS、网络隔离、Qdrant 访问控制、日志脱敏和备份。
- 审核计划添加的 GitHub Actions，确保不会上传数据、日志、缓存或 secrets。
- 复查 `git status --ignored` 与 `git diff --cached --name-only`，确认 staged 文件只包含本报告列出的公开内容。
