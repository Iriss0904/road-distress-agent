# Road Distress Agent

面向道路病害诊断与维修建议的本地工作台。它将 LangGraph 诊断流程、混合检索、重排序、证据门控、可追溯引用和人工确认串联为一个可自行部署的应用。

本仓库不提供任何道路规范、PDF、文本、索引、向量数据、案例、评测数据或运行记录。使用者必须自行导入拥有合法使用权的资料。

## 架构概览

```text
用户描述/图片 → 诊断状态机 → 混合检索（稠密 + 稀疏） → rerank
                                  ↓
                         evidence gating / 引用锚点
                                  ↓
                  人工澄清或确认（HITL） → 维修建议与可追溯引用
```

后端保留节点级审计事件；LLM 与检索节点会写入运行时 trace。SQLite 只用于本地会话、项目和记忆数据，Qdrant 保存由你导入资料产生的向量索引。

## 前置条件

- Python 3.10–3.12
- Node.js 18+
- Docker 与 Docker Compose（用于 Qdrant）
- live 模式所选模型供应商的凭证

## 快速启动（空知识库）

```bash
git clone <your-public-repository-url>
cd road-distress-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[web,qdrant,dev]"
cp .env.example .env
docker compose up -d qdrant
cd frontend
npm ci
npm run build
cd ..
road-distress-web
```

打开 `http://127.0.0.1:8010`。默认 `.env` 使用 `ROAD_DISTRESS_RUN_MODE=dev`，因此空知识库也能启动工作台而不会发出模型调用。知识库为空时，不应将其产生的诊断文本视为规范依据；请先完成下方导入和索引步骤，再将模式改为 `live`。

前端开发模式可另开终端运行：

```bash
cd frontend
npm run dev
```

`VITE_API_TARGET` 可用于将开发代理指向其他后端地址。生产部署前请自行配置访问控制与 TLS。

## 配置

`.env.example` 只包含变量名与安全默认值。重点变量：

- `DATA_DIR`：本地运行数据根目录，默认 `./data`。
- `QDRANT_URL`、`QDRANT_API_KEY`、`QDRANT_COLLECTION`：检索服务和 collection。
- `DEEPSEEK_*`：文本模型配置；live 模式需要 `DEEPSEEK_API_KEY`。
- `DASHSCOPE_*`：可选图像模型配置；live 模式处理图片时需要其凭证。
- `ROAD_DISTRESS_NORM_DB`：可选的、你自行维护的造价定额 SQLite 文件；仓库不提供此数据。

不要提交 `.env`、原始资料、解析结果、数据库或 Qdrant volume。详见 [data/README.md](data/README.md) 与 [SECURITY.md](SECURITY.md)。

## 导入自有资料并构建索引

支持 Markdown、纯文本和 PDF。Markdown/文本的最小链路不依赖 LLM：

```bash
# 将自己有权使用的 .md 或 .txt 放入 data/raw/ 后执行
python scripts/import_documents.py data/raw --out data/processed

# 对每个输出文档构建 BGE-M3 稠密 + 稀疏向量索引
python scripts/build_qdrant_index.py data/processed/<document-id>
```

PDF 走 MinerU 解析与可选的上下文增强链路：

```bash
python -m pip install -e ".[ingestion,pass1,qdrant]"
python scripts/ingest_pdf_full.py --pdf data/raw/<your-file>.pdf --out data/processed
```

该 PDF 链路需要按 MinerU 文档安装适配本机硬件的 PyTorch；若启用了上下文增强，还需配置对应的模型服务。所有 chunks、JSONL、缓存和向量数据都只留在被 Git 忽略的本地数据目录。

资料导入完成后，在 `.env` 设置实际模型变量并切换：

```env
ROAD_DISTRESS_RUN_MODE=live
DEEPSEEK_API_KEY=your_key_here
```

重启 `road-distress-web`。若缺少 live 模式的必要变量，CLI 会明确报告缺失项；不要以空或未授权资料生成生产建议。

## 验证命令

```bash
python -m pytest
ruff check .
cd frontend && npm run test && npm run build
cd .. && docker compose config
curl --fail http://127.0.0.1:8010/api/profile
```

首次构建索引会下载 BGE-M3 模型，取决于网络与硬件。若 Qdrant 未启动或 collection 不存在，检索请求会公开报错原因；请启动服务并完成索引，而不要将错误结果当作无证据结论。

## 当前限制

- 知识质量、版权与适用性由部署者负责；系统不提供法规或工程合规保证。
- PDF 表格、扫描质量和复杂版式的解析质量取决于 MinerU 与输入材料。
- 造价建议要求部署者提供具有合法授权且模式兼容的定额数据库。
- 默认部署仅面向单机开发，未内置多租户权限控制或生产级密钥管理。

发布前请阅读 [PUBLIC_RELEASE_AUDIT.md](PUBLIC_RELEASE_AUDIT.md) 和 [LICENSE_PENDING.md](LICENSE_PENDING.md)。
