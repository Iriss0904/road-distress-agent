# 本地数据目录

本仓库不提供道路规范、用户资料、向量索引或任何运行数据。请只放入你拥有合法使用权的资料；这些目录均不会被 Git 跟踪。

- `raw/`：放入原始 Markdown、文本或 PDF。
- `processed/`：导入与解析生成的 chunks/JSONL，只保留在本机。
- `indexes/`：可选的本地索引中间产物。

导入 Markdown 或文本：

```bash
python scripts/import_documents.py data/raw --out data/processed
```

对每一个 `data/processed/<document-id>/rag_chunks.jsonl` 构建索引：

```bash
python scripts/build_qdrant_index.py data/processed/<document-id>
```

PDF 请使用 `scripts/ingest_pdf_full.py --pdf data/raw/<file>.pdf --out data/processed`，并先安装 `.[ingestion,pass1,qdrant]`。该链路的所有解析结果仍只保存在未跟踪的 `processed/` 目录。
