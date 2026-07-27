"""LLM contextual retrieval for raw-standard chunks."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from road_distress_agent.ingestion.contextual_io import (
    done_ids,
    group_totals,
    pending_chunks,
    sort_rag_chunks,
    stage_paths,
)
from road_distress_agent.ingestion.jsonl import append_jsonl, read_jsonl
from road_distress_agent.llm_deepseek import (
    contextual_base_url,
    contextual_model,
    deepseek_api_key,
    disabled_thinking,
)

load_dotenv()

SYSTEM_PROMPT = """\
你是一个中文技术文档分析助手，任务是为道路养护规范的某个段落生成"上下文前缀"和"语义角色列表"。
输出格式：
必须调用 emit_context 工具，不要输出额外文本。
semantic_role 是一个 JSON 数组，包含该段落所有适用的语义角色（1-4个）：
- "disease_definition"   ← 段落内容含病害类型定义、判定标准（如典型病害表、病害描述条款）
- "method_selection"     ← 段落内容含养护/维修方法适用条件、方法选择依据
- "construction_step"    ← 段落内容含具体施工操作步骤（某步的操作内容）
- "acceptance_criteria"  ← 段落内容含验收标准、质量检查要求、检测指标
- "general_info"         ← 总则、适用范围、术语、材料规格、引用条款等通用信息
重要规则：
- 只要段落包含某类信息就加入该角色，一个段落可同时有多个角色
- 仅当段落完全不包含上述前四类信息时，才使用 "general_info"（单独出现）
- 若提供了【步骤信息】，则必须包含 "construction_step"
OCR 与数值可靠性规则：
- 段落内容可能包含 OCR 噪声，尤其是数字、单位、图表号、条文号、英文术语和标点
- 遇到疑似 OCR 噪声时，不要在 context_prefix 中复述可疑数字、单位、图号或乱码
- 不要自行更正或编造具体数值；只能概括其稳定语义，例如"限速取值要求"、
  "警告区长度表"或"作业控制区布置示例"
- 若 rawtext 与 heading_path 或条文号存在轻微格式冲突，优先依据 heading_path
  和条文位置理解段落主题
- 表格 chunk 只概括表格用途，不摘抄具体单元格数值
context_prefix 要求：
- 1-2句中文自然语言，描述该段落在整篇文档中的语义角色（可反映多角色）
- 必须覆盖（如适用）：①路面材质 ②病害类型 ③语义角色
- 若提供了【步骤信息】，必须在 context_prefix 中注明（例如"共X步中的第Y步"）
- 不超过60字，不复述原文，不加主观评价\
"""

VALID_ROLES = frozenset(
    {
        "disease_definition",
        "method_selection",
        "construction_step",
        "acceptance_criteria",
        "general_info",
    }
)
CONTEXT_TOOL_NAME = "emit_context"
CONTEXT_TOOL = {
    "name": CONTEXT_TOOL_NAME,
    "description": "返回当前规范段落的上下文前缀和语义角色。",
    "input_schema": {
        "type": "object",
        "properties": {
            "context_prefix": {"type": "string"},
            "semantic_role": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(VALID_ROLES)},
                "minItems": 1,
                "maxItems": 4,
            },
        },
        "required": ["context_prefix", "semantic_role"],
        "additionalProperties": False,
    },
}
FLUSH_EVERY = 20


def build_embed_text(context_prefix: str, heading_path: list[str], rawtext: str) -> str:
    return f"{context_prefix}\n\n【{' → '.join(heading_path)}】\n{rawtext}"


async def call_llm(
    client: anthropic.AsyncAnthropic,
    model: str,
    source_doc_id: str,
    heading_tree: str,
    heading_path: list[str],
    rawtext: str,
    step_info: str | None = None,
) -> tuple[str, list[str]]:
    static_block = f"【文档标题】\n{source_doc_id}\n\n【完整目录结构】\n{heading_tree}"
    step_line = f"\n【步骤信息】{step_info}" if step_info else ""
    dynamic_block = (
        f"【当前段落位置】\n{' → '.join(heading_path)}{step_line}\n\n"
        f"【段落内容】\n{rawtext}\n\n请调用 emit_context 工具。"
    )
    response = await client.messages.create(
        model=model,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        thinking=disabled_thinking(),
        tools=[CONTEXT_TOOL],
        tool_choice={"type": "tool", "name": CONTEXT_TOOL_NAME},
        messages=[_user_message(static_block, dynamic_block)],
    )
    parsed = _context_tool_input(response)
    context_prefix = str(parsed.get("context_prefix") or "").strip()
    if not context_prefix:
        raise ValueError("DeepSeek tool output missing context_prefix")
    return context_prefix, _normalise_roles(parsed.get("semantic_role", ["general_info"]))


async def process_chunk(
    sem: asyncio.Semaphore,
    client: anthropic.AsyncAnthropic,
    model: str,
    chunk: dict,
    heading_tree: str,
    group_totals: dict[str, int],
) -> dict | None:
    async with sem:
        try:
            return await _build_chunk_record(client, model, chunk, heading_tree, group_totals)
        except Exception as exc:
            return {"_error": True, "chunk_id": chunk["chunk_id"], "error": str(exc), **chunk}


async def run(doc_dir: Path, concurrency: int, limit: int | None) -> None:
    src, tree_path, dst, err_dst = stage_paths(doc_dir)
    if not src.exists():
        print(f"ERROR: {src} not found — run clean_rawtext.py first", file=sys.stderr)
        raise SystemExit(1)
    heading_tree = tree_path.read_text(encoding="utf-8") if tree_path.exists() else ""
    chunks = read_jsonl(src)
    chunk_order = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
    pending = pending_chunks(chunks, done_ids(dst), limit)
    print(f"Processing {len(pending)} chunks (concurrency={concurrency})")
    client = anthropic.AsyncAnthropic(api_key=deepseek_api_key(), base_url=contextual_base_url())
    model = contextual_model()
    tasks = _tasks(concurrency, client, model, pending, heading_tree, group_totals(chunks))
    results, errors, elapsed = await _consume_tasks(tasks, pending, dst, err_dst)
    sort_rag_chunks(dst, chunk_order)
    print(f"Done in {elapsed:.1f}s — {len(results)} ok, {len(errors)} errors")
    if errors:
        print(f"Errors written to {err_dst}")


def _user_message(static_block: str, dynamic_block: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": static_block},
            {"type": "text", "text": dynamic_block},
        ],
    }


def _context_tool_input(response: object) -> dict:
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != CONTEXT_TOOL_NAME:
            continue
        payload = getattr(block, "input", None)
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"DeepSeek response did not call {CONTEXT_TOOL_NAME}")


def _normalise_roles(raw_roles: object) -> list[str]:
    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]
    roles = [role for role in raw_roles if isinstance(role, str) and role in VALID_ROLES]
    if not roles:
        roles = ["general_info"]
    if "general_info" in roles and len(roles) > 1:
        roles = [role for role in roles if role != "general_info"]
    return roles


async def _build_chunk_record(
    client: anthropic.AsyncAnthropic,
    model: str,
    chunk: dict,
    heading_tree: str,
    group_totals: dict[str, int],
) -> dict:
    context_prefix, semantic_role = await call_llm(
        client=client,
        model=model,
        source_doc_id=chunk["source_doc_id"],
        heading_tree=heading_tree,
        heading_path=chunk.get("heading_path") or [],
        rawtext=chunk.get("rawtext") or "",
        step_info=_step_info(chunk, group_totals),
    )
    return _rag_record(chunk, context_prefix, semantic_role)


def _rag_record(chunk: dict, context_prefix: str, semantic_role: list[str]) -> dict:
    heading_path = chunk.get("heading_path") or []
    rawtext = chunk.get("rawtext") or ""
    return {
        "chunk_id": chunk["chunk_id"],
        "source_doc_id": chunk["source_doc_id"],
        "source_path": chunk["source_path"],
        "source_pages": chunk.get("source_pages"),
        "chunk_type": chunk["chunk_type"],
        "rawtext": chunk["rawtext"],
        "context_prefix": context_prefix,
        "semantic_role": semantic_role,
        "embed_text": build_embed_text(context_prefix, heading_path, rawtext),
        "heading_path": heading_path,
        "clause_id": chunk.get("canonical_clause_id"),
        "cross_refs": chunk.get("cross_refs") or [],
        "resolved_cross_refs": chunk.get("resolved_cross_refs") or [],
        "sequential_group_id": chunk.get("sequential_group_id"),
        "step_index": chunk.get("step_index"),
        "step_title": chunk.get("step_title"),
        "parent_chunk_id": chunk.get("parent_chunk_id"),
        "image_paths": chunk.get("image_paths") or [],
        "table_caption": _table_caption(chunk),
        "table_label": (chunk.get("metadata") or {}).get("table_label"),
        "local_context_title": (chunk.get("metadata") or {}).get("local_context_title"),
    }


def _step_info(chunk: dict, group_totals: dict[str, int]) -> str | None:
    seq_id = chunk.get("sequential_group_id")
    step_idx = chunk.get("step_index")
    if not seq_id or step_idx is None:
        return None
    return f"第{step_idx}步，共{group_totals.get(seq_id, '?')}步"


def _table_caption(chunk: dict) -> str | None:
    table_raw = chunk.get("table_raw")
    if table_raw and isinstance(table_raw, list) and table_raw[0].get("caption"):
        return table_raw[0]["caption"]
    return None


def _tasks(
    concurrency: int,
    client: anthropic.AsyncAnthropic,
    model: str,
    pending: list[dict],
    heading_tree: str,
    group_totals: dict[str, int],
) -> list:
    sem = asyncio.Semaphore(concurrency)
    return [
        process_chunk(sem, client, model, chunk, heading_tree, group_totals) for chunk in pending
    ]


async def _consume_tasks(
    tasks: list,
    pending: list[dict],
    dst: Path,
    err_dst: Path,
) -> tuple[list[dict], list[dict], float]:
    results: list[dict] = []
    errors: list[dict] = []
    start = time.monotonic()
    for index, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        if result is None:
            continue
        flushed = _record_result(result, results, errors, dst, err_dst)
        if flushed:
            print(f"  [{index}/{len(pending)}] flushed — {time.monotonic() - start:.0f}s elapsed")
    remainder_count = len(results) % FLUSH_EVERY
    if remainder_count > 0:
        append_jsonl(dst, results[-remainder_count:])
    return results, errors, time.monotonic() - start


def _record_result(
    result: dict,
    results: list[dict],
    errors: list[dict],
    dst: Path,
    err_dst: Path,
) -> bool:
    if result.get("_error"):
        errors.append(result)
        append_jsonl(err_dst, [result])
        return False
    results.append(result)
    if len(results) % FLUSH_EVERY == 0:
        append_jsonl(dst, results[-FLUSH_EVERY:])
        return True
    return False
