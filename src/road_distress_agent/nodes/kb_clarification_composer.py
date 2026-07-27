"""Compose conditional KB answers when planning lacks required conditions."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.nodes.kb_answer_composer import KbAnswer
from road_distress_agent.nodes.kb_planning_utils import (
    evidence_text,
    references_json,
    refs_by_chunk_id,
)
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    KbClarification,
    ReferenceItem,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (
        Path(__file__).resolve().parents[1] / "prompts" / "kb_clarification_composer.txt"
    ).read_text(encoding="utf-8")


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "function_calling"


def _build_messages(state: AgentState) -> list[Any]:
    references = _references(state)
    chunks = _chunks(state)
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(content=_human_prompt(state, chunks, references)),
    ]


def _human_prompt(
    state: AgentState,
    chunks: list[RetrievedChunk],
    references: list[ReferenceItem],
) -> str:
    return "\n".join(
        [
            f"target_locale = {(state.get('locale') or 'zh-CN')!r}",
            user_language_instruction(state.get("locale")),
            f"original_question = {(state.get('latest_user_text') or '')!r}",
            "clarification = " + json.dumps(_clarification_payload(state), ensure_ascii=False),
            "missing_slots = "
            + json.dumps(state.get("kb_missing_slots") or [], ensure_ascii=False),
            f"reference_index = {references_json(references)}",
            f"evidence =\n{evidence_text(chunks, references)}",
            "Every cited_chunk_ids entry must have its matching [[R...]] token in answer.",
            "Return the KbAnswer JSON now.",
        ]
    )


def _invoke_llm(messages: list[Any]) -> KbAnswer:
    timeout = llm_timeout_seconds("kb_clarification_composer")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(KbAnswer, **kwargs)
    result = invoke_llm_call(
        error_context_name="kb_clarification_composer",
        usage_correlation_name="kb_clarification_composer",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, KbAnswer):
        return result
    return KbAnswer.model_validate(result)


def kb_clarification_composer(state: AgentState) -> AgentState:
    """Generate a conditional KB answer without creating diagnosis interrupts."""
    chunks = _chunks(state)
    references = _references(state)
    messages = _build_messages(state)
    answer = _invoke_llm(messages)
    _ensure_citations(answer, chunks, references)
    return {
        "direct_message": answer.answer,
        "kb_answer": answer.model_dump(mode="json"),
        "reference_index": references,
        "kb_final_cited_chunk_ids": answer.cited_chunk_ids,
        "awaiting_user_input": state.get("interrupt") is not None,
        "phase": WorkflowPhase.DONE,
        "next_action": "return_kb_answer",
        "audit_log": _audit_log(state, messages, answer),
    }


def _clarification_payload(state: AgentState) -> dict[str, Any]:
    raw = state.get("kb_clarification_request")
    if isinstance(raw, KbClarification):
        return raw.model_dump(mode="json")
    if isinstance(raw, dict):
        return KbClarification.model_validate(raw).model_dump(mode="json")
    return {}


def _chunks(state: AgentState) -> list[RetrievedChunk]:
    return [
        chunk if isinstance(chunk, RetrievedChunk) else RetrievedChunk.model_validate(chunk)
        for chunk in (state.get("kb_retrieved_chunks") or [])
    ]


def _references(state: AgentState) -> list[ReferenceItem]:
    return [
        item if isinstance(item, ReferenceItem) else ReferenceItem.model_validate(item)
        for item in (state.get("reference_index") or [])
    ]


def _ensure_citations(
    answer: KbAnswer,
    chunks: list[RetrievedChunk],
    references: list[ReferenceItem],
) -> None:
    if chunks and not answer.cited_chunk_ids:
        raise ValueError("kb_clarification_composer returned no cited_chunk_ids.")
    citable_ids = _citable_ids(references)
    invalid = [chunk_id for chunk_id in answer.cited_chunk_ids if chunk_id not in citable_ids]
    if invalid:
        raise ValueError(f"kb_clarification_composer cited unknown chunk_ids: {invalid!r}.")
    missing_tokens = _missing_inline_tokens(answer, references)
    if missing_tokens:
        raise ValueError(
            f"kb_clarification_composer omitted inline ref tokens: {missing_tokens!r}."
        )


def _citable_ids(references: list[ReferenceItem]) -> set[str]:
    return set(refs_by_chunk_id(references))


def _missing_inline_tokens(answer: KbAnswer, references: list[ReferenceItem]) -> list[str]:
    refs_by_chunk = refs_by_chunk_id(references)
    ref_ids = {
        refs_by_chunk[chunk_id] for chunk_id in answer.cited_chunk_ids if chunk_id in refs_by_chunk
    }
    return [ref_id for ref_id in sorted(ref_ids) if f"[[{ref_id}]]" not in answer.answer]


def _audit_log(state: AgentState, messages: list[Any], answer: KbAnswer) -> list[AuditEvent]:
    return [
        trace_event(
            node_name="kb_clarification_composer",
            kind="llm_call",
            title="KB clarification answer composer LLM",
            inputs={
                "latest_user_text": state.get("latest_user_text"),
                "kb_plan_type": state.get("kb_plan_type"),
            },
            prompt=messages,
            output=answer,
        ),
        AuditEvent(
            node_name="kb_clarification_composer",
            message="Composed conditional knowledge-base answer.",
            metadata={"cited_chunk_ids": answer.cited_chunk_ids},
        ),
    ]
