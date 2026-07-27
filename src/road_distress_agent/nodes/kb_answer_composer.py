"""Compose chat-bubble answers for pure knowledge-base QA."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_gate import (
    RUNTIME_R1_GATE_POLICY,
    SimpleKbGateInput,
    evidence_gate,
)
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.memory_context import expression_memory_block
from road_distress_agent.nodes.evidence_gate_runtime import (
    assessment,
    gate_trace,
    raise_on_error,
)
from road_distress_agent.nodes.evidence_gate_runtime import (
    chunks as runtime_chunks,
)
from road_distress_agent.nodes.kb_evidence_boundary import (
    ComposerEvidence,
    composer_evidence,
    ensure_citations_allowed,
)
from road_distress_agent.nodes.kb_fixed_outcome import fixed_kb_outcome_for_status
from road_distress_agent.reference_index import build_reference_index
from road_distress_agent.retrieval.query_features import detect_query_features
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    Citation,
    ReferenceItem,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event

NODE_NAME = "kb_answer_composer"


class KbAnswer(BaseModel):
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "kb_answer_composer.txt").read_text(
        encoding="utf-8"
    )


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _evidence_text(
    chunks: list[RetrievedChunk],
    references: list[ReferenceItem],
) -> str:
    if not chunks:
        return "（未检索到证据）"
    refs_by_chunk = _refs_by_chunk_id(references)
    lines: list[str] = []
    for index, chunk in enumerate(chunks[:8], 1):
        chunk_id = chunk.chunk_id or chunk.citation_id or f"chunk-{index}"
        ref_id = refs_by_chunk.get(chunk_id, "")
        clause = chunk.clause_id or chunk.source_clause or ""
        prefix = chunk.context_prefix or ""
        body = (chunk.text or chunk.snippet or "").strip()[:800]
        lines.append(
            f"[{index}] ref_id={ref_id} chunk_id={chunk_id} clause={clause}\n{prefix}\n{body}"
        )
    return "\n\n".join(lines)


def _build_messages(
    state: AgentState,
    boundary: ComposerEvidence | None = None,
) -> list[Any]:
    active = boundary or _composer_evidence(state)
    chunks = list(active.chunks)
    references = list(active.references)
    unsupported = active.unsupported_payload()
    boundary_lines = (
        [f"unsupported_slots = {json.dumps(unsupported, ensure_ascii=False)}"]
        if unsupported
        else []
    )
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    f"target_locale = {(state.get('locale') or 'zh-CN')!r}",
                    user_language_instruction(state.get("locale")),
                    f"user_question = {(state.get('latest_user_text') or '')!r}",
                    expression_memory_block(state.get("loaded_memory")),
                    f"reference_index = {references_json(references)}",
                    f"evidence =\n{_evidence_text(chunks, references)}",
                    *boundary_lines,
                    "Return the KbAnswer JSON now.",
                ]
            )
        ),
    ]


def _invoke_llm(messages: list[Any]) -> KbAnswer:
    timeout = llm_timeout_seconds("kb_answer_composer")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(KbAnswer, **kwargs)
    result = invoke_llm_call(
        error_context_name="kb_answer_composer",
        usage_correlation_name="kb_answer_composer",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, KbAnswer):
        return result
    return KbAnswer.model_validate(result)


def kb_answer_composer(state: AgentState) -> AgentState:
    """Generate a direct chat answer without modifying diagnosis outputs."""
    policy = RUNTIME_R1_GATE_POLICY
    selected = runtime_chunks(state.get("kb_retrieved_chunks") or ())
    current_assessment = assessment(state)
    decision = evidence_gate(
        SimpleKbGateInput(
            assessment=current_assessment,
            query=state.get("latest_user_text") or "",
            chunks=selected,
            explicit_clause_ids=detect_query_features(
                state.get("latest_user_text") or "", {}
            ).clause_ids,
        ),
        policy,
    )
    decision_trace = gate_trace(NODE_NAME, decision, policy)
    raise_on_error(decision, current_assessment)
    if decision.decision == "REFUSE":
        fixed = fixed_kb_outcome_for_status(state, NODE_NAME, decision.reason)
        if fixed is None:
            raise ValueError(f"unsupported KB gate refusal: {decision.reason.value}")
        fixed["audit_log"] = [decision_trace, *(fixed.get("audit_log") or [])]
        return fixed
    usable = list(decision.usable_chunks)
    boundary = ComposerEvidence(tuple(usable), tuple(_reference_index(usable)))
    references = list(boundary.references)
    messages = _build_messages(state, boundary)
    answer = _invoke_llm(messages)
    llm_trace = _llm_trace(state, messages, answer)
    ensure_citations_allowed(answer.cited_chunk_ids, boundary.references, NODE_NAME)
    return {
        "direct_message": answer.answer,
        "refusal_type": None,
        "kb_answer": answer.model_dump(mode="json"),
        "reference_index": references,
        "kb_final_cited_chunk_ids": answer.cited_chunk_ids,
        "awaiting_user_input": state.get("interrupt") is not None,
        "phase": WorkflowPhase.DONE,
        "next_action": "return_kb_answer",
        "audit_log": [
            decision_trace,
            llm_trace,
            AuditEvent(
                node_name=NODE_NAME,
                message="Composed direct knowledge-base answer.",
                metadata=_audit_metadata(answer, references, boundary),
            ),
        ],
    }


def _composer_evidence(state: AgentState) -> ComposerEvidence:
    chunks = [
        chunk if isinstance(chunk, RetrievedChunk) else RetrievedChunk.model_validate(chunk)
        for chunk in (state.get("kb_retrieved_chunks") or [])
    ]
    return composer_evidence(state, chunks, _reference_index(chunks))


def _llm_trace(state: AgentState, messages: list[Any], answer: KbAnswer) -> AuditEvent:
    return trace_event(
        node_name=NODE_NAME,
        kind="llm_call",
        title="KB answer composer LLM",
        inputs={
            "latest_user_text": state.get("latest_user_text"),
            "locale": state.get("locale") or "zh-CN",
            "memory_present": state.get("loaded_memory") is not None,
        },
        prompt=messages,
        output=answer,
    )


def _audit_metadata(
    answer: KbAnswer,
    references: list[ReferenceItem],
    boundary: ComposerEvidence,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "cited_chunk_ids": answer.cited_chunk_ids,
        "reference_count": len(references),
    }
    if boundary.unsupported_slots:
        metadata["unsupported_slots"] = boundary.unsupported_payload()
    return metadata


def references_json(references: list[ReferenceItem]) -> str:
    items = [
        {
            "ref_id": item.ref_id,
            "chunk_ids": item.chunk_ids,
            "title": item.title,
            "source_clause": item.source_clause,
        }
        for item in references
    ]
    return json.dumps(items, ensure_ascii=False)


def _reference_index(chunks: list[RetrievedChunk]) -> list[ReferenceItem]:
    return build_reference_index([_citation(chunk) for chunk in chunks])


def _citation(chunk: RetrievedChunk) -> Citation:
    return Citation.model_validate(chunk.model_dump(mode="json"))


def _refs_by_chunk_id(references: list[ReferenceItem]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in references:
        for chunk_id in item.chunk_ids:
            lookup[chunk_id] = item.ref_id
    return lookup
