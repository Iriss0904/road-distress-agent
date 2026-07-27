"""Merge optional weather advice with cited construction safety norms."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.memory_context import construction_tip_memory_block
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    Citation,
    FinalAnswer,
    RetrievedChunk,
    WeatherAdvice,
)
from road_distress_agent.tracing import trace_event


class SafetyNormNote(BaseModel):
    text: str = Field(min_length=1)
    clause_ids: list[str] = Field(min_length=1)


class ConstructionArrangementDraft(BaseModel):
    safety_notes: list[SafetyNormNote] = Field(default_factory=list)
    weather_advice: WeatherAdvice | None = None
    need_human_review: bool = False
    evidence_gaps: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / "construction_arrangement_advisor.txt"
    return path.read_text(encoding="utf-8")


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _format_chunks(chunks: list[RetrievedChunk], max_body: int = 900) -> str:
    if not chunks:
        return "（无证据）"
    lines: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        clause = f"[{chunk.clause_id}]" if chunk.clause_id else ""
        body = (chunk.text or "").strip()[:max_body]
        lines.append(f"[{index}]{clause} {chunk.context_prefix or ''}\n{body}")
    return "\n\n".join(lines)


def _messages(state: AgentState) -> list[Any]:
    weather = state.get("weather_context")
    answer = state.get("final_answer")
    payload = {
        "road_class": state.get("road_class"),
        "inferred_work_duration": state.get("inferred_work_duration"),
        "safety_query": state.get("safety_query"),
        "weather_context": weather.model_dump(exclude_none=True) if weather else None,
        "final_answer_summary": answer.summary if isinstance(answer, FinalAnswer) else None,
        "target_locale": state.get("locale") or "zh-CN",
    }
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    construction_tip_memory_block(state.get("loaded_memory")),
                    user_language_instruction(state.get("locale")),
                    f"arrangement_input = {payload}",
                    "safety_norm_chunks =",
                    _format_chunks(state.get("safety_norm_chunks") or []),
                    "Return the ConstructionArrangementDraft JSON now.",
                ]
            )
        ),
    ]


def _invoke_llm(messages: list[Any]) -> ConstructionArrangementDraft:
    timeout = llm_timeout_seconds("construction_arrangement_advisor")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(ConstructionArrangementDraft, **kwargs)
    result = invoke_llm_call(
        error_context_name="construction_arrangement_advisor",
        usage_correlation_name="construction_arrangement_advisor",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, ConstructionArrangementDraft):
        return result
    return ConstructionArrangementDraft.model_validate(result)


def _known_clauses(chunks: list[RetrievedChunk]) -> set[str]:
    return {chunk.clause_id for chunk in chunks if chunk.clause_id}


def _validated_notes(
    draft: ConstructionArrangementDraft,
    chunks: list[RetrievedChunk],
) -> list[str]:
    known = _known_clauses(chunks)
    notes: list[str] = []
    for note in draft.safety_notes:
        unknown = [clause for clause in note.clause_ids if clause not in known]
        if unknown:
            raise ValueError(f"Safety note cited unknown clause_ids: {unknown}")
        clauses = "、".join(note.clause_ids)
        text = note.text
        if not any(clause in note.text for clause in note.clause_ids):
            text = f"{note.text}（依据：{clauses}）"
        notes.append(text)
    return notes


def _with_arrangement(
    answer: FinalAnswer | None,
    draft: ConstructionArrangementDraft,
    notes: list[str],
    chunks: list[RetrievedChunk],
) -> FinalAnswer | None:
    if answer is None:
        return None
    gaps = list(draft.evidence_gaps)
    need_review = draft.need_human_review
    if not chunks:
        gaps.append("safety_norm: no_safety_norm_chunks_retrieved")
        need_review = True
    return answer.model_copy(
        update={
            "safety_notes": _append_unique(answer.safety_notes, notes),
            "citations": _append_citations(answer.citations, chunks),
            "weather_advice": draft.weather_advice,
            "weather_constraints": draft.weather_advice.constraints if draft.weather_advice else [],
            "need_human_review": answer.need_human_review or need_review,
            "evidence_gaps": _append_unique(answer.evidence_gaps, gaps),
        }
    )


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for addition in additions:
        if addition and addition not in result:
            result.append(addition)
    return result


def _append_citations(
    citations: list[Citation],
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    result = list(citations)
    seen = {item.chunk_id or item.citation_id for item in result}
    for chunk in chunks:
        key = chunk.chunk_id or chunk.citation_id
        if key in seen:
            continue
        seen.add(key)
        result.append(Citation.model_validate(chunk.model_dump()))
    return result


def _validate_weather(state: AgentState, draft: ConstructionArrangementDraft) -> None:
    if state.get("weather_context") is None and draft.weather_advice is not None:
        raise ValueError(
            "construction_arrangement_advisor returned weather_advice without weather_context."
        )


def construction_arrangement_advisor(state: AgentState) -> AgentState:
    messages = _messages(state)
    draft = _invoke_llm(messages)
    _validate_weather(state, draft)
    chunks = state.get("safety_norm_chunks") or []
    notes = _validated_notes(draft, chunks)
    final_answer = _with_arrangement(state.get("final_answer"), draft, notes, chunks)
    return {
        "weather_advice": draft.weather_advice,
        "final_answer": final_answer,
        "weather_route": "arrangement_advised",
        "phase": WorkflowPhase.SAFETY,
        "next_action": "run_critic",
        "errors": [],
        "audit_log": [
            trace_event(
                node_name="construction_arrangement_advisor",
                kind="llm_call",
                title="Construction arrangement advisor LLM",
                inputs={
                    "road_class": state.get("road_class"),
                    "safety_chunk_count": len(chunks),
                    "weather_context": state.get("weather_context"),
                },
                prompt=messages,
                output=draft,
            ),
            AuditEvent(
                node_name="construction_arrangement_advisor",
                message="Generated optional construction arrangement advice.",
                metadata={
                    "safety_note_count": len(notes),
                    "weather_status": draft.weather_advice.status if draft.weather_advice else None,
                },
            ),
        ],
    }
