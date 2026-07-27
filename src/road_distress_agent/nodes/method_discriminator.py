"""Phase C.2 method selection discriminator node."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.nodes.discriminator_tool_schemas import (
    method_output_from_tool,
    method_tools,
)
from road_distress_agent.nodes.method_candidate_contract import (
    explicit_method_names,
    normalize_method_candidate_names,
)
from road_distress_agent.nodes.strict_tool_calling import (
    invoke_strict_tool_call,
    strict_tool_trace_payload,
)
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    MethodDiscriminatorOutput,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event

METHOD_EVIDENCE_BODY_LIMIT = 1200


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "method_discriminator.txt").read_text(
        encoding="utf-8"
    )


def _evidence_text(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "（无检索到的处治方案证据）"
    lines: list[str] = []
    for i, c in enumerate(chunks[:8], 1):
        prefix = c.context_prefix or ""
        body = (c.text or "").strip()[:METHOD_EVIDENCE_BODY_LIMIT]
        lines.append(f"[{i}] {prefix}\n{body}")
    return "\n\n".join(lines)


def _build_messages(
    defect: str,
    user_text: str,
    *,
    known: dict[str, Any],
    attempts: int,
    chunks: list[RetrievedChunk],
    locale: str,
) -> list[Any]:
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    f"defect_category = {defect!r}",
                    f"user_text = {user_text!r}",
                    f"target_locale = {locale!r}",
                    user_language_instruction(locale),
                    "Candidate name fields must remain Chinese canonical treatment names.",
                    f"known_features = {known}",
                    f"clarification_attempts = {attempts}",
                    f"retrieved_evidence =\n{_evidence_text(chunks)}",
                    "canonical_method_names_extracted_from_evidence = "
                    f"{explicit_method_names(chunks)!r}",
                    "When that list is non-empty, use its exact applicable names and "
                    "return each applicable method as an independent candidate.",
                    "Tool string arguments must not contain ASCII double quote characters. "
                    "Paraphrase evidence instead of quoting phrases.",
                    "Call exactly one provided tool now.",
                ]
            )
        ),
    ]


def _invoke_llm(
    messages: list[Any],
    attempts: int,
) -> tuple[MethodDiscriminatorOutput, dict[str, Any]]:
    result = invoke_strict_tool_call(
        node_name="method_discriminator",
        messages=messages,
        tools=method_tools(attempts),
    )
    return method_output_from_tool(result), strict_tool_trace_payload(result)


def method_discriminator(state: AgentState) -> AgentState:
    """Phase C.2: select a treatment plan from retrieved evidence.

    Three output routes:
    - sufficient=True               → protocol violation; diagnosis requires HITL selection
    - sufficient=False, question    → asks user one clarifying question (HITL)
    - sufficient=False, candidates  → presents 1-3 treatment plans (HITL)
    """
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks") or []
    attempts: int = state.get("clarification_attempts") or 0
    known = state.get("known_features") or {}
    user_text = state.get("latest_user_text") or state.get("raw_user_text") or ""
    defect = state.get("defect_category") or ""
    locale: str = state.get("locale") or "zh-CN"
    inputs = {
        "defect_category": defect,
        "user_text": user_text,
        "locale": locale,
        "known_features": known,
        "clarification_attempts": attempts,
    }

    messages = _build_messages(
        defect,
        user_text,
        known=known,
        attempts=attempts,
        chunks=chunks,
        locale=locale,
    )
    output, tool_decision = _invoke_llm(messages, attempts)
    _enforce_clarification_budget(output, attempts)
    output = normalize_method_candidate_names(output, chunks)

    return {
        "method_evidence_chunks": chunks,
        "method_discriminator_output": output,
        **_route_delta(output, attempts),
        "audit_log": _audit_log(
            inputs=inputs,
            messages=messages,
            tool_decision=tool_decision,
            output=output,
        ),
    }


def _audit_log(
    *,
    inputs: dict[str, Any],
    messages: list[Any],
    tool_decision: dict[str, Any],
    output: MethodDiscriminatorOutput,
) -> list[AuditEvent]:
    return [
        trace_event(
            node_name="method_discriminator",
            kind="llm_call",
            title="Treatment discriminator LLM",
            inputs=inputs,
            prompt=messages,
            output={"tool_decision": tool_decision, "effective_output": output},
        ),
        AuditEvent(
            node_name="method_discriminator",
            message="Phase C.2 treatment plan selection ran.",
            metadata={
                "sufficient": output.sufficient,
                "identified_method": output.identified_method,
                "confidence": output.confidence,
                "candidate_count": len(output.candidates),
            },
        ),
    ]


def _route_delta(output: MethodDiscriminatorOutput, attempts: int) -> AgentState:
    if output.sufficient:
        raise ValueError("Treatment discriminator direct lock reached routing.")
    if output.candidates:
        return {
            "need_more_info": True,
            "question_to_user": None,
            "next_action": "hitl_method_selection",
        }
    return {
        "need_more_info": True,
        "question_to_user": output.clarifying_question,
        "clarification_attempts": attempts + 1,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "hitl_method_clarification",
    }


def _enforce_clarification_budget(
    output: MethodDiscriminatorOutput,
    attempts: int,
) -> None:
    if attempts < 1 or output.sufficient or output.candidates:
        return
    raise ValueError(
        "Method discriminator violated the diagnosis contract: clarification "
        "budget is exhausted; return at least one method candidate."
    )
