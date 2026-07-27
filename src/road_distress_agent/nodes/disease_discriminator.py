"""Phase C.1 disease identification discriminator node."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.nodes.discriminator_tool_schemas import (
    disease_output_from_tool,
    disease_tools,
)
from road_distress_agent.nodes.strict_tool_calling import (
    invoke_strict_tool_call,
    strict_tool_trace_payload,
)
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    DiseaseDiscriminatorOutput,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (
        Path(__file__).resolve().parents[1] / "prompts" / "disease_discriminator.txt"
    ).read_text(encoding="utf-8")


def _evidence_text(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "（无检索到的病害定义证据）"
    lines: list[str] = []
    for i, c in enumerate(chunks[:6], 1):
        prefix = c.context_prefix or ""
        body = (c.text or "").strip()[:600]
        lines.append(f"[{i}] {prefix}\n{body}")
    return "\n\n".join(lines)


def _build_messages(
    user_text: str,
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
                    f"user_text = {user_text!r}",
                    f"target_locale = {locale!r}",
                    user_language_instruction(locale),
                    "Candidate name fields must remain Chinese canonical distress names.",
                    f"known_features = {known}",
                    f"clarification_attempts = {attempts}",
                    f"retrieved_evidence =\n{_evidence_text(chunks)}",
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
) -> tuple[DiseaseDiscriminatorOutput, dict[str, Any]]:
    result = invoke_strict_tool_call(
        node_name="disease_discriminator",
        messages=messages,
        tools=disease_tools(attempts),
    )
    return disease_output_from_tool(result), strict_tool_trace_payload(result)


def disease_discriminator(state: AgentState) -> AgentState:
    """Phase C.1: identify disease type from retrieved disease-definition evidence.

    Three output routes:
    - candidates                    → presents 1-3 candidates for user to select (HITL)
    - sufficient=False, question    → asks user one clarifying question (HITL)
    - sufficient=True               → protocol violation; diagnosis requires HITL selection
    """
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks") or []
    attempts: int = state.get("clarification_attempts") or 0
    known = state.get("known_features") or {}
    user_text = state.get("latest_user_text") or state.get("raw_user_text") or ""
    inputs = {
        "user_text": user_text,
        "locale": state.get("locale") or "zh-CN",
        "known_features": known,
        "clarification_attempts": attempts,
    }

    messages = _build_messages(user_text, known, attempts, chunks, inputs["locale"])
    output, tool_decision = _invoke_llm(messages, attempts)
    _enforce_clarification_budget(output, attempts)

    return {
        "disease_evidence_chunks": chunks,
        "disease_discriminator_output": output,
        **_route_delta(output, attempts),
        "audit_log": [
            trace_event(
                node_name="disease_discriminator",
                kind="llm_call",
                title="Disease discriminator LLM",
                inputs=inputs,
                prompt=messages,
                output={"tool_decision": tool_decision, "effective_output": output},
            ),
            AuditEvent(
                node_name="disease_discriminator",
                message="Phase C.1 disease identification ran.",
                metadata={
                    "sufficient": output.sufficient,
                    "identified_disease": output.identified_disease,
                    "confidence": output.confidence,
                    "candidate_count": len(output.candidates),
                },
            ),
        ],
    }


def _route_delta(output: DiseaseDiscriminatorOutput, attempts: int) -> AgentState:
    if output.sufficient:
        raise ValueError("Disease discriminator direct lock reached routing.")
    if output.candidates:
        return {
            "need_more_info": True,
            "question_to_user": None,
            "next_action": "hitl_disease_selection",
        }
    return {
        "need_more_info": True,
        "question_to_user": output.clarifying_question,
        "clarification_attempts": attempts + 1,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "hitl_disease_clarification",
    }


def _enforce_clarification_budget(
    output: DiseaseDiscriminatorOutput,
    attempts: int,
) -> None:
    if attempts < 1 or output.sufficient or output.candidates:
        return
    raise ValueError(
        "Disease discriminator violated the diagnosis contract: clarification "
        "budget is exhausted; return at least one disease candidate."
    )
