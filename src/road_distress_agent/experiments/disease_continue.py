"""Experimental C.1 clarification continuation without a fresh RAG search."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.nodes.disease_discriminator import _evidence_text
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    DiseaseDiscriminatorOutput,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event

_CONFIDENCE_HUMAN_REVIEW_THRESHOLD = 0.60


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (
        Path(__file__).resolve().parents[1] / "prompts" / "disease_discriminator_continue.txt"
    ).read_text(encoding="utf-8")


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _message_text(message: Any) -> str | None:
    content = getattr(message, "content", None)
    message_type = getattr(message, "type", None)
    if isinstance(message, dict):
        content = message.get("content", content)
        message_type = message.get("type", message_type)
    if message_type not in {"human", "user"} or content is None:
        return None
    return content if isinstance(content, str) else str(content)


def _conversation_context(state: AgentState) -> str:
    lines = [
        f"[user {index}] {text}"
        for index, text in enumerate(
            filter(None, (_message_text(item) for item in state.get("messages") or [])),
            start=1,
        )
    ]
    if not lines:
        raise ValueError("continue mode requires accumulated human messages in state['messages'].")
    return "\n".join(lines)


def _previous_output(state: AgentState) -> DiseaseDiscriminatorOutput:
    output = state.get("disease_discriminator_output")
    if output is None:
        raise ValueError("continue mode requires previous disease_discriminator_output.")
    if isinstance(output, DiseaseDiscriminatorOutput):
        return output
    return DiseaseDiscriminatorOutput.model_validate(output)


def _previous_evidence(state: AgentState) -> list[RetrievedChunk]:
    chunks = state.get("disease_evidence_chunks") or []
    if not chunks:
        raise ValueError("continue mode requires previous disease_evidence_chunks.")
    return [
        chunk if isinstance(chunk, RetrievedChunk) else RetrievedChunk.model_validate(chunk)
        for chunk in chunks
    ]


def _updated_known_features(
    state: AgentState,
    previous: DiseaseDiscriminatorOutput,
) -> dict[str, Any]:
    features = dict(state.get("known_features") or {})
    reply = (state.get("latest_user_text") or "").strip()
    if not reply:
        return features
    features["latest_disease_clarification"] = reply
    if previous.missing_feature:
        features[previous.missing_feature] = reply
    return features


def _build_messages(state: AgentState) -> list[Any]:
    interrupt = state.get("interrupt")
    previous = _previous_output(state)
    chunks = _previous_evidence(state)
    known = _updated_known_features(state, previous)
    payload = "\n".join(
        [
            f"target_locale = {(state.get('locale') or 'zh-CN')!r}",
            user_language_instruction(state.get("locale")),
            "Candidate name fields must remain Chinese canonical distress names.",
            f"pending_question = {(interrupt.prompt if interrupt else None)!r}",
            f"latest_user_reply = {(state.get('latest_user_text') or '')!r}",
            f"conversation_context =\n{_conversation_context(state)}",
            f"known_features = {known}",
            f"clarification_attempts = {state.get('clarification_attempts') or 0}",
            f"previous_decision = {previous.model_dump_json(exclude_none=True)}",
            f"retrieved_evidence =\n{_evidence_text(chunks)}",
            "Return the DiseaseDiscriminatorOutput JSON now.",
        ]
    )
    return [SystemMessage(content=_prompt_text()), HumanMessage(content=payload)]


def disease_discriminator_continue(state: AgentState) -> AgentState:
    """Continue C.1 after a clarification answer using only prior evidence."""
    chunks = _previous_evidence(state)
    previous = _previous_output(state)
    known = _updated_known_features(state, previous)
    messages = _build_messages(state)
    node_name = "disease_discriminator_continue"
    timeout = llm_timeout_seconds(node_name)
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(DiseaseDiscriminatorOutput, **kwargs)
    result = invoke_llm_call(
        error_context_name=node_name,
        usage_correlation_name=node_name,
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    output = (
        result
        if isinstance(result, DiseaseDiscriminatorOutput)
        else DiseaseDiscriminatorOutput.model_validate(result)
    )
    if output.sufficient and output.confidence < _CONFIDENCE_HUMAN_REVIEW_THRESHOLD:
        output = output.model_copy(update={"need_human_review": True})
    llm_trace = _llm_trace_event(state, known, previous, messages, output)
    return _build_delta(
        state=state,
        chunks=chunks,
        output=output,
        known_features=known,
        llm_trace=llm_trace,
    )


def _llm_trace_event(
    state: AgentState,
    known: dict[str, Any],
    previous: DiseaseDiscriminatorOutput,
    messages: list[Any],
    output: DiseaseDiscriminatorOutput,
) -> AuditEvent:
    return trace_event(
        node_name="disease_discriminator_continue",
        kind="llm_call",
        title="Disease discriminator continue LLM",
        inputs={
            "latest_user_text": state.get("latest_user_text") or "",
            "locale": state.get("locale") or "zh-CN",
            "known_features": known,
            "previous_output": previous,
        },
        prompt=messages,
        output=output,
    )


def _build_delta(
    *,
    state: AgentState,
    chunks: list[RetrievedChunk],
    output: DiseaseDiscriminatorOutput,
    known_features: dict[str, Any],
    llm_trace: AuditEvent,
) -> AgentState:
    delta: AgentState = {
        "retrieved_chunks": chunks,
        "disease_evidence_chunks": chunks,
        "disease_discriminator_output": output,
        "known_features": known_features,
        "phase": WorkflowPhase.EVIDENCE,
        "audit_log": [
            llm_trace,
            AuditEvent(
                node_name="disease_discriminator_continue",
                message="Experimental C.1 continue mode ran without fresh retrieval.",
                metadata={
                    "sufficient": output.sufficient,
                    "identified_disease": output.identified_disease,
                    "confidence": output.confidence,
                    "candidate_count": len(output.candidates),
                },
            ),
        ],
    }
    if output.sufficient:
        return {
            **delta,
            "need_more_info": False,
            "question_to_user": None,
            "next_action": "invalid_disease_selection_contract",
        }
    if output.candidates:
        return {**delta, "need_more_info": True, "next_action": "hitl_disease_selection"}
    attempts = state.get("clarification_attempts") or 0
    return {
        **delta,
        "need_more_info": True,
        "question_to_user": output.clarifying_question,
        "clarification_attempts": attempts + 1,
        "next_action": "hitl_disease_clarification",
    }
