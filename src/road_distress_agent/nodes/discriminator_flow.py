from __future__ import annotations

from uuid import uuid4

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_observation import observe_clarification_interrupt
from road_distress_agent.experiments.disease_continue import disease_discriminator_continue
from road_distress_agent.localization import canonical_from_english_text
from road_distress_agent.nodes.diagnosis_dependencies import canonical_key
from road_distress_agent.nodes.discriminator_prompts import (
    more_site_info_prompt,
    selection_prompt,
)
from road_distress_agent.nodes.interrupt_trace import interrupt_trace_event
from road_distress_agent.nodes.method_retrieval_context import (
    method_context_from_state,
    method_stage_goal,
)
from road_distress_agent.resume import (
    DISEASE_CLARIFICATION,
    DISEASE_SELECTION,
    METHOD_CLARIFICATION,
    METHOD_SELECTION,
)
from road_distress_agent.retrieval.evidence import EvidenceRetrievalOptions, retrieve_evidence
from road_distress_agent.speculative_prefetch import attach_detail_prefetch
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    DiseaseDiscriminatorOutput,
    InterruptState,
    MethodDiscriminatorOutput,
)
from road_distress_agent.tracing import trace_event

_MODE_CONTINUE = "continue"
_MODE_RERAG = "rerag"
_ALLOWED_DISEASE_STRATEGIES = frozenset({_MODE_RERAG, _MODE_CONTINUE})
_STRIP_CHARS = " \t\r\n，。？！、；：,.!?;:（）()【】[]{}《》<>“”\"'`"
_TERM_NORMALIZATIONS = (("裂隙", "裂缝"),)
_METHOD_EVIDENCE_ROLES = ("method_selection", "construction_step")


def disease_selection_handler(state: AgentState) -> AgentState:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    if kind != DISEASE_SELECTION:
        return {"next_action": "rewrite_disease_query"}
    selected = _match_candidate(
        state.get("latest_user_text") or "",
        state.get("disease_discriminator_output"),
    )
    if not selected:
        return {"next_action": "rewrite_disease_query"}
    return {
        "defect_category": selected,
        "interrupt": None,
        "awaiting_user_input": False,
        "clarification_attempts": 0,
        "next_action": "method_phase",
        "audit_log": [_selection_trace("disease_selection_handler", selected, state)],
    }


def disease_retriever(state: AgentState) -> AgentState:
    query = state["rewritten_query"]
    filters = state["query_plan"].filters
    chunks, attempts, traces = retrieve_evidence(
        EvidenceRetrievalOptions(
            node_name="disease_retriever",
            title="Disease evidence search",
            query=query,
            filters=filters,
            stage_goal="病害类型判别：选择能支持病害定义、外观特征和判定标准的证据。",
        )
    )
    return {
        "retrieved_chunks": chunks,
        "disease_evidence_chunks": chunks,
        "retrieval_attempts": attempts,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "discriminate_disease",
        "audit_log": traces,
    }


def disease_continue_discriminator(state: AgentState) -> AgentState:
    delta = disease_discriminator_continue(state)
    return {
        **delta,
        "disease_evidence_chunks": state.get("disease_evidence_chunks") or [],
        "next_action": delta.get("next_action"),
    }


def disease_result_router(state: AgentState) -> AgentState:
    out = _disease_output(state)
    if out.sufficient:
        raise ValueError(
            "Disease discriminator violated the diagnosis contract: direct disease "
            "locking is disallowed; return candidates for user confirmation."
        )
    if out.candidates:
        interrupt = InterruptState(
            interrupt_id=f"interrupt-{uuid4().hex[:8]}",
            node_name="disease_result_router",
            kind=DISEASE_SELECTION,
            prompt=selection_prompt("病害类型", "distress type", len(out.candidates), state=state),
            candidate_ids=[candidate.name for candidate in out.candidates],
        )
        return _interrupt_delta(interrupt, "hitl_disease_selection")
    interrupt = InterruptState(
        interrupt_id=f"interrupt-{uuid4().hex[:8]}",
        node_name="disease_result_router",
        kind=DISEASE_CLARIFICATION,
        prompt=out.clarifying_question or more_site_info_prompt(state),
        required_fields=_required_fields(out.missing_feature),
    )
    return _interrupt_delta(interrupt, "hitl_disease_clarification")


def method_selection_handler(state: AgentState) -> AgentState:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    if kind == METHOD_CLARIFICATION:
        return {"next_action": "rewrite_method_query"}
    if kind != METHOD_SELECTION:
        return {"next_action": "rewrite_method_query"}
    selected = _match_candidate(
        state.get("latest_user_text") or "",
        state.get("method_discriminator_output"),
    )
    if not selected:
        return {"next_action": "rewrite_method_query"}
    return {
        "chosen_method": selected,
        "interrupt": None,
        "awaiting_user_input": False,
        "clarification_attempts": 0,
        "next_action": "detail_phase",
        "audit_log": [_selection_trace("method_selection_handler", selected, state)],
    }


def method_retriever(state: AgentState) -> AgentState:
    query = state["rewritten_query"]
    filters = {
        **state["query_plan"].filters,
        "semantic_role": _METHOD_EVIDENCE_ROLES,
    }
    chunks, attempts, traces = retrieve_evidence(
        EvidenceRetrievalOptions(
            node_name="method_retriever",
            title="Method decision search",
            query=query,
            filters=filters,
            stage_goal=method_stage_goal(method_context_from_state(state)),
            preserve_semantic_role_filter=True,
            split_numbered_rerank_passages=True,
        )
    )
    return {
        "retrieved_chunks": chunks,
        "method_evidence_chunks": chunks,
        "retrieval_attempts": attempts,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "discriminate_method",
        "audit_log": traces,
    }


def method_result_router(state: AgentState) -> AgentState:
    out = _method_output(state)
    if out.sufficient:
        raise ValueError(
            "Method discriminator violated the diagnosis contract: direct method "
            "locking is disallowed; return candidates for user confirmation."
        )
    if out.candidates:
        interrupt = InterruptState(
            interrupt_id=f"interrupt-{uuid4().hex[:8]}",
            node_name="method_result_router",
            kind=METHOD_SELECTION,
            prompt=selection_prompt(
                "处治方案", "treatment option", len(out.candidates), state=state
            ),
            candidate_ids=[candidate.name for candidate in out.candidates],
        )
        delta = _interrupt_delta(interrupt, "hitl_method_selection")
        return attach_detail_prefetch(state, interrupt, delta)
    interrupt = InterruptState(
        interrupt_id=f"interrupt-{uuid4().hex[:8]}",
        node_name="method_result_router",
        kind=METHOD_CLARIFICATION,
        prompt=out.clarifying_question or more_site_info_prompt(state),
        required_fields=_required_fields(out.missing_feature),
    )
    return _interrupt_delta(interrupt, "hitl_method_clarification")


def should_continue_disease(state: AgentState) -> bool:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    return kind == DISEASE_CLARIFICATION and _disease_strategy(state) == _MODE_CONTINUE


def _disease_strategy(state: AgentState) -> str:
    strategy = (state.get("retrieval_strategy") or _MODE_RERAG).strip().lower()
    if strategy not in _ALLOWED_DISEASE_STRATEGIES:
        raise ValueError(
            "retrieval_strategy for disease discrimination must be one of "
            f"{sorted(_ALLOWED_DISEASE_STRATEGIES)}, got {strategy!r}."
        )
    return strategy


def _disease_output(state: AgentState) -> DiseaseDiscriminatorOutput:
    out = state["disease_discriminator_output"]
    if isinstance(out, DiseaseDiscriminatorOutput):
        return out
    return DiseaseDiscriminatorOutput.model_validate(out)


def _method_output(state: AgentState) -> MethodDiscriminatorOutput:
    out = state["method_discriminator_output"]
    if isinstance(out, MethodDiscriminatorOutput):
        return out
    return MethodDiscriminatorOutput.model_validate(out)


def _candidate_names(output: object) -> list[str]:
    candidates = getattr(output, "candidates", None)
    if candidates is None and isinstance(output, dict):
        candidates = output.get("candidates")
    names: list[str] = []
    for candidate in candidates or []:
        name = getattr(candidate, "name", None)
        if name is None and isinstance(candidate, dict):
            name = candidate.get("name")
        if name:
            names.append(str(name))
    return names


def _normalize_candidate_text(text: str) -> str:
    normalized = text.strip().lower()
    for source, target in _TERM_NORMALIZATIONS:
        normalized = normalized.replace(source, target)
    return normalized.translate(str.maketrans("", "", _STRIP_CHARS))


def _match_candidate(user_text: str, output: object) -> str | None:
    normalized_text = _normalize_candidate_text(user_text)
    if not normalized_text:
        return None
    matches: list[str] = []
    for name in _candidate_names(output):
        normalized_name = _normalize_candidate_text(name)
        if normalized_name and (
            normalized_name in normalized_text or normalized_text in normalized_name
        ):
            matches.append(name)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return None
    canonical_matches = canonical_from_english_text(user_text)
    for name in _candidate_names(output):
        if name in canonical_matches:
            return name
    return None


def _required_fields(missing_feature: str | None) -> list[str]:
    return [canonical_key(missing_feature)] if missing_feature else []


def _interrupt_delta(interrupt: InterruptState, next_action: str) -> AgentState:
    assessment_patch, assessment_events = observe_clarification_interrupt(interrupt)
    return {
        **assessment_patch,
        "interrupt": interrupt,
        "interrupts": [interrupt],
        "awaiting_user_input": True,
        "next_action": next_action,
        "audit_log": [interrupt_trace_event(interrupt), *assessment_events],
    }


def _selection_trace(node_name: str, selected: str, state: AgentState) -> AuditEvent:
    return trace_event(
        node_name=node_name,
        kind="user_selection",
        title="Candidate selected by user",
        inputs={"latest_user_text": state.get("latest_user_text")},
        output={"selected": selected},
    )
