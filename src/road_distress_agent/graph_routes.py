"""Conditional-edge routing helpers for the LangGraph runtime."""

from __future__ import annotations

from typing import Literal

from road_distress_agent.kb_planning_flags import kb_query_planning_enabled
from road_distress_agent.nodes.discriminator_flow import should_continue_disease
from road_distress_agent.resume import (
    DISEASE_CLARIFICATION,
    DISEASE_SELECTION,
    METHOD_CLARIFICATION,
    METHOD_SELECTION,
    WEATHER_LOCATION_REQUEST,
)
from road_distress_agent.state import AgentState, FinalAnswer


def route_top(
    state: AgentState,
) -> Literal[
    "vision_subgraph",
    "diagnosis_reconcile",
    "kb_query_planner",
    "kb_query_rewriter",
    "kb_retriever",
    "kb_direct_meta_answer",
    "weather_location_handler",
    "off_topic_refuser",
]:
    if _waiting_for_weather_location(state):
        return "weather_location_handler"
    route = state.get("top_route") or {}
    action = route.get("action")
    if action == "kb_aside":
        return _route_kb_aside(route)
    if action == "off_topic":
        return "off_topic_refuser"
    if action != "diagnosis_proceed":
        raise ValueError(f"Unknown top route action: {action!r}.")
    if _has_image(state):
        return "vision_subgraph"
    return "diagnosis_reconcile"


def route_intent(
    state: AgentState,
) -> Literal[
    "disease_selection_handler",
    "method_selection_handler",
    "weather_location_handler",
    "handle_off_intent",
]:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    if kind in (DISEASE_CLARIFICATION, DISEASE_SELECTION):
        return "disease_selection_handler"
    if kind in (METHOD_CLARIFICATION, METHOD_SELECTION):
        return "method_selection_handler"
    if kind == WEATHER_LOCATION_REQUEST:
        return "weather_location_handler"
    return "handle_off_intent"


def route_after_disease_selection(
    state: AgentState,
) -> Literal[
    "method_selection_handler",
    "disease_continue_discriminator",
    "disease_query_rewriter",
]:
    if state.get("next_action") == "method_phase":
        return "method_selection_handler"
    if should_continue_disease(state):
        return "disease_continue_discriminator"
    return "disease_query_rewriter"


def route_after_disease_result(
    state: AgentState,
) -> Literal["method_selection_handler", "intent_router"]:
    if state.get("next_action") == "method_phase":
        return "method_selection_handler"
    return "intent_router"


def route_after_method_selection(
    state: AgentState,
) -> Literal["detail_retriever_v2", "method_query_rewriter"]:
    if state.get("next_action") == "detail_phase":
        return "detail_retriever_v2"
    return "method_query_rewriter"


def route_after_method_result(
    state: AgentState,
) -> Literal["detail_retriever_v2", "intent_router"]:
    if state.get("next_action") == "detail_phase":
        return "detail_retriever_v2"
    return "intent_router"


def route_after_synthesis(
    state: AgentState,
) -> Literal["intent_router", "safety_critic"]:
    if state.get("critic_retry_count", 0) > 0:
        return "safety_critic"
    if state.get("interrupt") is not None:
        return "intent_router"
    return "safety_critic"


def route_weather(
    state: AgentState,
) -> Literal["intent_router", "safety_norm_rewriter", "safety_critic"]:
    route = state.get("weather_route")
    if route in {"ask_weather_location", "ask_road_class"}:
        return "intent_router"
    if route in {"load_weather", "safety_norm_only"}:
        return "safety_norm_rewriter"
    return "safety_critic"


def route_after_safety_norm_retrieval(
    state: AgentState,
) -> Literal["address_weather_loader", "construction_arrangement_advisor"]:
    if state.get("address_context") is not None:
        return "address_weather_loader"
    return "construction_arrangement_advisor"


def route_critic(
    state: AgentState,
) -> Literal["answer_composer", "construction_tip_offer", "memory_writer"]:
    review = state.get("safety_review")
    retry_count = state.get("critic_retry_count", 0)
    if review and review.passed:
        if _should_offer_construction_tip(state):
            return "construction_tip_offer"
        return "memory_writer"
    if retry_count <= 1:
        return "answer_composer"
    return "memory_writer"


def route_off_intent(state: AgentState) -> Literal["intent_router", "memory_writer"]:
    if state.get("interrupt") is not None:
        return "intent_router"
    return "memory_writer"


def route_reconcile(
    state: AgentState,
) -> Literal[
    "disease_selection_handler",
    "method_selection_handler",
    "detail_retriever_v2",
    "memory_writer",
]:
    result = state.get("reconcile_result") or {}
    stage = result.get("next_stage")
    if stage == "method":
        return "method_selection_handler"
    if stage == "detail":
        return "detail_retriever_v2"
    if stage == "done":
        return "memory_writer"
    return "disease_selection_handler"


def route_direct_answer(state: AgentState) -> Literal["intent_router", "memory_writer"]:
    if state.get("interrupt") is not None:
        return "intent_router"
    return "memory_writer"


def route_kb_plan(
    state: AgentState,
) -> Literal[
    "kb_retriever",
    "kb_hop_retriever",
    "kb_clarification_composer",
]:
    plan_type = state.get("kb_plan_type")
    if plan_type == "single_hop":
        return "kb_retriever"
    if plan_type in {"chain", "compare", "evidence_composition"}:
        return "kb_hop_retriever"
    if plan_type == "clarification":
        return "kb_clarification_composer"
    raise ValueError(f"Unknown KB plan type: {plan_type!r}.")


def _has_image(state: AgentState) -> bool:
    attachments = state.get("latest_attachments") or []
    return any(a.media_type.startswith("image/") for a in attachments)


def _route_kb_aside(
    route: dict[str, object],
) -> Literal[
    "kb_query_planner",
    "kb_query_rewriter",
    "kb_retriever",
    "kb_direct_meta_answer",
]:
    if not kb_query_planning_enabled():
        return "kb_query_rewriter"
    rag_tier = route.get("rag_tier")
    if rag_tier == "single_hop":
        return "kb_retriever"
    if rag_tier == "single_hop_needs_rewrite":
        return "kb_query_rewriter"
    if rag_tier == "multi_hop":
        return "kb_query_planner"
    if rag_tier == "no_retrieval":
        return "kb_direct_meta_answer"
    raise ValueError(f"Unknown KB rag_tier: {rag_tier!r}.")


def _waiting_for_weather_location(state: AgentState) -> bool:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    return kind == WEATHER_LOCATION_REQUEST


def _should_offer_construction_tip(state: AgentState) -> bool:
    route = state.get("top_route") or {}
    if route.get("action") != "diagnosis_proceed":
        return False
    if state.get("construction_tip_offered"):
        return False
    if state.get("weather_advice") is not None:
        return False
    return isinstance(state.get("final_answer"), FinalAnswer)
