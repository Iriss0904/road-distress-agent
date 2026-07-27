"""LangGraph assembly for the post-refactor RAG runtime."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

import road_distress_agent.nodes.construction_arrangement_advisor as construction_advisor
from road_distress_agent.graph_routes import (
    route_after_disease_result,
    route_after_disease_selection,
    route_after_method_result,
    route_after_method_selection,
    route_after_safety_norm_retrieval,
    route_after_synthesis,
    route_critic,
    route_direct_answer,
    route_intent,
    route_kb_plan,
    route_off_intent,
    route_reconcile,
    route_top,
    route_weather,
)
from road_distress_agent.nodes.address_weather_loader import address_weather_loader
from road_distress_agent.nodes.answer_composer import answer_composer
from road_distress_agent.nodes.construction_tip_offer import construction_tip_offer
from road_distress_agent.nodes.detail_retriever_v2 import detail_retriever_v2
from road_distress_agent.nodes.diagnosis_reconcile import diagnosis_reconcile
from road_distress_agent.nodes.discriminator_flow import (
    disease_continue_discriminator,
    disease_result_router,
    disease_retriever,
    disease_selection_handler,
    method_result_router,
    method_retriever,
    method_selection_handler,
)
from road_distress_agent.nodes.disease_discriminator import disease_discriminator
from road_distress_agent.nodes.handle_off_intent import handle_off_intent
from road_distress_agent.nodes.intent_router import intent_router
from road_distress_agent.nodes.kb_answer_composer import kb_answer_composer
from road_distress_agent.nodes.kb_clarification_composer import kb_clarification_composer
from road_distress_agent.nodes.kb_direct_meta_answer import kb_direct_meta_answer
from road_distress_agent.nodes.kb_hop_retriever import kb_hop_retriever
from road_distress_agent.nodes.kb_planned_answer_composer import kb_planned_answer_composer
from road_distress_agent.nodes.kb_query_planner import kb_query_planner
from road_distress_agent.nodes.kb_query_rewriter import kb_query_rewriter
from road_distress_agent.nodes.kb_retriever import kb_retriever
from road_distress_agent.nodes.memory_writer import memory_writer
from road_distress_agent.nodes.method_discriminator import method_discriminator
from road_distress_agent.nodes.off_topic_refuser import off_topic_refuser
from road_distress_agent.nodes.parallel_context_loader import parallel_context_loader
from road_distress_agent.nodes.query_rewriter import disease_query_rewriter, query_rewriter
from road_distress_agent.nodes.safety_critic import safety_critic
from road_distress_agent.nodes.safety_norm_retriever import safety_norm_retriever
from road_distress_agent.nodes.safety_norm_rewriter import safety_norm_rewriter
from road_distress_agent.nodes.top_router import top_router
from road_distress_agent.nodes.weather_location_handler import weather_location_handler
from road_distress_agent.runtime_timing import timed_node
from road_distress_agent.state import AgentState
from road_distress_agent.subgraphs.vision import vision_subgraph


def build_graph(checkpointer: Any | None = None, store: Any | None = None) -> Any:
    """Assemble the C.1→C.2→D→E QueryTime pipeline with HITL support."""
    graph = StateGraph(AgentState)
    _add_nodes(graph)
    _wire_entry(graph)
    _wire_interrupt_gate(graph)
    _wire_diagnosis(graph)
    _wire_kb(graph)
    _wire_synthesis_weather_memory(graph)
    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=["intent_router"],
    )


def _add_nodes(graph: StateGraph) -> None:
    for node_name, node in _runtime_nodes():
        graph.add_node(node_name, timed_node(node_name, node))


def _runtime_nodes() -> tuple[tuple[str, Any], ...]:
    return (
        ("parallel_context_loader", parallel_context_loader),
        ("top_router", top_router),
        ("vision_subgraph", vision_subgraph),
        ("intent_router", intent_router),
        ("handle_off_intent", handle_off_intent),
        ("off_topic_refuser", off_topic_refuser),
        ("diagnosis_reconcile", diagnosis_reconcile),
        ("disease_selection_handler", disease_selection_handler),
        ("disease_query_rewriter", disease_query_rewriter),
        ("disease_retriever", disease_retriever),
        ("disease_continue_discriminator", disease_continue_discriminator),
        ("disease_discriminator", disease_discriminator),
        ("disease_result_router", disease_result_router),
        ("method_selection_handler", method_selection_handler),
        ("method_query_rewriter", query_rewriter),
        ("method_retriever", method_retriever),
        ("method_discriminator", method_discriminator),
        ("method_result_router", method_result_router),
        ("detail_retriever_v2", detail_retriever_v2),
        ("kb_query_planner", kb_query_planner),
        ("kb_query_rewriter", kb_query_rewriter),
        ("kb_retriever", kb_retriever),
        ("kb_direct_meta_answer", kb_direct_meta_answer),
        ("kb_answer_composer", kb_answer_composer),
        ("kb_hop_retriever", kb_hop_retriever),
        ("kb_planned_answer_composer", kb_planned_answer_composer),
        ("kb_clarification_composer", kb_clarification_composer),
        ("answer_composer", answer_composer),
        ("weather_location_handler", weather_location_handler),
        ("safety_norm_rewriter", safety_norm_rewriter),
        ("safety_norm_retriever", safety_norm_retriever),
        ("address_weather_loader", address_weather_loader),
        ("construction_arrangement_advisor", construction_advisor.construction_arrangement_advisor),
        ("construction_tip_offer", construction_tip_offer),
        ("safety_critic", safety_critic),
        ("memory_writer", memory_writer),
    )


def _wire_entry(graph: StateGraph) -> None:
    graph.add_edge(START, "parallel_context_loader")
    graph.add_edge("parallel_context_loader", "top_router")
    graph.add_conditional_edges(
        "top_router",
        route_top,
        {
            "vision_subgraph": "vision_subgraph",
            "diagnosis_reconcile": "diagnosis_reconcile",
            "kb_query_planner": "kb_query_planner",
            "kb_query_rewriter": "kb_query_rewriter",
            "kb_retriever": "kb_retriever",
            "kb_direct_meta_answer": "kb_direct_meta_answer",
            "weather_location_handler": "weather_location_handler",
            "off_topic_refuser": "off_topic_refuser",
        },
    )
    graph.add_edge("vision_subgraph", "diagnosis_reconcile")


def _wire_interrupt_gate(graph: StateGraph) -> None:
    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "disease_selection_handler": "disease_selection_handler",
            "method_selection_handler": "method_selection_handler",
            "weather_location_handler": "weather_location_handler",
            "handle_off_intent": "handle_off_intent",
        },
    )
    graph.add_conditional_edges(
        "handle_off_intent",
        route_off_intent,
        {"intent_router": "intent_router", "memory_writer": "memory_writer"},
    )
    _wire_direct_exit(graph, "off_topic_refuser")


def _wire_diagnosis(graph: StateGraph) -> None:
    _wire_disease_stages(graph)
    _wire_method_stages(graph)
    graph.add_edge("detail_retriever_v2", "answer_composer")


def _wire_disease_stages(graph: StateGraph) -> None:
    graph.add_conditional_edges(
        "diagnosis_reconcile",
        route_reconcile,
        {
            "disease_selection_handler": "disease_selection_handler",
            "method_selection_handler": "method_selection_handler",
            "detail_retriever_v2": "detail_retriever_v2",
            "memory_writer": "memory_writer",
        },
    )
    graph.add_conditional_edges(
        "disease_selection_handler",
        route_after_disease_selection,
        {
            "method_selection_handler": "method_selection_handler",
            "disease_query_rewriter": "disease_query_rewriter",
            "disease_continue_discriminator": "disease_continue_discriminator",
        },
    )
    graph.add_edge("disease_query_rewriter", "disease_retriever")
    graph.add_edge("disease_retriever", "disease_discriminator")
    graph.add_edge("disease_discriminator", "disease_result_router")
    graph.add_edge("disease_continue_discriminator", "disease_result_router")
    graph.add_conditional_edges(
        "disease_result_router",
        route_after_disease_result,
        {
            "method_selection_handler": "method_selection_handler",
            "intent_router": "intent_router",
        },
    )


def _wire_method_stages(graph: StateGraph) -> None:
    graph.add_conditional_edges(
        "method_selection_handler",
        route_after_method_selection,
        {
            "detail_retriever_v2": "detail_retriever_v2",
            "method_query_rewriter": "method_query_rewriter",
        },
    )
    graph.add_edge("method_query_rewriter", "method_retriever")
    graph.add_edge("method_retriever", "method_discriminator")
    graph.add_edge("method_discriminator", "method_result_router")
    graph.add_conditional_edges(
        "method_result_router",
        route_after_method_result,
        {
            "detail_retriever_v2": "detail_retriever_v2",
            "intent_router": "intent_router",
        },
    )


def _wire_kb(graph: StateGraph) -> None:
    graph.add_conditional_edges(
        "kb_query_planner",
        route_kb_plan,
        {
            "kb_retriever": "kb_retriever",
            "kb_hop_retriever": "kb_hop_retriever",
            "kb_clarification_composer": "kb_clarification_composer",
        },
    )
    graph.add_edge("kb_query_rewriter", "kb_retriever")
    graph.add_edge("kb_retriever", "kb_answer_composer")
    graph.add_edge("kb_hop_retriever", "kb_planned_answer_composer")
    _wire_direct_exit(graph, "kb_answer_composer")
    _wire_direct_exit(graph, "kb_planned_answer_composer")
    _wire_direct_exit(graph, "kb_clarification_composer")
    _wire_direct_exit(graph, "kb_direct_meta_answer")


def _wire_direct_exit(graph: StateGraph, node_name: str) -> None:
    graph.add_conditional_edges(
        node_name,
        route_direct_answer,
        {"intent_router": "intent_router", "memory_writer": "memory_writer"},
    )


def _wire_synthesis_weather_memory(graph: StateGraph) -> None:
    graph.add_conditional_edges(
        "answer_composer",
        route_after_synthesis,
        {
            "intent_router": "intent_router",
            "safety_critic": "safety_critic",
        },
    )

    # --- weather ---
    graph.add_conditional_edges(
        "weather_location_handler",
        route_weather,
        {
            "intent_router": "intent_router",
            "safety_norm_rewriter": "safety_norm_rewriter",
            "safety_critic": "safety_critic",
        },
    )
    graph.add_edge("safety_norm_rewriter", "safety_norm_retriever")
    graph.add_conditional_edges(
        "safety_norm_retriever",
        route_after_safety_norm_retrieval,
        {
            "address_weather_loader": "address_weather_loader",
            "construction_arrangement_advisor": "construction_arrangement_advisor",
        },
    )
    graph.add_edge("address_weather_loader", "construction_arrangement_advisor")
    graph.add_edge("construction_arrangement_advisor", "safety_critic")

    # --- critic + memory ---
    graph.add_conditional_edges(
        "safety_critic",
        route_critic,
        {
            "answer_composer": "answer_composer",
            "construction_tip_offer": "construction_tip_offer",
            "memory_writer": "memory_writer",
        },
    )
    graph.add_edge("construction_tip_offer", "intent_router")
    graph.add_edge("memory_writer", END)
