"""LangGraph assembly for the project-level delivery subgraph.

Topology::

    START -> archive_intake -> ledger_loader -> dedup_resolver
          -> [interrupt] dedup_confirm_gate -> delivery_supervisor
          -> cost_quantity_agent -> {report_agent, work_order_agent}  (parallel)
          -> compliance_critic -> delivery_packager -> project_memory_writer -> END

Cost runs first so the report can cite the estimate; report and work-order then
fan out in parallel, each writing a distinct state key (no write conflicts), and
``compliance_critic`` fans them back in.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from road_distress_agent.delivery.nodes.compliance import compliance_critic
from road_distress_agent.delivery.nodes.cost import cost_quantity_agent
from road_distress_agent.delivery.nodes.dedup import dedup_confirm_gate, dedup_resolver
from road_distress_agent.delivery.nodes.intake import archive_intake, ledger_loader
from road_distress_agent.delivery.nodes.packaging import delivery_packager, project_memory_writer
from road_distress_agent.delivery.nodes.report import report_agent
from road_distress_agent.delivery.nodes.supervisor import delivery_supervisor
from road_distress_agent.delivery.nodes.work_order import work_order_agent
from road_distress_agent.delivery.state import DeliveryState

_PARALLEL_SPECIALISTS = ("report_agent", "work_order_agent")


def build_delivery_graph(checkpointer: Any | None = None) -> Any:
    """Assemble the delivery subgraph with the archive-confirmation HITL gate."""
    graph = StateGraph(DeliveryState)

    graph.add_node("archive_intake", archive_intake)
    graph.add_node("ledger_loader", ledger_loader)
    graph.add_node("dedup_resolver", dedup_resolver)
    graph.add_node("dedup_confirm_gate", dedup_confirm_gate)
    graph.add_node("delivery_supervisor", delivery_supervisor)
    graph.add_node("report_agent", report_agent)
    graph.add_node("cost_quantity_agent", cost_quantity_agent)
    graph.add_node("work_order_agent", work_order_agent)
    graph.add_node("compliance_critic", compliance_critic)
    graph.add_node("delivery_packager", delivery_packager)
    graph.add_node("project_memory_writer", project_memory_writer)

    graph.add_edge(START, "archive_intake")
    graph.add_edge("archive_intake", "ledger_loader")
    graph.add_edge("ledger_loader", "dedup_resolver")
    graph.add_edge("dedup_resolver", "dedup_confirm_gate")
    graph.add_edge("dedup_confirm_gate", "delivery_supervisor")
    graph.add_edge("delivery_supervisor", "cost_quantity_agent")

    for specialist in _PARALLEL_SPECIALISTS:
        graph.add_edge("cost_quantity_agent", specialist)
        graph.add_edge(specialist, "compliance_critic")

    graph.add_edge("compliance_critic", "delivery_packager")
    graph.add_edge("delivery_packager", "project_memory_writer")
    graph.add_edge("project_memory_writer", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["dedup_confirm_gate"])
