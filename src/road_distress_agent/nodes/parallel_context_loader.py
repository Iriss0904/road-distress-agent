"""Load per-user memory and empty environmental context slots."""

from __future__ import annotations

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.state import AddressContext, AgentState, AuditEvent, WeatherContext
from road_distress_agent.tools.memory import make_memory_tool


def parallel_context_loader(state: AgentState) -> AgentState:
    """Load long-term memory before routing without feeding it to retrieval."""
    loaded_memory = make_memory_tool().load(state["user_id"])
    return {
        "loaded_memory": loaded_memory,
        "address_context": AddressContext(status="not_requested"),
        "weather_context": WeatherContext(status="not_requested"),
        "phase": WorkflowPhase.CONTEXT,
        "next_action": "rewrite_query",
        "audit_log": [
            AuditEvent(
                node_name="parallel_context_loader",
                message="Loaded persistent user memory and initialized context slots.",
                metadata={
                    "has_preferences": bool(loaded_memory.user_preferences),
                    "has_regional_context": bool(loaded_memory.regional_context),
                    "has_resource_constraints": bool(loaded_memory.resource_constraints),
                    "case_summary_count": len(loaded_memory.case_summaries),
                },
            )
        ],
    }
