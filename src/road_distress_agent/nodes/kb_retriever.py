"""Qdrant retrieval for pure knowledge-base QA."""

from __future__ import annotations

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_assessment import EvidenceStatus
from road_distress_agent.evidence_observation import (
    evidence_observation_event,
    observe_simple_kb,
)
from road_distress_agent.retrieval.evidence import EvidenceRetrievalOptions, retrieve_evidence
from road_distress_agent.retrieval.query_features import detect_query_features
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    KbQueryPlan,
    QueryPlan,
)


def _plan(state: AgentState) -> QueryPlan:
    raw = state.get("kb_query_plan")
    if isinstance(raw, QueryPlan):
        return raw
    if isinstance(raw, dict):
        return QueryPlan.model_validate(raw)
    query = state.get("latest_user_text") or ""
    return QueryPlan(queries=[query], filters={})


def kb_retriever(state: AgentState) -> AgentState:
    """Retrieve evidence for KB QA; failures are intentionally not swallowed."""
    plan = _plan(state)
    query = _query(plan)
    chunks, attempts, traces = retrieve_evidence(
        EvidenceRetrievalOptions(
            node_name="kb_retriever",
            title="KB evidence search",
            query=query,
            filters=plan.filters,
            stage_goal="普通知识问答：选择能直接回答术语定义、方法说明或规范依据的证据。",
            explicit_clause_ids=_explicit_clause_ids(state),
        )
    )
    observation = observe_simple_kb(
        chunks,
        _optional_v2_plan(state),
        original_user_text=state.get("latest_user_text"),
        retrieval_attempts=attempts,
    )
    return {
        "kb_retrieved_chunks": chunks,
        "evidence_assessment": observation.assessment,
        "retrieval_attempts": attempts,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "compose_kb_answer",
        "audit_log": [
            *traces,
            AuditEvent(
                node_name="kb_retriever",
                message="Retrieved knowledge-base evidence.",
                metadata={
                    "chunk_count": len(chunks),
                    "query": query,
                },
            ),
            evidence_observation_event(
                node_name="kb_retriever",
                observation=observation,
                affects_runtime_behavior=_is_no_kb(observation.assessment),
            ),
        ],
    }


def _query(plan: QueryPlan) -> str:
    if not plan.queries:
        raise ValueError("kb_retriever requires kb_query_plan.queries to contain one query.")
    return plan.queries[0]


def _optional_v2_plan(state: AgentState) -> KbQueryPlan | None:
    raw = state.get("kb_query_plan_v2")
    if isinstance(raw, KbQueryPlan):
        return raw
    if isinstance(raw, dict):
        return KbQueryPlan.model_validate(raw)
    return None


def _explicit_clause_ids(state: AgentState) -> tuple[str, ...]:
    return detect_query_features(state.get("latest_user_text") or "", {}).clause_ids


def _is_no_kb(assessment: object | None) -> bool:
    return getattr(assessment, "status", None) is EvidenceStatus.NO_KB_EVIDENCE
