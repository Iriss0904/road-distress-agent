"""Reranked procedure and acceptance retrieval after method lock."""

from __future__ import annotations

from road_distress_agent.detail_retrieval_execution import (
    DetailPipelines,
    detail_retrieval_parallel_enabled,
    run_detail_pipelines,
)
from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_observation import (
    evidence_observation_event,
    observe_detail_evidence,
)
from road_distress_agent.nodes.detail_query_context import (
    detail_queries,
    distress_description,
)
from road_distress_agent.retrieval.evidence import (
    EvidenceRetrievalOptions,
    retrieve_evidence,
)
from road_distress_agent.speculative_prefetch import resolve_detail_prefetch
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    Citation,
    EvidenceBundle,
    RetrievalAttempt,
    RetrievedChunk,
)
from road_distress_agent.tools.qdrant_rag import (
    QdrantRAGTool,
    get_default_qdrant_rag_tool,
)
from road_distress_agent.tracing import retrieval_trace_event

MISSING_STEP_INDEX = 9999
DETAIL_NODE_NAME = "detail_retriever_v2"
PROCEDURE_TITLE = "Procedure detail search"
ACCEPTANCE_TITLE = "Acceptance criteria search"
PROCEDURE_ROLE = "construction_step"
ACCEPTANCE_ROLE = "acceptance_criteria"


def _get_method(state: AgentState) -> str:
    if m := state.get("chosen_method"):
        return str(m)
    selection = state.get("candidate_selection")
    if selection and selection.selected_name:
        return str(selection.selected_name)
    candidates = state.get("solution_candidates") or []
    if candidates:
        return str(candidates[0].name)
    raise ValueError("detail_retriever_v2 requires a locked or user-specified method.")


def _sort_steps(chunks: list[Citation]) -> list[Citation]:
    def _key(c: Citation) -> tuple[int, int]:
        try:
            step = int(c.step_index) if c.step_index is not None else MISSING_STEP_INDEX
        except (TypeError, ValueError):
            step = MISSING_STEP_INDEX
        return step, (c.rank or MISSING_STEP_INDEX)

    return sorted(chunks, key=_key)


def _dedupe(chunks: list[Citation]) -> list[Citation]:
    seen: set[str] = set()
    unique: list[Citation] = []
    for c in chunks:
        key = c.chunk_id or f"{c.source_doc}:{c.clause_id}:{c.rank}"
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in chunks:
        if isinstance(chunk, Citation):
            citations.append(chunk)
            continue
        citations.append(Citation.model_validate(chunk.model_dump()))
    return citations


def _retrieve_reranked_evidence(
    *,
    query: str,
    title: str,
    stage_goal: str,
    semantic_role: str,
) -> tuple[list[Citation], list[RetrievalAttempt], list[AuditEvent]]:
    chunks, attempts, traces = retrieve_evidence(
        EvidenceRetrievalOptions(
            node_name=DETAIL_NODE_NAME,
            title=title,
            query=query,
            filters={"semantic_role": semantic_role},
            stage_goal=stage_goal,
            preserve_semantic_role_filter=True,
        )
    )
    return _citations(chunks), attempts, traces


def _expand_sequential_groups(
    *,
    chunks: list[Citation],
    tool: QdrantRAGTool,
    attempts: list[RetrievalAttempt],
) -> tuple[list[Citation], list[AuditEvent]]:
    """Fetch ALL steps from each sequential_group_id that appears in the seed chunks.

    Uses scroll (not vector search) so the complete step sequence is guaranteed
    regardless of top_k — e.g. a 9-step group won't be truncated to 5.
    """
    expanded = list(chunks)
    trace_events: list[AuditEvent] = []
    seen_groups: set[str] = set()
    for chunk in chunks:
        group = chunk.sequential_group_id
        if not group or group in seen_groups:
            continue
        seen_groups.add(group)
        pulled = tool.fetch_by_sequential_group(group)
        expanded.extend(pulled)
        attempts.append(
            RetrievalAttempt(
                tool_name="QdrantRAGTool.fetch_by_sequential_group",
                query=group,
                filters={"sequential_group_id": group},
                citation_count=len(pulled),
            )
        )
        trace_events.append(
            retrieval_trace_event(
                node_name="qdrant_rag",
                title="Sequential group expansion",
                query=group,
                filters={"sequential_group_id": group},
                chunks=pulled,
                tool=tool,
                metadata={"runner": "detail_retriever_v2"},
            )
        )
    return expanded, trace_events


def _retrieve_procedure_chunks(
    *,
    tool: QdrantRAGTool,
    method: str,
    query: str,
) -> tuple[list[Citation], list[RetrievalAttempt], list[AuditEvent]]:
    chunks, attempts, traces = _retrieve_reranked_evidence(
        query=query,
        title=PROCEDURE_TITLE,
        stage_goal=_procedure_stage_goal(method),
        semantic_role=PROCEDURE_ROLE,
    )
    procedure_raw, group_traces = _expand_sequential_groups(
        chunks=chunks,
        tool=tool,
        attempts=attempts,
    )
    procedure_chunks = _sort_steps(_dedupe(procedure_raw))
    return procedure_chunks, attempts, [*traces, *group_traces]


def _retrieve_acceptance_chunks(
    method: str,
    query: str,
) -> tuple[list[Citation], list[RetrievalAttempt], list[AuditEvent]]:
    chunks, attempts, traces = _retrieve_reranked_evidence(
        query=query,
        title=ACCEPTANCE_TITLE,
        stage_goal=_acceptance_stage_goal(method),
        semantic_role=ACCEPTANCE_ROLE,
    )
    return _dedupe(chunks), attempts, traces


def _procedure_stage_goal(method: str) -> str:
    return (
        f"详情阶段：检索与已选方法「{method}」正向对应的施工步骤证据；"
        "优先选择能直接支撑具体操作流程的原文，排除与已选方法不同的相邻工艺证据。"
    )


def _acceptance_stage_goal(method: str) -> str:
    return (
        f"详情阶段：检索与已选方法「{method}」施工后对应的质量检查、验收标准、"
        "检验项目、允许偏差和检验方法证据；排除仅适用于其他工艺的冲突指标。"
    )


def _build_bundles(
    query_steps: str,
    procedure_chunks: list[Citation],
    *,
    query_accept: str,
    acceptance_chunks: list[Citation],
) -> dict[str, EvidenceBundle]:
    return {
        "construction_steps": EvidenceBundle(
            query=query_steps,
            citations=procedure_chunks,
            sufficient=bool(procedure_chunks),
            gap_reason=None if procedure_chunks else "no_procedure_chunks_retrieved",
        ),
        "acceptance_criteria": EvidenceBundle(
            query=query_accept,
            citations=acceptance_chunks,
            sufficient=bool(acceptance_chunks),
            gap_reason=None if acceptance_chunks else "no_acceptance_chunks_retrieved",
        ),
    }


def _audit_log(
    *,
    method: str,
    distress: str,
    procedure_chunks: list[Citation],
    acceptance_chunks: list[Citation],
    retrieval_traces: list[AuditEvent],
) -> list[AuditEvent]:
    return [
        *retrieval_traces,
        AuditEvent(
            node_name="detail_retriever_v2",
            message="Step D: retrieved procedure steps and acceptance criteria.",
            metadata={
                "method": method,
                "distress": distress,
                "procedure_count": len(procedure_chunks),
                "acceptance_count": len(acceptance_chunks),
            },
        ),
    ]


def _run_detail_retriever_v2(state: AgentState) -> AgentState:
    """Step D: retrieve procedure steps and acceptance criteria for the locked method."""
    tool = get_default_qdrant_rag_tool()
    method = _get_method(state)
    distress = distress_description(state)
    query_steps, query_accept = detail_queries(method, distress)
    procedure_result, acceptance_result = run_detail_pipelines(
        DetailPipelines(
            procedure=lambda: _retrieve_procedure_chunks(
                tool=tool, method=method, query=query_steps
            ),
            acceptance=lambda: _retrieve_acceptance_chunks(method, query_accept),
        ),
        parallel=detail_retrieval_parallel_enabled(),
    )
    procedure_chunks, procedure_attempts, procedure_traces = procedure_result
    acceptance_chunks, acceptance_attempts, acceptance_traces = acceptance_result
    bundles = _build_bundles(
        query_steps,
        procedure_chunks,
        query_accept=query_accept,
        acceptance_chunks=acceptance_chunks,
    )
    observation = observe_detail_evidence(procedure_chunks, acceptance_chunks)
    return {
        "procedure_chunks": procedure_chunks,
        "acceptance_chunks": acceptance_chunks,
        "detail_evidence_bundles": bundles,
        "evidence_assessment": observation.assessment,
        "retrieval_attempts": [*procedure_attempts, *acceptance_attempts],
        "errors": [],
        "phase": WorkflowPhase.DETAIL,
        "next_action": "compose_answer",
        "audit_log": [
            *_audit_log(
                method=method,
                distress=distress,
                procedure_chunks=procedure_chunks,
                acceptance_chunks=acceptance_chunks,
                retrieval_traces=[*procedure_traces, *acceptance_traces],
            ),
            evidence_observation_event(
                node_name=DETAIL_NODE_NAME,
                observation=observation,
            ),
        ],
    }


def detail_retriever_v2(state: AgentState) -> AgentState:
    """Reuse an exact B-09 future or execute the same production implementation."""
    return resolve_detail_prefetch(state, _run_detail_retriever_v2)
