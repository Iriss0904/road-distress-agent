"""State contract for the baseline road-maintenance RAG graph.

This module follows the five-domain boundary in ``langgraph_state.py`` while
keeping a small compatibility surface for existing draft nodes. The canonical
business state is intentionally compact:

A. conversation control
B. accumulated user facts
C. per-turn retrieval results
D. discriminator decision
E. final delivery

Weather / construction-tip helpers are deferred optimization modules. Their
models stay in this file only so older nodes can still import them, but those
fields are not part of the baseline ``AgentState``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict, TypeVar
from uuid import uuid4

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.errors import ErrorCategory, ErrorInfo
from road_distress_agent.evidence_assessment import EvidenceAssessment
from road_distress_agent.localization import Locale

T = TypeVar("T")

Stage = Literal[
    "STAGE_1_DISTRESS_ID",
    "STAGE_2_METHOD_SELECT",
    "STAGE_3_PROCEDURE",
    "FOLLOW_UP",
    "END",
]
GlobalSignal = Literal["RESET", "CHANGE_DEFECT", "CORRECT_FEATURE"]
RefusalType = Literal["off_topic", "no_kb_evidence"]
RetrievalChannel = Literal["qdrant", "bm25", "clause"]


def append_list(left: list[T] | None, right: list[T] | None) -> list[T]:
    """Reducer for append-only audit/history fields."""
    if not left:
        return list(right or [])
    if not right:
        return list(left)
    return [*left, *right]


class AttachmentRef(BaseModel):
    """User-provided media reference.

    Images are optional context for Stage 1. They are not the source of truth;
    extracted facts must still land in the accumulated user-fact fields.
    """

    uri: str
    media_type: str = "image/jpeg"
    description: str | None = None
    attachment_id: str = Field(default_factory=lambda: f"att-{uuid4().hex[:12]}")
    sha256: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SceneContext(BaseModel):
    """Vision-derived scene notes, kept separate from user-confirmed facts."""

    scene_description: str
    source_attachment_id: str | None = None
    source_uri: str | None = None
    structured_facts: dict[str, Any] = Field(default_factory=dict)
    uncertainty_notes: list[str] = Field(default_factory=list)


class FieldChange(BaseModel):
    """Human-readable diff for user fact corrections."""

    field_path: str
    old_value: Any = None
    new_value: Any = None
    reason: str


class DistressInput(BaseModel):
    """Compatibility wrapper for accumulated user facts.

    Canonical state stores these same facts at top level:
    ``material``, ``defect_category``, ``defect_subtype`` and
    ``known_features``. Existing draft nodes still exchange a ``distress``
    object, so this model remains as a transition bridge.
    """

    model_config = ConfigDict(extra="ignore")

    material: str | None = None
    defect_category: str | None = None
    defect_subtype: str | None = None
    known_features: dict[str, Any] = Field(default_factory=dict)
    raw_user_text: str | None = None
    scene_description: str | None = None

    @field_validator("known_features", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return {} if value is None else value


class RetrievedChunk(BaseModel):
    """One retrieved standard/spec chunk after contextual ingestion.

    Canonical fields mirror the new design:
    ``chunk_id``, ``clause_id``, ``text``, ``score``, ``cross_refs``,
    ``sequential_group_id`` and ``step_index``.

    A few citation-style aliases remain because existing RAG/adaptor code still
    passes ``citation_id`` / ``snippet`` / ``source_clause``.
    """

    model_config = ConfigDict(extra="ignore")

    chunk_id: str | None = None
    clause_id: str | None = None
    text: str = ""
    score: float | None = None
    cross_refs: list[str] = Field(default_factory=list)
    sequential_group_id: str | None = None
    step_index: int | None = None

    heading_path: list[str] = Field(default_factory=list)
    context_prefix: str | None = None
    table_raw: Any = None
    image_paths: list[str] = Field(default_factory=list)

    document_id: str | None = None
    title: str | None = None
    source_doc: str | None = None
    source_pages: str | None = None
    source_standard: str | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)

    # Transition aliases used by the earlier draft runtime.
    citation_id: str | None = None
    snippet: str | None = None
    source_clause: str | None = None
    similarity: float | None = None
    rank: int | None = None

    @field_validator("cross_refs", "heading_path", "image_paths", mode="before")
    @classmethod
    def _none_to_list(cls, value: Any) -> Any:
        return [] if value is None else value

    @model_validator(mode="after")
    def _harmonize_aliases(self) -> RetrievedChunk:
        if self.chunk_id is None and self.citation_id:
            self.chunk_id = self.citation_id
        if self.citation_id is None and self.chunk_id:
            self.citation_id = self.chunk_id

        if not self.text and self.snippet:
            self.text = self.snippet
        if self.snippet is None and self.text:
            self.snippet = self.text

        if self.clause_id is None and self.source_clause:
            self.clause_id = self.source_clause
        if self.source_clause is None and self.clause_id:
            self.source_clause = self.clause_id

        if self.score is None and self.similarity is not None:
            self.score = self.similarity
        if self.similarity is None and self.score is not None:
            self.similarity = self.score

        metadata = self.annotations or {}
        if self.clause_id is None and metadata.get("clause_id"):
            self.clause_id = str(metadata["clause_id"])
            self.source_clause = self.source_clause or self.clause_id
        if not self.cross_refs and isinstance(metadata.get("cross_refs"), list):
            self.cross_refs = [str(item) for item in metadata["cross_refs"]]
        if self.sequential_group_id is None and metadata.get("sequential_group_id"):
            self.sequential_group_id = str(metadata["sequential_group_id"])
        if self.step_index is None and metadata.get("step_index") is not None:
            try:
                self.step_index = int(metadata["step_index"])
            except (TypeError, ValueError):
                pass
        if not self.heading_path and isinstance(metadata.get("heading_path"), list):
            self.heading_path = [str(item) for item in metadata["heading_path"]]
        if self.context_prefix is None and metadata.get("context_prefix"):
            self.context_prefix = str(metadata["context_prefix"])
        return self


class Citation(RetrievedChunk):
    """Backward-compatible name for retrieved evidence chunks."""


class ReferenceItem(BaseModel):
    """Deduplicated source record rendered as an inline citation chip."""

    ref_id: str
    title: str
    source_doc: str | None = None
    source_standard: str | None = None
    source_clause: str | None = None
    source_pages: str | None = None
    snippet: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    evidence_quality: float | None = None


class EvidenceBundle(BaseModel):
    """Named group of chunks produced by one retrieval purpose."""

    query: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    sufficient: bool = False
    gap_reason: str | None = None


class QueryPlan(BaseModel):
    """Transition object for nodes that still issue one or more RAG queries."""

    queries: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)


KbPlanType = Literal[
    "single_hop",
    "chain",
    "compare",
    "evidence_composition",
    "clarification",
]
KbAnswerMode = Literal[
    "direct",
    "chain_reasoning",
    "compare_table",
    "procedure",
    "clarification",
]
KbCitationPolicy = Literal["per_claim", "per_object", "per_slot", "per_branch"]
DEFAULT_KB_HOP_FINAL_TOP_K = 3
# Executor-side retrieval budget (bounds real retrieve_evidence calls), NOT a
# schema correctness invariant. Sized for the largest real KB task — a highway
# work-control area has 6 zones (警告/上游过渡/缓冲/工作/下游过渡/终止), so a
# per-zone evidence_composition plan legitimately needs ~6 hops. The executor
# truncates to this budget and logs it; the data model no longer hard-raises on
# hop count (a well-formed 7-hop plan is expensive, not invalid).
MAX_KB_PLAN_HOPS = 8


class KbHop(BaseModel):
    hop_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    stage_goal: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    object_label: str | None = None
    slot: str | None = None
    final_top_k: int = Field(default=DEFAULT_KB_HOP_FINAL_TOP_K, ge=1)


class KbSubQuestion(BaseModel):
    sub_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    hop_ids: list[str] = Field(default_factory=list)


class KbConditionalBranch(BaseModel):
    condition: str = Field(min_length=1)
    answer_hint: str = Field(min_length=1)


class KbClarification(BaseModel):
    missing_user_slots: list[str] = Field(default_factory=list)
    branches: list[KbConditionalBranch] = Field(default_factory=list)
    assumption: str | None = None


class KbQueryPlan(BaseModel):
    plan_type: KbPlanType
    original_question: str = Field(min_length=1)
    rewritten_main_query: str = Field(min_length=1)
    subquestions: list[KbSubQuestion] = Field(default_factory=list)
    hops: list[KbHop] = Field(default_factory=list)
    evidence_slots: list[str] = Field(default_factory=list)
    missing_user_slots: list[str] = Field(default_factory=list)
    clarification: KbClarification | None = None
    answer_mode: KbAnswerMode
    citation_policy: KbCitationPolicy = "per_claim"
    confidence: float = Field(ge=0, le=1)
    decision_reason: str | None = None

    @model_validator(mode="after")
    def _check_plan_shape(self) -> KbQueryPlan:
        # Only correctness invariants here. The hop *count* upper bound is a
        # resource budget enforced by the executor (kb_hop_retriever truncates),
        # not a validity rule — a well-formed 7-hop plan must not crash the turn.
        if self.plan_type == "chain" and len(self.hops) < 2:
            raise ValueError("chain plans require at least two independent hops.")
        if self.plan_type == "compare" and len(self.hops) < 2:
            raise ValueError("compare plans require at least two object hops.")
        if self.plan_type == "evidence_composition" and not self.evidence_slots:
            raise ValueError("evidence_composition plans require evidence_slots.")
        if self.plan_type == "evidence_composition" and not self.hops:
            raise ValueError("evidence_composition plans require slot hops.")
        return self


class KbHopResult(BaseModel):
    hop_id: str
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class RetrievalAttempt(BaseModel):
    tool_name: str
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    citation_count: int = Field(default=0, ge=0)
    channel: RetrievalChannel | None = None
    requested_clause_ids: tuple[str, ...] = ()


class MethodCandidate(BaseModel):
    """One method candidate emitted by the Discriminator."""

    model_config = ConfigDict(extra="ignore")

    method_name: str | None = None
    display_name: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str = ""
    source: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    selectable: bool = True

    # Transition aliases used by the UI/candidate handler draft.
    candidate_id: str | None = None
    name: str | None = None
    rationale: str | None = None
    rough_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _harmonize_aliases(self) -> MethodCandidate:
        if self.method_name is None:
            self.method_name = self.candidate_id or self.name
        if self.display_name is None:
            self.display_name = self.name or self.method_name
        if self.name is None:
            self.name = self.display_name or self.method_name
        if self.candidate_id is None:
            self.candidate_id = self.method_name

        if not self.reason and self.rationale:
            self.reason = self.rationale
        if self.rationale is None:
            self.rationale = self.reason

        if "confidence" not in self.__pydantic_fields_set__ and self.rough_confidence is not None:
            self.confidence = self.rough_confidence
        if self.rough_confidence is None:
            self.rough_confidence = self.confidence
        return self


class SolutionCandidate(MethodCandidate):
    """Backward-compatible candidate name used by current UI handlers."""


class RankedMethod(MethodCandidate):
    """Backward-compatible discriminator output item name."""


class DiscriminatorOutput(BaseModel):
    discriminating_feature: str | None = None
    question_to_user: str | None = None
    ranked_methods: list[RankedMethod] = Field(default_factory=list)
    need_more_info: bool = False
    reasoning: str | None = None


class DiseaseCandidate(BaseModel):
    """One ranked candidate in the top-3 selection list presented to the user."""

    name: str  # canonical disease name, e.g. "坑槽"
    description: str  # one-sentence excerpt from the standard
    confidence: float = Field(ge=0.0, le=1.0)


class DiseaseDiscriminatorOutput(BaseModel):
    """Output of the Phase C.1 disease identification discriminator.

    Three mutually exclusive states:

    1. sufficient=False, candidates=[], clarifying_question set
       → Need one more fact; ask the user (only allowed when clarification_attempts == 0).

    2. sufficient=False, candidates=[top-3 list]
       → Present ranked candidates for user to pick before C.2.

    3. sufficient=True
       -> Schema compatibility only. Runtime rejects direct locks in diagnosis flow.
    """

    sufficient: bool
    # State 2 — ask one follow-up question
    missing_feature: str | None = None  # machine-readable key, e.g. "crack_width_mm"
    clarifying_question: str | None = None  # natural-language question for the user

    # State 1 — identified
    identified_disease: str | None = None  # canonical disease name, e.g. "坑槽"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None  # one sentence citing evidence thresholds

    # State 3 — present top-3 candidates for user selection
    candidates: list[DiseaseCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_invariants(self) -> DiseaseDiscriminatorOutput:
        has_question = self.clarifying_question is not None or self.missing_feature is not None
        has_candidates = len(self.candidates) > 0
        if not self.sufficient:
            if not has_question and not has_candidates:
                raise ValueError(
                    "sufficient=False requires either clarifying_question/missing_feature "
                    "(ask-one-question state) or candidates (selection state)."
                )
            if has_question and has_candidates:
                raise ValueError(
                    "sufficient=False must be either ask-question OR candidates, not both."
                )
        else:
            if self.identified_disease is None:
                raise ValueError("sufficient=True requires identified_disease to be set.")
        return self


class MethodOption(BaseModel):
    """One ranked method candidate presented to the user for confirmation."""

    name: str  # canonical method name, e.g. "局部挖补法"
    reason: str  # one sentence citing why this method fits
    confidence: float = Field(ge=0.0, le=1.0)


class MethodDiscriminatorOutput(BaseModel):
    """Output of the Phase C.2 method selection discriminator.

    Three mutually exclusive states (same pattern as DiseaseDiscriminatorOutput):

    1. sufficient=True, candidates=[]
       -> Schema compatibility only. Runtime rejects direct locks in diagnosis flow.

    2. sufficient=False, candidates=[], clarifying_question set
       -> Need one more fact; ask the user (only when clarification_attempts == 0).

    3. sufficient=False, candidates=[1-3 item list]
       -> Present ranked methods for user to confirm or pick.
    """

    sufficient: bool
    # State 2 — ask one follow-up question
    missing_feature: str | None = None  # machine-readable key, e.g. "base_exposed"
    clarifying_question: str | None = None

    # State 1 — identified
    identified_method: str | None = None  # canonical method name
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None  # one sentence citing evidence conditions

    # State 3 — present top-3 for user selection
    candidates: list[MethodOption] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_invariants(self) -> MethodDiscriminatorOutput:
        has_question = self.clarifying_question is not None or self.missing_feature is not None
        has_candidates = len(self.candidates) > 0
        if not self.sufficient:
            if not has_question and not has_candidates:
                raise ValueError(
                    "sufficient=False requires clarifying_question/missing_feature or candidates."
                )
            if has_question and has_candidates:
                raise ValueError(
                    "sufficient=False must be either ask-question OR candidates, not both."
                )
        else:
            if self.identified_method is None:
                raise ValueError("sufficient=True requires identified_method to be set.")
        return self


class CandidateSelection(BaseModel):
    status: Literal["not_requested", "awaiting_user", "selected", "invalid_selection"] = (
        "not_requested"
    )
    selected_candidate_id: str | None = None
    selected_name: str | None = None
    selected_at: str | None = None
    action: Literal[
        "choose_existing",
        "auto_select",
        "reject_all",
        "modify_constraints",
    ] = "choose_existing"


class FinalAnswer(BaseModel):
    """Structured Stage 3 delivery.

    The baseline answer is procedure + acceptance + citations. Weather-related
    fields are compatibility-only and should stay unused until the optimization
    module is deliberately reintroduced.
    """

    summary: str
    diagnosis: str | None = None
    method_selection: str | None = None
    steps: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    need_human_review: bool = False
    evidence_gaps: list[str] = Field(default_factory=list)

    # Deferred extension fields. Not part of the baseline graph contract.
    safety_notes: list[str] = Field(default_factory=list)
    weather_constraints: list[str] = Field(default_factory=list)
    weather_advice: WeatherAdvice | None = None


class FinalAnswerDisplay(BaseModel):
    """Localized, user-facing projection of ``FinalAnswer`` for the web UI."""

    summary: str
    diagnosis: str | None = None
    method_selection: str | None = None
    steps: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    weather_title: str | None = None
    weather_summary: str | None = None
    weather_constraints: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    need_human_review: bool = False


class InterruptState(BaseModel):
    interrupt_id: str
    node_name: str
    prompt: str
    kind: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)


class AuditEvent(BaseModel):
    node_name: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    node_name: str
    message: str
    recoverable: bool = True
    category: ErrorCategory = ErrorCategory.INTERNAL
    code: str = "INTERNAL.UNCLASSIFIED"
    step: str | None = None
    retriable: bool = False
    hint: str = ""
    raw: str = ""
    surface_to_user: bool = False

    @classmethod
    def from_info(
        cls,
        *,
        node_name: str,
        info: ErrorInfo,
        recoverable: bool,
        surface_to_user: bool,
    ) -> ErrorEvent:
        return cls(
            node_name=node_name,
            message=info.message,
            recoverable=recoverable,
            category=info.category,
            code=info.code,
            step=info.step,
            retriable=info.retriable,
            hint=info.hint,
            raw=info.raw,
            surface_to_user=surface_to_user,
        )


class CriticIssue(BaseModel):
    """Diagnostic-only issue used by the current draft safety critic."""

    severity: Literal["error", "warning", "info"]
    code: str = "unspecified"
    message: str
    field_name: str | None = None
    field_path: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class SafetyReview(BaseModel):
    """Diagnostic guardrail result, not a user-feature state domain."""

    passed: bool = False
    issues: list[CriticIssue] = Field(default_factory=list)
    forced_need_human_review: bool = False
    checked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────
# Deferred extension / legacy import models
# ─────────────────────────────────────────────────────────────


class LoadedMemory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_preferences: dict[str, Any] = Field(default_factory=dict)
    regional_context: dict[str, Any] = Field(default_factory=dict)
    resource_constraints: dict[str, Any] = Field(default_factory=dict)
    case_summaries: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("user_preferences", "regional_context", "resource_constraints", mode="before")
    @classmethod
    def _none_to_dict(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("case_summaries", mode="before")
    @classmethod
    def _none_to_case_list(cls, value: Any) -> Any:
        return [] if value is None else value


class AddressContext(BaseModel):
    zip_code: str | None = None
    location_query: str | None = None
    city: str | None = None
    county: str | None = None
    state: str | None = None
    raw_response: dict[str, Any] | None = None
    status: str = "unknown"


class WeatherForecastHour(BaseModel):
    time: str | None = None
    temp_c: float | None = None
    feels_like_c: float | None = None
    chance_of_rain: int | None = None
    chance_of_snow: int | None = None
    precip_mm: float | None = None
    wind_kmph: float | None = None
    humidity: int | None = None
    description: str | None = None


class WeatherForecastDay(BaseModel):
    date: str
    min_temp_c: float | None = None
    max_temp_c: float | None = None
    total_precip_mm: float | None = None
    max_wind_kmph: float | None = None
    hourly: list[WeatherForecastHour] = Field(default_factory=list)
    suitable_for_formal_work: bool | None = None
    risk_reasons: list[str] = Field(default_factory=list)


class WeatherContext(BaseModel):
    status: str = "unknown"
    source: str | None = None
    location_label: str | None = None
    forecast_window_days: int = 3
    current_temp_c: float | None = None
    current_feels_like_c: float | None = None
    current_condition: str | None = None
    humidity: int | None = None
    wind_kmph: float | None = None
    precip_mm: float | None = None
    precipitation_expected: bool | None = None
    dry_window_hours: int | None = None
    forecast_days: list[WeatherForecastDay] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None


class WeatherAdvice(BaseModel):
    status: str = "not_requested"
    title: str = "天气施工建议"
    summary: str | None = None
    next_24h_suitable: bool | None = None
    recommended_window: str | None = None
    constraints: list[str] = Field(default_factory=list)
    practical_tips: list[str] = Field(default_factory=list)
    source: str | None = None


class AgentState(TypedDict, total=False):
    # A. Conversation control
    request_id: str
    thread_id: str
    user_id: str
    session_id: str | None
    locale: Locale
    created_at: str
    updated_at: str | None

    current_stage: Stage
    turn_count: dict[str, int]
    global_signal: GlobalSignal | None
    messages: Annotated[list[AnyMessage], add_messages]

    latest_user_text: str | None
    latest_attachments: list[AttachmentRef]
    new_user_input_pending: bool
    awaiting_user_input: bool
    interrupt: InterruptState | None
    interrupts: Annotated[list[InterruptState], append_list]
    user_intent: str | None
    guard_decision: dict[str, Any] | None
    standalone_query_plan: dict[str, Any] | None
    retrieval_targets: list[str]
    retrieval_strategy: str | None
    speculative_prefetch: dict[str, Any] | None
    top_route: dict[str, Any] | None
    direct_message: str | None
    refusal_type: RefusalType | None
    loaded_memory: LoadedMemory | None
    address_context: AddressContext | None
    weather_context: WeatherContext | None
    weather_advice: WeatherAdvice | None
    weather_route: str | None
    construction_tip_offered: bool
    road_class: str | None
    inferred_work_duration: str | None
    safety_query: str | None
    safety_norm_chunks: list[RetrievedChunk]

    # Transition control aliases used by the current draft graph.
    phase: WorkflowPhase
    next_action: str | None

    # B. Accumulated user facts
    material: str | None
    defect_category: str | None
    defect_subtype: str | None
    known_features: dict[str, Any]
    raw_user_text: str | None
    scene_description: str | None
    scene_context: SceneContext | None

    # Transition bridge for existing nodes. New nodes should prefer top-level
    # user-fact fields.
    distress: DistressInput | None

    # C. Per-turn retrieval results
    rewritten_query: str | None
    retrieved_chunks: list[RetrievedChunk]
    expanded_chunks: list[RetrievedChunk]

    # C.1 disease identification evidence (populated by Phase C.1 RAG + saved for citation)
    disease_evidence_chunks: list[RetrievedChunk]
    disease_discriminator_output: DiseaseDiscriminatorOutput | None
    reconcile_result: dict[str, Any] | None

    # Transition retrieval aliases used by the current draft graph.
    query_plan: QueryPlan | None
    kb_query_plan: QueryPlan | None
    kb_query_plan_v2: KbQueryPlan | None
    kb_plan_type: KbPlanType | None
    kb_subquestions: list[KbSubQuestion]
    kb_hops: list[KbHop]
    kb_hop_results: list[KbHopResult]
    kb_evidence_slots: dict[str, list[RetrievedChunk]]
    kb_missing_slots: list[str]
    kb_clarification_request: KbClarification | None
    kb_planning_trace: dict[str, Any] | None
    kb_final_cited_chunk_ids: list[str]
    kb_rewritten_query: str | None
    kb_retrieved_chunks: list[RetrievedChunk]
    kb_answer: dict[str, Any] | None
    evidence_assessment: EvidenceAssessment | None
    method_evidence_bundles: dict[str, EvidenceBundle]
    rag_route: str | None
    retrieval_attempts: Annotated[list[RetrievalAttempt], append_list]

    # D. Discriminator decision
    method_candidates: list[MethodCandidate]
    discriminating_feature: str | None
    question_to_user: str | None
    need_more_info: bool
    chosen_method: str | None

    # C.2 method selection outputs
    method_evidence_chunks: list[RetrievedChunk]
    method_discriminator_output: MethodDiscriminatorOutput | None

    # Transition decision aliases.
    discriminator_output: DiscriminatorOutput | None
    solution_candidates: list[SolutionCandidate]
    candidate_selection: CandidateSelection | None
    needed_fields: list[str]
    clarification_attempts: int

    # E. Final delivery
    procedure_chunks: list[RetrievedChunk]
    acceptance_chunks: list[RetrievedChunk]
    final_answer: FinalAnswer | str | None
    final_answer_message: str | None
    final_answer_display: FinalAnswerDisplay | None
    reference_index: list[ReferenceItem]

    # Transition detail aliases.
    detail_query_plan: QueryPlan | None
    detail_evidence_bundles: dict[str, EvidenceBundle]

    # Diagnostics / guardrails. These are not business inputs to the
    # discriminator.
    safety_review: SafetyReview | None
    critic_retry_count: int
    field_history: Annotated[list[FieldChange], append_list]
    audit_log: Annotated[list[AuditEvent], append_list]
    errors: Annotated[list[ErrorEvent], append_list]
