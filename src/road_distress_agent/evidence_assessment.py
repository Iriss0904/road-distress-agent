"""Deterministic, observation-only evidence answerability assessment."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from road_distress_agent.errors import BoundaryError, ErrorCategory

ASSESSMENT_VERSION = "a0.1"


class EvidenceStatus(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    NO_KB_EVIDENCE = "no_kb_evidence"
    MISSING_USER_CONTEXT = "missing_user_context"
    CONFLICTING = "conflicting"
    OUT_OF_SCOPE = "out_of_scope"
    DEPENDENCY_FAILURE = "dependency_failure"


class EvidenceRiskFlag(str, Enum):
    NUMERIC = "numeric"
    MATERIAL = "material"
    PROCEDURE = "procedure"
    ACCEPTANCE = "acceptance"


class EvidenceSlotCoverage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slot_id: str = Field(min_length=1)
    allowed_chunk_ids: tuple[str, ...] = ()
    reason_code: str | None = None

    @field_validator("allowed_chunk_ids")
    @classmethod
    def _unique_chunk_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonempty(value, "allowed_chunk_ids")

    @model_validator(mode="after")
    def _coverage_reason_contract(self) -> EvidenceSlotCoverage:
        if bool(self.allowed_chunk_ids) == bool(self.reason_code):
            raise ValueError("covered slots omit reason; uncovered slots require reason")
        return self


class EvidenceConflictSignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    slot_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    values: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()

    @field_validator("chunk_ids")
    @classmethod
    def _unique_chunk_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonempty(value, "conflict chunk_ids")


class DependencyFailureRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: ErrorCategory
    code: str
    step: str
    retriable: bool

    @classmethod
    def from_boundary_error(cls, error: BoundaryError) -> DependencyFailureRef:
        if not isinstance(error, BoundaryError):
            raise TypeError("dependency failure requires a structured BoundaryError")
        info = error.info
        return cls(
            category=info.category,
            code=info.code,
            step=info.step,
            retriable=info.retriable,
        )


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EvidenceStatus
    reason_code: str = Field(min_length=1)
    slot_coverage: tuple[EvidenceSlotCoverage, ...] = ()
    missing_user_fields: tuple[str, ...] = ()
    conflict_signals: tuple[EvidenceConflictSignal, ...] = ()
    allowed_chunk_ids: tuple[str, ...] = ()
    dependency_failure: DependencyFailureRef | None = None
    risk_flags: tuple[EvidenceRiskFlag, ...] = ()
    assessment_version: Literal["a0.1"] = ASSESSMENT_VERSION

    @field_validator("missing_user_fields", "allowed_chunk_ids")
    @classmethod
    def _unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonempty(value, "assessment values")

    @model_validator(mode="after")
    def _failure_contract(self) -> EvidenceAssessment:
        is_failure = self.status is EvidenceStatus.DEPENDENCY_FAILURE
        if is_failure != (self.dependency_failure is not None):
            raise ValueError("dependency_failure must be set exactly for dependency status")
        return self


class EvidenceAssessmentInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    route_action: str | None = None
    retrieval_completed: bool = False
    selected_chunk_ids: tuple[str, ...] = ()
    required_slots: tuple[str, ...] = ()
    slot_coverage: tuple[EvidenceSlotCoverage, ...] = ()
    missing_user_fields: tuple[str, ...] = ()
    conflict_signals: tuple[EvidenceConflictSignal, ...] = ()
    dependency_failure: DependencyFailureRef | None = None
    risk_flags: tuple[EvidenceRiskFlag, ...] = ()

    @field_validator("selected_chunk_ids", "required_slots", "missing_user_fields")
    @classmethod
    def _unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonempty(value, "assessment input values")

    @model_validator(mode="after")
    def _selected_evidence_contract(self) -> EvidenceAssessmentInput:
        selected = set(self.selected_chunk_ids)
        required = set(self.required_slots)
        coverage_slots = tuple(item.slot_id for item in self.slot_coverage)
        if len(set(coverage_slots)) != len(coverage_slots):
            raise ValueError("slot coverage cannot contain duplicate slot ids")
        if not set(coverage_slots).issubset(required):
            raise ValueError("slot coverage must reference required slots")
        conflict_slots = {signal.slot_id for signal in self.conflict_signals}
        if not conflict_slots.issubset(required):
            raise ValueError("conflict signals must reference required slots")
        linked = {
            chunk_id for coverage in self.slot_coverage for chunk_id in coverage.allowed_chunk_ids
        }
        linked.update(chunk_id for signal in self.conflict_signals for chunk_id in signal.chunk_ids)
        if not linked.issubset(selected):
            raise ValueError("slot/conflict chunk ids must be selected evidence ids")
        return self


class EvidenceObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment: EvidenceAssessment | None = None
    field_unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _exactly_one_result(self) -> EvidenceObservation:
        if (self.assessment is None) == (self.field_unavailable_reason is None):
            raise ValueError("observation requires assessment or unavailable reason")
        return self


def assess_evidence(context: EvidenceAssessmentInput) -> EvidenceObservation:
    """Apply the frozen A0 priority without changing runtime behavior."""
    if context.dependency_failure is not None:
        return _observed(context, EvidenceStatus.DEPENDENCY_FAILURE, "dependency_failure")
    if context.route_action == "off_topic":
        return _observed(context, EvidenceStatus.OUT_OF_SCOPE, "effective_route_out_of_scope")
    if context.missing_user_fields:
        return _observed(context, EvidenceStatus.MISSING_USER_CONTEXT, "missing_user_context")
    if context.conflict_signals:
        return _observed(context, EvidenceStatus.CONFLICTING, "explicit_conflict_signal")
    if not context.retrieval_completed:
        return EvidenceObservation(field_unavailable_reason="retrieval_not_completed")
    if not context.required_slots:
        return _without_required_slots(context)
    return _from_slot_coverage(context)


def observe_dependency_failure(error: BoundaryError) -> EvidenceObservation:
    """Project a structured boundary failure without rewriting or swallowing it."""
    return assess_evidence(
        EvidenceAssessmentInput(dependency_failure=DependencyFailureRef.from_boundary_error(error))
    )


def _without_required_slots(context: EvidenceAssessmentInput) -> EvidenceObservation:
    if context.selected_chunk_ids:
        return _observed(
            context,
            EvidenceStatus.SUFFICIENT,
            "healthy_nonempty_retrieval",
            allowed_ids=context.selected_chunk_ids,
        )
    return _observed(context, EvidenceStatus.NO_KB_EVIDENCE, "healthy_empty_retrieval")


def _from_slot_coverage(context: EvidenceAssessmentInput) -> EvidenceObservation:
    coverage = _normalized_coverage(context)
    allowed = _allowed_ids(coverage, context.selected_chunk_ids)
    covered_count = sum(bool(item.allowed_chunk_ids) for item in coverage)
    if covered_count == 0:
        return _observed(
            context,
            EvidenceStatus.NO_KB_EVIDENCE,
            "no_required_slot_covered",
            coverage=coverage,
        )
    if covered_count < len(coverage):
        return _observed(
            context,
            EvidenceStatus.PARTIAL,
            "partial_required_slot_coverage",
            coverage=coverage,
            allowed_ids=allowed,
        )
    return _observed(
        context,
        EvidenceStatus.SUFFICIENT,
        "full_required_slot_coverage",
        coverage=coverage,
        allowed_ids=allowed,
    )


def _normalized_coverage(
    context: EvidenceAssessmentInput,
) -> tuple[EvidenceSlotCoverage, ...]:
    supplied = {item.slot_id: item for item in context.slot_coverage}
    return tuple(
        supplied.get(slot_id)
        or EvidenceSlotCoverage(slot_id=slot_id, reason_code="required_slot_uncovered")
        for slot_id in context.required_slots
    )


def _allowed_ids(
    coverage: tuple[EvidenceSlotCoverage, ...],
    selected_ids: tuple[str, ...],
) -> tuple[str, ...]:
    linked = {chunk_id for item in coverage for chunk_id in item.allowed_chunk_ids}
    return tuple(chunk_id for chunk_id in selected_ids if chunk_id in linked)


def _observed(
    context: EvidenceAssessmentInput,
    status: EvidenceStatus,
    reason_code: str,
    *,
    coverage: tuple[EvidenceSlotCoverage, ...] = (),
    allowed_ids: tuple[str, ...] = (),
) -> EvidenceObservation:
    return EvidenceObservation(
        assessment=EvidenceAssessment(
            status=status,
            reason_code=reason_code,
            slot_coverage=coverage,
            missing_user_fields=context.missing_user_fields,
            conflict_signals=context.conflict_signals,
            allowed_chunk_ids=allowed_ids,
            dependency_failure=context.dependency_failure,
            risk_flags=context.risk_flags,
        )
    )


def _unique_nonempty(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} cannot contain blank values")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return values
