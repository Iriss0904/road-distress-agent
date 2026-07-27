"""Immutable domain models for the inspection-project ledger.

These are the cross-session aggregation entities. They are deliberately
``frozen`` (immutable-first): state transitions return new instances rather than
mutating in place, mirroring the rest of the codebase's value-object style.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from road_distress_agent.localization import DEFAULT_LOCALE, Locale

ProjectStatus = Literal["active", "archiving", "delivered"]
RecordStatus = Literal["active", "superseded", "incomplete"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_id() -> str:
    return f"proj-{uuid4().hex[:12]}"


def _record_id() -> str:
    return f"rec-{uuid4().hex[:12]}"


class DefectPayload(BaseModel):
    """Structured snapshot promoted from one diagnosis thread's terminal state.

    The diagnosis layer remains the source of truth; this is a read-only
    projection captured at promotion time. Citation/evidence references are kept
    as ids/dicts so the compliance critic can trace them back without re-running
    retrieval.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    defect_category: str | None = None
    defect_subtype: str | None = None
    material: str | None = None
    known_features: dict[str, Any] = Field(default_factory=dict)
    chosen_method: str | None = None
    summary: str | None = None
    procedure_refs: list[str] = Field(default_factory=list)
    acceptance_refs: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    photo_refs: list[str] = Field(default_factory=list)
    final_answer: dict[str, Any] | None = None


class DefectRecord(BaseModel):
    """One promoted defect inside a project ledger, bound to its source thread."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    record_id: str = Field(default_factory=_record_id)
    project_id: str
    source_thread_id: str
    status: RecordStatus = "active"
    payload: DefectPayload
    created_at: str = Field(default_factory=_utc_now_iso)

    def with_status(self, status: RecordStatus) -> DefectRecord:
        return self.model_copy(update={"status": status})


class InspectionProject(BaseModel):
    """A road-segment inspection task aggregating many defect records."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    project_id: str = Field(default_factory=_project_id)
    user_id: str
    name: str
    locale: Locale = DEFAULT_LOCALE
    segment: str | None = None
    inspector: str | None = None
    crew: str | None = None
    status: ProjectStatus = "active"
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)

    def touched(self, **updates: Any) -> InspectionProject:
        """Return a copy with ``updates`` applied and ``updated_at`` refreshed."""
        return self.model_copy(update={**updates, "updated_at": _utc_now_iso()})
