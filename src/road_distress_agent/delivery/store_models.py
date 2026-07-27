"""Immutable metadata models for the delivery archive."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

DeliveryStatus = Literal["draft", "final"]
GeneratedBy = Literal["auto", "human"]
FileFormat = Literal["docx", "pdf", "xlsx"]
ID_HEX_LEN = 12


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _delivery_id() -> str:
    return f"del-{uuid4().hex[:ID_HEX_LEN]}"


def _version_id() -> str:
    return f"ver-{uuid4().hex[:ID_HEX_LEN]}"


class Delivery(BaseModel):
    """One archived delivery package for an inspection project."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    delivery_id: str = Field(default_factory=_delivery_id)
    project_id: str
    user_id: str
    status: DeliveryStatus = "draft"
    title: str
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)


class DeliveryVersion(BaseModel):
    """One filesystem-backed version of a delivery document."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    version_id: str = Field(default_factory=_version_id)
    delivery_id: str
    version_no: int
    file_path: str
    file_format: FileFormat
    checksum: str
    generated_by: GeneratedBy
    note: str | None = None
    record_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now_iso)
