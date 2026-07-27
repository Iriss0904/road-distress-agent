"""Read-only structured lookup over the maintenance norm library (SQLite).

The cost library is structured data, not a RAG source: per-norm resource/price
breakdowns and segment-level mobilization rules. The cost agent uses an LLM only
to map a defect/method onto a ``norm_code``; all quantities and prices come from
here via deterministic queries, so numbers can never be hallucinated.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

DEFAULT_NORM_DB = ""


class NormItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    norm_code: str
    process_name: str
    category: str | None = None
    applicable_defect: str | None = None
    calculation_unit: str
    default_thickness_mm: float | None = None
    quantity_rule_cn: str | None = None
    labor_workday_per_unit: float | None = None
    main_material_summary: str | None = None
    basis_note: str | None = None
    source_url: str | None = None

    @field_validator("default_thickness_mm", "labor_workday_per_unit", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value


class ResourceLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    norm_code: str
    line_no: int
    resource_type: str
    resource_name: str
    resource_unit: str
    consumption_per_norm_unit: float
    demo_unit_price_cny: float
    amount_cny_per_norm_unit: float


class SharedCostRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    rule_code: str
    applicable_norm_code: str
    cost_name: str
    cost_type: str
    demo_amount_cny: float
    aggregation_rule: str | None = None
    apply_in_demo: str | None = None
    note: str | None = None

    def applies_to(self, norm_codes: set[str]) -> bool:
        rule_norms = {c.strip() for c in self.applicable_norm_code.split(",") if c.strip()}
        return bool(rule_norms & norm_codes)


class CostNormLookup(Protocol):
    def list_norms(self) -> list[NormItem]: ...

    def get_norm(self, norm_code: str) -> NormItem | None: ...

    def resource_lines(self, norm_code: str) -> list[ResourceLine]: ...

    def active_shared_cost_rules(self) -> list[SharedCostRule]: ...


class SQLiteCostNormLookup:
    """SQLite-backed :class:`CostNormLookup`."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        raw = db_path or os.environ.get("ROAD_DISTRESS_NORM_DB", DEFAULT_NORM_DB)
        if not raw:
            raise RuntimeError(
                "ROAD_DISTRESS_NORM_DB is required for cost estimation. "
                "Provide a SQLite norm library you are authorized to use."
            )
        self.db_path = Path(raw)

    def list_norms(self) -> list[NormItem]:
        rows = self._query("SELECT * FROM norm_items ORDER BY norm_code")
        return [NormItem.model_validate(dict(row)) for row in rows]

    def get_norm(self, norm_code: str) -> NormItem | None:
        rows = self._query("SELECT * FROM norm_items WHERE norm_code = ?", (norm_code,))
        return NormItem.model_validate(dict(rows[0])) if rows else None

    def resource_lines(self, norm_code: str) -> list[ResourceLine]:
        rows = self._query(
            "SELECT * FROM resource_lines WHERE norm_code = ? ORDER BY line_no", (norm_code,)
        )
        return [ResourceLine.model_validate(dict(row)) for row in rows]

    def active_shared_cost_rules(self) -> list[SharedCostRule]:
        rows = self._query("SELECT * FROM shared_cost_rules WHERE apply_in_demo = 'Y'")
        return [SharedCostRule.model_validate(dict(row)) for row in rows]

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"norm library not found: {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()


def make_cost_norm_lookup(db_path: str | Path | None = None) -> CostNormLookup:
    return SQLiteCostNormLookup(db_path=db_path)
