"""Deterministic quantity + cost computation over the norm library.

Pure functions only: given a norm-code mapping and dimensions, compute quantity
(per ``calculation_unit``), per-defect cost (sum of resource-line amounts), and
segment-level totals where mobilization fees are counted **once per task** per
``shared_cost_rules``. No LLM, so numbers are auditable and never hallucinated.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from road_distress_agent.delivery.cost_norm_lookup import CostNormLookup, NormItem, SharedCostRule

MM_PER_M = 1000.0
_LENGTH_UNITS = {"m"}
_AREA_UNITS = {"㎡", "m2", "m²"}
_VOLUME_UNITS = {"m³", "m3"}


class Dimensions(BaseModel):
    """Dimensions extracted (by LLM) from a defect's known_features."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    length_m: float | None = None
    area_m2: float | None = None
    depth_mm: float | None = None


class CostLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    source_thread_id: str | None = None
    defect_category: str | None
    norm_code: str
    process_name: str
    calculation_unit: str
    quantity: float
    unit_cost_cny: float
    subtotal_cny: float


class MobilizationLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_code: str
    cost_name: str
    amount_cny: float
    aggregation_rule: str | None = None


class CostSheet(BaseModel):
    model_config = ConfigDict(frozen=True)

    defect_lines: list[CostLine] = Field(default_factory=list)
    mobilization_lines: list[MobilizationLine] = Field(default_factory=list)
    defect_subtotal_cny: float = 0.0
    mobilization_subtotal_cny: float = 0.0
    total_cny: float = 0.0


class CostInput(BaseModel):
    """One per-defect costing input: the norm mapping + dimensions for a record."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    record_id: str
    source_thread_id: str | None = None
    defect_category: str | None = None
    norm_code: str
    dimensions: Dimensions


def quantity_for_unit(calculation_unit: str, dims: Dimensions) -> float:
    """Compute the billable quantity for a calculation unit from dimensions."""
    unit = calculation_unit.strip()
    if unit in _LENGTH_UNITS:
        return _require(dims.length_m, "length_m", unit)
    if unit in _AREA_UNITS:
        return _require(dims.area_m2, "area_m2", unit)
    if unit in _VOLUME_UNITS:
        area = _require(dims.area_m2, "area_m2", unit)
        depth = _require(dims.depth_mm, "depth_mm", unit)
        return round(area * depth / MM_PER_M, 4)
    raise ValueError(f"unsupported calculation_unit: {calculation_unit!r}")


def unit_cost(lookup: CostNormLookup, norm_code: str) -> float:
    """Sum of resource-line amounts per norm unit (labor already included)."""
    lines = lookup.resource_lines(norm_code)
    if not lines:
        raise ValueError(f"norm {norm_code} has no resource lines.")
    return round(sum(line.amount_cny_per_norm_unit for line in lines), 4)


def cost_line(lookup: CostNormLookup, item: CostInput) -> CostLine:
    norm: NormItem | None = lookup.get_norm(item.norm_code)
    if norm is None:
        raise ValueError(f"unknown norm_code: {item.norm_code}")
    qty = quantity_for_unit(norm.calculation_unit, item.dimensions)
    per_unit = unit_cost(lookup, item.norm_code)
    return CostLine(
        record_id=item.record_id,
        source_thread_id=item.source_thread_id,
        defect_category=item.defect_category,
        norm_code=norm.norm_code,
        process_name=norm.process_name,
        calculation_unit=norm.calculation_unit,
        quantity=qty,
        unit_cost_cny=per_unit,
        subtotal_cny=round(qty * per_unit, 2),
    )


def build_cost_sheet(lookup: CostNormLookup, items: list[CostInput]) -> CostSheet:
    """Per-defect costs + segment-level mobilization fees counted once per task."""
    defect_lines = [cost_line(lookup, item) for item in items]
    used_norms = {line.norm_code for line in defect_lines}
    mobilization = _mobilization_lines(lookup.active_shared_cost_rules(), used_norms)

    defect_subtotal = round(sum(line.subtotal_cny for line in defect_lines), 2)
    mobilization_subtotal = round(sum(line.amount_cny for line in mobilization), 2)
    return CostSheet(
        defect_lines=defect_lines,
        mobilization_lines=mobilization,
        defect_subtotal_cny=defect_subtotal,
        mobilization_subtotal_cny=mobilization_subtotal,
        total_cny=round(defect_subtotal + mobilization_subtotal, 2),
    )


def _mobilization_lines(
    rules: list[SharedCostRule], used_norms: set[str]
) -> list[MobilizationLine]:
    """One line per applicable rule — the rule already encodes 'count once'."""
    lines: list[MobilizationLine] = []
    seen: set[str] = set()
    for rule in rules:
        if rule.rule_code in seen or not rule.applies_to(used_norms):
            continue
        seen.add(rule.rule_code)
        lines.append(
            MobilizationLine(
                rule_code=rule.rule_code,
                cost_name=rule.cost_name,
                amount_cny=rule.demo_amount_cny,
                aggregation_rule=rule.aggregation_rule,
            )
        )
    return lines


def _require(value: Any, name: str, unit: str) -> float:
    if value is None:
        raise ValueError(f"calculation_unit {unit!r} requires dimension {name}.")
    return float(value)
