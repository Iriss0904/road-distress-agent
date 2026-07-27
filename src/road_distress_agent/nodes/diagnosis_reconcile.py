"""Reconcile user facts and invalidate derived diagnosis state."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.nodes.diagnosis_dependencies import (
    NextStage,
    canonical_key,
    earliest_stage,
    highest_invalidated_stage,
    invalidations,
    target_stage,
)
from road_distress_agent.resume import (
    DISEASE_CLARIFICATION,
    DISEASE_SELECTION,
    METHOD_CLARIFICATION,
    METHOD_SELECTION,
)
from road_distress_agent.state import AgentState, AuditEvent, FieldChange
from road_distress_agent.tracing import trace_event


class FactUpdate(BaseModel):
    key: str
    value: Any
    reason: str | None = None


class ReconcilePatch(BaseModel):
    material: str | None = None
    fact_updates: list[FactUpdate] = Field(default_factory=list)
    correction: bool = False
    notes: str | None = None


class ReconcileResult(BaseModel):
    changed_fields: list[str] = Field(default_factory=list)
    invalidated: list[str] = Field(default_factory=list)
    next_stage: NextStage
    source: str
    notes: str | None = None


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / "diagnosis_reconcile.txt"
    return path.read_text(encoding="utf-8")


def _plain_value(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _same_value(left: Any, right: Any) -> bool:
    left_json = json.dumps(_plain_value(left), ensure_ascii=False, sort_keys=True)
    return left_json == json.dumps(_plain_value(right), ensure_ascii=False, sort_keys=True)


def _pending_payload(state: AgentState) -> dict[str, Any] | None:
    interrupt = state.get("interrupt")
    if not interrupt:
        return None
    data = interrupt.model_dump(mode="json")
    data["required_fields"] = [canonical_key(field) for field in data["required_fields"]]
    return data


def _missing_feature_from_output(output: Any) -> str | None:
    if output is None:
        return None
    if hasattr(output, "missing_feature"):
        return output.missing_feature
    if isinstance(output, dict):
        return output.get("missing_feature")
    return None


def _pending_missing_feature(state: AgentState) -> str | None:
    interrupt = state.get("interrupt")
    if interrupt and interrupt.required_fields:
        return canonical_key(interrupt.required_fields[0])
    if _missing := _missing_feature_from_output(state.get("method_discriminator_output")):
        return canonical_key(_missing)
    missing = _missing_feature_from_output(state.get("disease_discriminator_output"))
    return canonical_key(missing) if missing else None


def _build_messages(state: AgentState) -> list[Any]:
    route = state.get("top_route") or {}
    payload = {
        "target_locale": state.get("locale") or "zh-CN",
        "language_instruction": user_language_instruction(state.get("locale")),
        "latest_user_text": state.get("latest_user_text") or "",
        "material": state.get("material"),
        "known_features": state.get("known_features") or {},
        "defect_category": state.get("defect_category"),
        "chosen_method": state.get("chosen_method"),
        "top_route": route,
        "active_interrupt": _pending_payload(state),
        "pending_missing_feature": _pending_missing_feature(state),
        "scene_description": state.get("scene_description"),
    }
    human = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    return [SystemMessage(content=_prompt_text()), human]


def _invoke_llm(messages: list[Any]) -> ReconcilePatch:
    timeout = llm_timeout_seconds("diagnosis_reconcile")
    llm = deepseek_chat(correlation_name="diagnosis_reconcile", timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(ReconcilePatch, **kwargs)
    result = invoke_llm_call(
        error_context_name="diagnosis_reconcile",
        usage_correlation_name="diagnosis_reconcile",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, ReconcilePatch):
        return result
    return ReconcilePatch.model_validate(result)


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _selection_resume_stage(state: AgentState) -> NextStage | None:
    interrupt = state.get("interrupt")
    route = state.get("top_route") or {}
    if route.get("diagnosis_intent") != "candidate_choice" or not interrupt:
        return None
    if interrupt.kind == DISEASE_SELECTION:
        return "disease"
    if interrupt.kind == METHOD_SELECTION:
        return "method"
    return None


def _change(field_path: str, old: Any, new: Any, reason: str) -> FieldChange:
    return FieldChange(field_path=field_path, old_value=old, new_value=new, reason=reason)


def _merge_patch(
    state: AgentState,
    patch: ReconcilePatch,
) -> tuple[dict[str, Any], list[FieldChange]]:
    updates: dict[str, Any] = {}
    changes: list[FieldChange] = []
    if patch.material and patch.material != state.get("material"):
        updates["material"] = patch.material
        reason = patch.notes or "user_fact_update"
        changes.append(_change("material", state.get("material"), patch.material, reason))
    features = dict(state.get("known_features") or {})
    for item in patch.fact_updates:
        key = canonical_key(item.key)
        old = features.get(key)
        if _same_value(old, item.value):
            continue
        features[key] = item.value
        reason = item.reason or patch.notes or "user_fact_update"
        changes.append(_change(f"known_features.{key}", old, item.value, reason))
    if changes:
        updates["known_features"] = features
    return updates, changes


def _next_without_changes(state: AgentState) -> NextStage:
    if stage := target_stage(state.get("top_route") or {}):
        return stage
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    if kind in (DISEASE_CLARIFICATION, DISEASE_SELECTION):
        return "disease"
    if kind in (METHOD_CLARIFICATION, METHOD_SELECTION):
        return "method"
    if not state.get("defect_category"):
        return "disease"
    if not state.get("chosen_method"):
        return "method"
    if state.get("final_answer") is None:
        return "detail"
    return "done"


_FEATURE_PREFIX = "known_features."


def _changed_feature_keys(changes: list[FieldChange]) -> set[str]:
    return {
        c.field_path.rsplit(".", 1)[-1] for c in changes if c.field_path.startswith(_FEATURE_PREFIX)
    }


def _answers_method_clarification(
    state: AgentState,
    changes: list[FieldChange],
) -> bool:
    interrupt = state.get("interrupt")
    if not interrupt or interrupt.kind != METHOD_CLARIFICATION:
        return False
    required = {canonical_key(field) for field in interrupt.required_fields}
    if pending := _pending_missing_feature(state):
        required.add(canonical_key(pending))
    return bool(required) and required.issubset(_changed_feature_keys(changes))


def _stage_after_changes(
    state: AgentState,
    fact_updates: dict[str, Any],
    changes: list[FieldChange],
    correction: bool,
) -> NextStage:
    merged = {**state, **fact_updates}
    has_defect = bool(merged.get("defect_category"))
    if has_defect and _answers_method_clarification(state, changes):
        return "method"
    changed_stage = highest_invalidated_stage(changes, has_defect)
    if has_defect and changed_stage == "disease" and not correction:
        changed_stage = "method"
    incomplete_stage = _next_without_changes(merged)
    return earliest_stage(changed_stage, incomplete_stage)


def _result(
    stage: NextStage,
    source: str,
    changes: list[FieldChange],
    invalidated: dict[str, Any],
    notes: str | None,
) -> ReconcileResult:
    return ReconcileResult(
        changed_fields=[change.field_path for change in changes],
        invalidated=sorted(invalidated.keys()),
        next_stage=stage,
        source=source,
        notes=notes,
    )


def diagnosis_reconcile(state: AgentState) -> AgentState:
    """Update known_features first, then invalidate stale derived values."""
    if stage := _selection_resume_stage(state):
        result = _result(stage, "candidate_selection_passthrough", [], {}, None)
        return {"reconcile_result": result.model_dump(mode="json"), "next_action": f"run_{stage}"}
    messages = _build_messages(state)
    patch = _invoke_llm(messages)
    fact_updates, changes = _merge_patch(state, patch)
    stage = _next_without_changes(state)
    if changes:
        stage = _stage_after_changes(state, fact_updates, changes, patch.correction)
    invalidated = invalidations(stage) if changes else {}
    result = _result(stage, "llm", changes, invalidated, patch.notes)
    direct_message = None
    if stage == "done" and changes:
        direct_message = (
            "Noted. This update does not change the current distress classification "
            "or treatment method."
            if state.get("locale") == "en-US"
            else "已记录这条补充信息；它不改变当前病害判断和处治方法。"
        )
    return {
        **fact_updates,
        **invalidated,
        "reconcile_result": result.model_dump(mode="json"),
        "field_history": changes,
        "direct_message": direct_message,
        "phase": WorkflowPhase.INPUT,
        "next_action": f"run_{stage}",
        "audit_log": [
            trace_event(
                node_name="diagnosis_reconcile",
                kind="llm_call",
                title="Reconcile diagnosis facts",
                inputs={"known_features": state.get("known_features") or {}},
                prompt=messages,
                output={"patch": patch, "result": result},
                state_delta={**fact_updates, **invalidated},
            ),
            AuditEvent(
                node_name="diagnosis_reconcile",
                message="Reconciled diagnosis facts and invalidated stale derived state.",
                metadata=result.model_dump(mode="json"),
            ),
        ],
    }
