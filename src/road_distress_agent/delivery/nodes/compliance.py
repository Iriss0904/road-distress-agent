"""Compliance critic for the delivery package.

Independent of the diagnosis-layer ``safety_critic``. A deterministic guardrail
(numbers must be auditable, not LLM-judged): work orders must carry each defect's
confirmed method, every kept defect must cite evidence, and the cost aggregation
must be internally consistent with the norm library — mobilization fees counted
once, no unauthorized rules, totals that add up. Failure forces human review and
never silently delivers.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from road_distress_agent.delivery.cost_norm_lookup import make_cost_norm_lookup
from road_distress_agent.delivery.selection import kept_records
from road_distress_agent.delivery.state import DeliveryState
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.state import AuditEvent, CriticIssue, SafetyReview

_CENTS = 0.01


def compliance_critic(state: DeliveryState) -> DeliveryState:
    records = kept_records(state)
    locale = _state_locale(state)
    cost = state.get("cost_result") or {}
    issues: list[CriticIssue] = []

    _check_methods(records, issues, locale)
    _check_citations(records, issues, locale)
    _check_cost_integrity(cost, issues, locale)
    _check_mobilization_authorized(cost, issues, locale)
    _check_skipped(cost, issues, locale)

    errors = [i for i in issues if i.severity == "error"]
    force_review = bool(errors) or bool(cost.get("skipped"))
    review = SafetyReview(passed=not errors, issues=issues, forced_need_human_review=force_review)
    return {
        "compliance_review": review,
        "audit_log": [
            AuditEvent(
                node_name="compliance_critic",
                message=_audit_message(bool(errors), locale),
                metadata={
                    "passed": review.passed,
                    "error_count": len(errors),
                    "issue_count": len(issues),
                    "forced_need_human_review": force_review,
                },
            )
        ],
    }


def _check_methods(records: list[dict[str, Any]], issues: list[CriticIssue], locale: str) -> None:
    for record in records:
        if not (record.get("payload") or {}).get("chosen_method"):
            issues.append(
                _issue(
                    "error",
                    "work_order_method_missing",
                    _message("method_missing", locale, record_id=record.get("record_id")),
                )
            )


def _check_citations(records: list[dict[str, Any]], issues: list[CriticIssue], locale: str) -> None:
    for record in records:
        if not (record.get("payload") or {}).get("citations"):
            issues.append(
                _issue(
                    "warning",
                    "deliverable_citation_missing",
                    _message("citation_missing", locale, record_id=record.get("record_id")),
                )
            )


def _check_cost_integrity(cost: dict[str, Any], issues: list[CriticIssue], locale: str) -> None:
    if not cost:
        return
    defect = float(cost.get("defect_subtotal_cny") or 0.0)
    mobilization = float(cost.get("mobilization_subtotal_cny") or 0.0)
    total = float(cost.get("total_cny") or 0.0)
    mob_lines_sum = sum(float(m.get("amount_cny") or 0.0) for m in cost.get("mobilization") or [])

    if abs(mob_lines_sum - mobilization) > _CENTS:
        issues.append(
            _issue("error", "mobilization_subtotal_mismatch", _message("mob_mismatch", locale))
        )
    if abs((defect + mobilization) - total) > _CENTS:
        issues.append(_issue("error", "cost_total_mismatch", _message("cost_mismatch", locale)))


def _check_mobilization_authorized(
    cost: dict[str, Any], issues: list[CriticIssue], locale: str
) -> None:
    mobilization = cost.get("mobilization") or []
    if not mobilization:
        return
    rule_codes = [m.get("rule_code") for m in mobilization]
    for code, count in Counter(rule_codes).items():
        if count > 1:
            issues.append(
                _issue(
                    "error",
                    "mobilization_double_counted",
                    _message("mob_double", locale, code=code),
                )
            )

    active = {r.rule_code: r for r in make_cost_norm_lookup().active_shared_cost_rules()}
    used_norms = set(cost.get("norm_codes") or [])
    for code in rule_codes:
        rule = active.get(code)
        if rule is None:
            issues.append(
                _issue(
                    "error",
                    "mobilization_rule_unauthorized",
                    _message("mob_unauthorized", locale, code=code),
                )
            )
        elif not rule.applies_to(used_norms):
            issues.append(
                _issue(
                    "error",
                    "mobilization_rule_not_applicable",
                    _message("mob_not_applicable", locale, code=code),
                )
            )


def _check_skipped(cost: dict[str, Any], issues: list[CriticIssue], locale: str) -> None:
    for skip in cost.get("skipped") or []:
        issues.append(
            _issue(
                "warning",
                "defect_uncosted",
                _message(
                    "uncosted",
                    locale,
                    record_id=skip.get("record_id"),
                    reason=skip.get("reason"),
                ),
            )
        )


def _issue(severity: str, code: str, message: str) -> CriticIssue:
    return CriticIssue(severity=severity, code=code, message=message)


def _state_locale(state: DeliveryState) -> str:
    return normalize_locale(state.get("locale") or DEFAULT_LOCALE)


def _audit_message(has_errors: bool, locale: str) -> str:
    if locale == "en-US":
        return (
            "Delivery compliance gate found blocking issues."
            if has_errors
            else "Delivery compliance gate completed."
        )
    return "交付合规守门发现阻断问题。" if has_errors else "交付合规守门完成。"


def _message(key: str, locale: str, **values: Any) -> str:
    messages = _MESSAGES_EN if locale == "en-US" else _MESSAGES_ZH
    return messages[key].format(**values)


_MESSAGES_ZH = {
    "method_missing": "记录 {record_id} 缺少已确认处治方法，工单不得派发。",
    "citation_missing": "记录 {record_id} 无规范引用，交付物缺少依据。",
    "mob_mismatch": "进场费小计与明细之和不一致。",
    "cost_mismatch": "造价合计不等于病害小计与进场费小计之和。",
    "mob_double": "进场费规则 {code} 被重复计列。",
    "mob_unauthorized": "进场费规则 {code} 非启用规则。",
    "mob_not_applicable": "进场费规则 {code} 不适用于本次使用的定额。",
    "uncosted": "记录 {record_id} 未计价：{reason}。",
}
_MESSAGES_EN = {
    "method_missing": (
        "Record {record_id} is missing a confirmed treatment method; the work order "
        "cannot be issued."
    ),
    "citation_missing": (
        "Record {record_id} has no standards citation, so the deliverable lacks evidence."
    ),
    "mob_mismatch": "Mobilization subtotal does not match the sum of mobilization lines.",
    "cost_mismatch": "Total cost does not equal defect subtotal plus mobilization subtotal.",
    "mob_double": "Mobilization rule {code} is counted more than once.",
    "mob_unauthorized": "Mobilization rule {code} is not an enabled rule.",
    "mob_not_applicable": (
        "Mobilization rule {code} is not applicable to the norms used in this task."
    ),
    "uncosted": "Record {record_id} was not costed: {reason}.",
}
