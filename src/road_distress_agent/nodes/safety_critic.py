# ─────────────────────────────────────────────────────────────
# REFACTOR NOTES (new road_rag_design.md §五/§六)
#
# 保留（与新设计 §五"可解释来源"对齐的防幻觉骨架）：
#   - _check_citations：citation_id ⊂ 检索证据 + source_clause 反查防伪造条款；
#   - _check_method_consistency 的前半段：用户已选 candidate ↔ 最终方案名称匹配；
#   - _check_review_triggers：置信度阈值、检索相似度阈值、bundle 不足→强制 human review；
#   - _check_weather：天气不可用时禁止"未来24小时适合施工"幻觉；
#                       blocking 天气下补"暂缓/排水"安全提示；
#   - _selected_candidate / _method_matches / _known_source_clauses /
#     _mentioned_clauses 等通用工具；
#   - 整个 retryable_error → critic_retry_count 闭环。
#
# 删除（与硬编码病害 taxonomy 一同被新设计第六节舍弃）：
#   - HOT_REPAIR_TERMS / RUTTING_REHAB_TERMS 关键词常量；
#   - _check_method_consistency 后半段"distress.distress_type==RUTTING +
#     deep_or_wet_rutting + 热修补术语命中"的硬编码规则；
#     这类按 DistressType enum + 数值阈值定性的"一致性"判断，
#     正是新设计反对的 hardcoded applicable_conditions。
#     方案适配性应由 Discriminator + answer composer + LLM critic 协同处理。
#   - _check_scene_safety 整段（intersection / heavy_traffic / water_risk 等
#     关键词命中 → 自动补"交通管控/排水"提示）。这套依赖
#     distress.extra_observations / lane_position / location_text 等已删字段，
#     且本身就是 brittle 关键词匹配。安全提示应由 answer_composer / weather_advisor 生成。
#   - 对 DistressType / extra_observations / has_water / lane_position 的全部引用。
#
# 待新写：
#   - 可选的 LLM-based critic（在 retryable_error 之外，加一层 LLM 对方案-证据
#     一致性的语义复核），保持现在的规则层为低延迟前置筛。
# ─────────────────────────────────────────────────────────────
"""Rule-layer safety critic for final-answer validation."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from road_distress_agent.deterministic_final_renderer import (
    can_render_deterministically,
    render_final_answer,
)
from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.final_response_citations import (
    clause_ids,
    split_clause_ids,
    with_clause_citation_tokens,
)
from road_distress_agent.final_response_presenter import (
    PresentedFinalAnswer,
    present_final_answer,
)
from road_distress_agent.localization import is_english
from road_distress_agent.reference_index import build_reference_index, reference_ids
from road_distress_agent.settings import deterministic_final_renderer_enabled
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    Citation,
    CriticIssue,
    EvidenceBundle,
    FinalAnswer,
    FinalAnswerDisplay,
    ReferenceItem,
    SafetyReview,
    SolutionCandidate,
    WeatherAdvice,
)
from road_distress_agent.tracing import trace_event

DEFAULT_MIN_CANDIDATE_CONFIDENCE = 0.60
DEFAULT_MIN_EVIDENCE_SIMILARITY = 0.60
EVIDENCE_QUALITY_KEY = "evidence_quality"
FIRST_RETRY_COUNT = 1

_CRITIC_MESSAGES_ZH = {
    "tip_only_audit": "仅展示非规范施工贴士，未重复主体答案。",
    "presenter_title": "最终回复展示 LLM",
    "presenter_audit": "已按偏好渲染最终聊天回复。",
    "deterministic_presenter_audit": "已用确定性模板渲染最终聊天回复。",
    "missing_final_citations": "最终答案没有集中挂载 citations，无法证明答案有依据。",
    "citation_not_from_retrieved_evidence": "最终答案包含未来自检索证据的 citation。",
    "unsupported_source_clause": "最终答案出现未由 Citation.source_clause 支撑的规范条款号。",
    "selected_candidate_not_found": "用户选择的候选不在当前 solution_candidates 中。",
    "final_method_differs": "最终方案没有沿用用户确认的候选方案。",
    "final_method_not_in_candidates": "最终方案名不属于候选方案列表。",
    "top_candidate_low_confidence": (
        "系统 Top-1 候选置信度低于安全阈值，任何候选落地前都需要人工复核。"
    ),
    "selected_candidate_low_confidence": "用户选择的候选置信度低于安全阈值。",
    "user_selected_non_top1": "用户没有选择系统 Top-1 候选，说明系统对病害或方案判断存在不确定性。",
    "evidence_bundle_insufficient": "{purpose} 的检索证据不足。",
    "evidence_quality_below_threshold": "{purpose} 的最高证据质量分低于安全阈值。",
    "weather_skip_note_added": "用户跳过天气校验，Critic 补充开工前复查提示。",
    "weather_hallucination": "天气不可用时，最终答案不能生成具体天气适宜性断言。",
    "weather_blocking_note_added": "天气阻断施工，但最终安全提示不完整，Critic 已补充。",
    "missing_final_answer": "SafetyCritic 运行时 final_answer 为空。",
    "missing_final_answer_audit": "安全审查失败：final_answer 为空。",
    "retryable_audit": "安全审查发现可重试问题。",
    "completed_audit": "安全审查完成规则层复核。",
}
_CRITIC_MESSAGES_EN = {
    "tip_only_audit": (
        "Rendered non-normative construction tips without repeating the main answer."
    ),
    "presenter_title": "Final response presenter LLM",
    "presenter_audit": "Rendered the final answer as preference-aware chat text.",
    "deterministic_presenter_audit": "Rendered the final answer deterministically.",
    "missing_final_citations": (
        "The final answer has no centralized citations, so its basis cannot be verified."
    ),
    "citation_not_from_retrieved_evidence": (
        "The final answer contains a citation that was not retrieved as evidence."
    ),
    "unsupported_source_clause": (
        "The final answer mentions a standards clause not supported by Citation.source_clause."
    ),
    "selected_candidate_not_found": (
        "The user's selected candidate is not in the current solution_candidates list."
    ),
    "final_method_differs": "The final method does not use the candidate confirmed by the user.",
    "final_method_not_in_candidates": "The final method name is not in the candidate list.",
    "top_candidate_low_confidence": (
        "The top-ranked candidate is below the safety threshold; manual review is "
        "required before use."
    ),
    "selected_candidate_low_confidence": (
        "The user-selected candidate is below the safety threshold."
    ),
    "user_selected_non_top1": (
        "The user did not select the top-ranked candidate, indicating uncertainty in "
        "the classification or method."
    ),
    "evidence_bundle_insufficient": "Retrieved evidence for {purpose} is insufficient.",
    "evidence_quality_below_threshold": (
        "The highest evidence-quality score for {purpose} is below the safety threshold."
    ),
    "weather_skip_note_added": (
        "The user skipped weather verification, so the critic added a pre-work weather check."
    ),
    "weather_hallucination": (
        "When weather is unavailable, the final answer must not assert specific "
        "weather suitability."
    ),
    "weather_blocking_note_added": (
        "Weather blocks construction, and the critic added missing safety guidance."
    ),
    "missing_final_answer": "SafetyCritic ran without a final_answer.",
    "missing_final_answer_audit": "Safety critic failed because final_answer is missing.",
    "retryable_audit": "Safety critic found retryable issues.",
    "completed_audit": "Safety critic completed the rule-layer review.",
}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _critic_message(key: str, locale: str | None, **values: object) -> str:
    messages = _CRITIC_MESSAGES_EN if locale == "en-US" else _CRITIC_MESSAGES_ZH
    return messages[key].format(**values)


def _issue(
    *,
    severity: str,
    code: str,
    message: str,
    field_path: str | None = None,
    evidence_ids: Iterable[str] = (),
) -> CriticIssue:
    return CriticIssue(
        severity=severity,
        code=code,
        message=message,
        field_name=field_path,
        field_path=field_path,
        evidence_ids=list(evidence_ids),
    )


def _bundles(state: AgentState) -> dict[str, EvidenceBundle]:
    return {
        **state.get("method_evidence_bundles", {}),
        **state.get("detail_evidence_bundles", {}),
    }


def _evidence_citations(state: AgentState) -> dict[str, Citation]:
    """Build a lookup of all retrieved evidence from every pipeline phase.

    The legacy path covers method_evidence_bundles / detail_evidence_bundles.
    The new C.1→C.2→D pipeline stores chunks directly in state fields rather
    than in bundles, so we also index those to prevent false-positive
    "citation_not_from_retrieved_evidence" errors.
    """
    citations: dict[str, Citation] = {}

    for bundle in _bundles(state).values():
        for citation in bundle.citations:
            cid = citation.citation_id or citation.chunk_id
            if cid:
                citations[cid] = citation

    for field in (
        "disease_evidence_chunks",
        "method_evidence_chunks",
        "procedure_chunks",
        "acceptance_chunks",
        "safety_norm_chunks",
    ):
        for raw in state.get(field) or []:
            try:
                c = Citation.model_validate(raw.model_dump() if hasattr(raw, "model_dump") else raw)
            except Exception:
                continue
            cid = c.chunk_id or c.citation_id
            if cid and cid not in citations:
                citations[cid] = c

    return citations


def _final_text(answer: FinalAnswer) -> str:
    parts: list[str] = [
        answer.summary,
        answer.diagnosis or "",
        answer.method_selection or "",
        *answer.steps,
        *answer.materials,
        *answer.acceptance_criteria,
        *answer.safety_notes,
        *answer.weather_constraints,
    ]
    if answer.weather_advice is not None:
        parts.extend(
            [
                answer.weather_advice.summary or "",
                *answer.weather_advice.constraints,
                *answer.weather_advice.practical_tips,
            ]
        )
    return "\n".join(part for part in parts if part)


def _method_text(answer: FinalAnswer) -> str:
    return answer.method_selection or answer.summary


def _normalize_method(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[“”\"'《》\s，,。；;：:、（）()]", "", value)
    for suffix in ("处治方法", "处治", "施工", "工法", "方法", "法"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def _method_matches(left: str | None, right: str | None) -> bool:
    norm_left = _normalize_method(left)
    norm_right = _normalize_method(right)
    if not norm_left or not norm_right:
        return False
    return norm_left == norm_right or norm_left in norm_right or norm_right in norm_left


def _selected_candidate(
    candidates: list[SolutionCandidate],
    selected_id: str | None,
    selected_name: str | None,
) -> tuple[int | None, SolutionCandidate | None]:
    for index, candidate in enumerate(candidates):
        if candidate.candidate_id == selected_id:
            return index, candidate
    for index, candidate in enumerate(candidates):
        if _method_matches(candidate.name, selected_name):
            return index, candidate
    return None, None


def _known_source_clauses(citations: Iterable[Citation]) -> set[str]:
    clauses: set[str] = set()
    for citation in citations:
        if not citation.source_clause:
            continue
        clauses.update(split_clause_ids(citation.source_clause))
    return clauses


def _mentioned_clauses(answer: FinalAnswer) -> set[str]:
    return set(clause_ids(_final_text(answer)))


def _append_unique(values: list[str], additions: Iterable[str]) -> list[str]:
    result = list(values)
    for addition in additions:
        if addition and addition not in result:
            result.append(addition)
    return result


def _evidence_quality(citation: Citation) -> float | None:
    quality = (citation.annotations or {}).get(EVIDENCE_QUALITY_KEY)
    return float(quality) if isinstance(quality, (int, float)) else None


def _with_review_updates(
    answer: FinalAnswer | None,
    *,
    force_review: bool,
    safety_notes: list[str],
    evidence_gaps: list[str],
) -> FinalAnswer | None:
    if answer is None:
        return None
    return answer.model_copy(
        update={
            "need_human_review": answer.need_human_review or force_review,
            "safety_notes": _append_unique(answer.safety_notes, safety_notes),
            "evidence_gaps": _append_unique(answer.evidence_gaps, evidence_gaps),
        }
    )


def _presented_message(
    answer: FinalAnswer | None,
    state: AgentState,
    retryable_error: bool,
    retry_count: int,
) -> tuple[str | None, list[ReferenceItem], list[AuditEvent], FinalAnswerDisplay | None]:
    if answer is None:
        return None, [], [], None
    if retryable_error and retry_count <= FIRST_RETRY_COUNT:
        return None, build_reference_index(answer.citations), [], None
    references = build_reference_index(answer.citations)
    if _should_render_tip_only(answer, state):
        locale = state.get("locale") or "zh-CN"
        return (
            _tip_only_message(answer, locale, references),
            references,
            [
                AuditEvent(
                    node_name="safety_critic",
                    message=_critic_message("tip_only_audit", locale),
                )
            ],
            None,
        )
    locale = state.get("locale") or "zh-CN"
    if deterministic_final_renderer_enabled() and can_render_deterministically(
        state.get("loaded_memory"), locale
    ):
        presented = render_final_answer(answer, state.get("loaded_memory"), references)
        return (
            presented.message,
            references,
            _deterministic_presentation_audit(locale, references, presented),
            presented.display,
        )
    presented, messages = present_final_answer(
        answer, state.get("loaded_memory"), references, locale
    )
    if is_english(locale) and presented.display is None:
        raise ValueError("English final presentation must include final_answer_display.")
    return (
        presented.message,
        references,
        _presentation_audit(
            state=state,
            answer=answer,
            references=references,
            messages=messages,
            presented=presented,
        ),
        presented.display,
    )


def _deterministic_presentation_audit(
    locale: str,
    references: list[ReferenceItem],
    presented: PresentedFinalAnswer,
) -> list[AuditEvent]:
    return [
        AuditEvent(
            node_name="deterministic_final_renderer",
            message=_critic_message("deterministic_presenter_audit", locale),
            metadata={
                "used_reference_ids": presented.used_reference_ids,
                "reference_count": len(references),
            },
        )
    ]


def _should_render_tip_only(answer: FinalAnswer, state: AgentState) -> bool:
    if not state.get("final_answer_message"):
        return False
    if state.get("weather_route") == "arrangement_advised":
        return True
    return answer.weather_advice is not None


def _tip_only_message(
    answer: FinalAnswer,
    locale: str,
    references: list[ReferenceItem],
) -> str:
    advice = answer.weather_advice
    english = locale == "en-US"
    title = "**Construction Arrangement Advice**" if english else "**施工安排建议**"
    sections = [title]
    sections.extend(
        _tip_section(
            "Safety Norm Notes" if english else "安全作业规范要点",
            with_clause_citation_tokens(answer.safety_notes, references),
        )
    )
    if advice is not None:
        sections.extend(_tip_section("Safety Notes" if english else "安全提示", advice.constraints))
        sections.extend(_weather_section(advice))
        sections.extend(
            _tip_section(
                "Temporary Work-Zone Tips" if english else "临时组织建议",
                advice.practical_tips,
            )
        )
    if len(sections) == 1:
        sections.extend(
            _tip_section("Evidence Gaps" if english else "证据缺口", answer.evidence_gaps)
        )
    return "\n\n".join(sections)


def _weather_section(advice: WeatherAdvice) -> list[str]:
    items = [item for item in (advice.summary, advice.recommended_window) if item]
    title = "Weather Advice" if advice.title == "Weather Construction Tip" else "天气建议"
    return _tip_section(title, items)


def _tip_section(title: str, items: Iterable[str]) -> list[str]:
    clean = [item.strip() for item in items if item and item.strip()]
    if not clean:
        return []
    return [f"**{title}**\n" + "\n".join(f"- {item}" for item in clean)]


def _presentation_audit(
    *,
    state: AgentState,
    answer: FinalAnswer,
    references: list[ReferenceItem],
    messages: list[object],
    presented: PresentedFinalAnswer,
) -> list[AuditEvent]:
    locale = state.get("locale") or "zh-CN"
    return [
        trace_event(
            node_name="final_response_presenter",
            kind="llm_call",
            title=_critic_message("presenter_title", locale),
            inputs={
                "need_human_review": answer.need_human_review,
                "reference_ids": reference_ids(references),
                "memory_present": state.get("loaded_memory") is not None,
            },
            prompt=messages,
            output=presented,
        ),
        AuditEvent(
            node_name="final_response_presenter",
            message=_critic_message("presenter_audit", locale),
            metadata={
                "used_reference_ids": presented.used_reference_ids,
                "reference_count": len(references),
            },
        ),
    ]


def _check_citations(
    state: AgentState,
    answer: FinalAnswer,
    issues: list[CriticIssue],
) -> bool:
    retryable_error = False
    locale = state.get("locale") or "zh-CN"
    evidence = _evidence_citations(state)
    if not answer.citations:
        issues.append(
            _issue(
                severity="error",
                code="missing_final_citations",
                message=_critic_message("missing_final_citations", locale),
                field_path="final_answer.citations",
            )
        )
        return True

    unsupported = [
        citation.citation_id
        for citation in answer.citations
        if citation.citation_id not in evidence
    ]
    if unsupported:
        retryable_error = True
        issues.append(
            _issue(
                severity="error",
                code="citation_not_from_retrieved_evidence",
                message=_critic_message("citation_not_from_retrieved_evidence", locale),
                field_path="final_answer.citations",
                evidence_ids=unsupported,
            )
        )

    known_clauses = _known_source_clauses(answer.citations)
    unsupported_clauses = {
        clause for clause in _mentioned_clauses(answer) if clause not in known_clauses
    }
    if unsupported_clauses:
        retryable_error = True
        issues.append(
            _issue(
                severity="error",
                code="unsupported_source_clause",
                message=_critic_message("unsupported_source_clause", locale),
                field_path="final_answer",
            )
        )
    return retryable_error


def _check_method_consistency(
    state: AgentState,
    answer: FinalAnswer,
    issues: list[CriticIssue],
) -> bool:
    """User-selected candidate must equal the final answer's method.

    The old hot-repair / rutting hardcoded rule was removed — that class of
    "适配性" check belongs to the Discriminator + LLM-critic layer now.
    """
    retryable_error = False
    locale = state.get("locale") or "zh-CN"
    candidates = state.get("solution_candidates") or []
    selection = state.get("candidate_selection")
    selected_name = selection.selected_name if selection else None
    selected_id = selection.selected_candidate_id if selection else None
    _, selected_candidate = _selected_candidate(candidates, selected_id, selected_name)
    method_text = _method_text(answer)

    if selection and selection.status == "selected":
        if selected_candidate is None:
            retryable_error = True
            issues.append(
                _issue(
                    severity="error",
                    code="selected_candidate_not_found",
                    message=_critic_message("selected_candidate_not_found", locale),
                    field_path="candidate_selection.selected_candidate_id",
                )
            )
        elif not _method_matches(method_text, selected_candidate.name):
            retryable_error = True
            issues.append(
                _issue(
                    severity="error",
                    code="final_method_differs_from_user_selection",
                    message=_critic_message("final_method_differs", locale),
                    field_path="final_answer.method_selection",
                    evidence_ids=selected_candidate.evidence_ids,
                )
            )
    elif candidates and not any(_method_matches(method_text, item.name) for item in candidates):
        retryable_error = True
        issues.append(
            _issue(
                severity="error",
                code="final_method_not_in_candidates",
                message=_critic_message("final_method_not_in_candidates", locale),
                field_path="final_answer.method_selection",
            )
        )

    return retryable_error


def _check_review_triggers(
    state: AgentState,
    issues: list[CriticIssue],
    evidence_gaps: list[str],
) -> bool:
    force_review = False
    locale = state.get("locale") or "zh-CN"
    min_candidate_confidence = _float_env(
        "ROAD_DISTRESS_MIN_CANDIDATE_CONFIDENCE",
        DEFAULT_MIN_CANDIDATE_CONFIDENCE,
    )
    min_evidence_similarity = _float_env(
        "ROAD_DISTRESS_MIN_EVIDENCE_SIMILARITY",
        DEFAULT_MIN_EVIDENCE_SIMILARITY,
    )
    candidates = state.get("solution_candidates") or []
    selection = state.get("candidate_selection")
    selected_index, selected_candidate = _selected_candidate(
        candidates,
        selection.selected_candidate_id if selection else None,
        selection.selected_name if selection else None,
    )

    if candidates and candidates[0].rough_confidence < min_candidate_confidence:
        force_review = True
        evidence_gaps.append("top_candidate_confidence_below_threshold")
        issues.append(
            _issue(
                severity="warning",
                code="top_candidate_low_confidence",
                message=_critic_message("top_candidate_low_confidence", locale),
                field_path="solution_candidates[0].rough_confidence",
                evidence_ids=candidates[0].evidence_ids,
            )
        )

    if selected_candidate and selected_candidate.rough_confidence < min_candidate_confidence:
        force_review = True
        evidence_gaps.append("selected_candidate_confidence_below_threshold")
        issues.append(
            _issue(
                severity="warning",
                code="selected_candidate_low_confidence",
                message=_critic_message("selected_candidate_low_confidence", locale),
                field_path="candidate_selection.selected_candidate_id",
                evidence_ids=selected_candidate.evidence_ids,
            )
        )

    if selected_index is not None and selected_index > 0:
        force_review = True
        evidence_gaps.append("user_selected_non_top1_candidate")
        issues.append(
            _issue(
                severity="warning",
                code="user_selected_non_top1",
                message=_critic_message("user_selected_non_top1", locale),
                field_path="candidate_selection.selected_candidate_id",
                evidence_ids=selected_candidate.evidence_ids if selected_candidate else (),
            )
        )

    for purpose, bundle in _bundles(state).items():
        citation_ids = [citation.citation_id for citation in bundle.citations]
        if not bundle.sufficient:
            force_review = True
            reason = bundle.gap_reason or "evidence_insufficient"
            evidence_gaps.append(f"{purpose}: {reason}")
            issues.append(
                _issue(
                    severity="warning",
                    code="evidence_bundle_insufficient",
                    message=_critic_message(
                        "evidence_bundle_insufficient",
                        locale,
                        purpose=purpose,
                    ),
                    field_path=f"evidence_bundles.{purpose}",
                    evidence_ids=citation_ids,
                )
            )
        qualities = [
            quality
            for citation in bundle.citations
            if (quality := _evidence_quality(citation)) is not None
        ]
        if qualities and max(qualities) < min_evidence_similarity:
            force_review = True
            evidence_gaps.append(f"{purpose}: evidence_quality_below_threshold")
            issues.append(
                _issue(
                    severity="warning",
                    code="evidence_quality_below_threshold",
                    message=_critic_message(
                        "evidence_quality_below_threshold",
                        locale,
                        purpose=purpose,
                    ),
                    field_path=f"evidence_bundles.{purpose}.citations",
                    evidence_ids=citation_ids,
                )
            )
    return force_review


def _check_weather(
    state: AgentState,
    answer: FinalAnswer,
    issues: list[CriticIssue],
    safety_notes: list[str],
) -> bool:
    retryable_error = False
    weather_context = state.get("weather_context")
    weather_advice = state.get("weather_advice") or answer.weather_advice
    text = _final_text(answer)
    english = state.get("locale") == "en-US"

    if weather_advice and weather_advice.status == "skipped":
        terms = (
            ("weather was not verified", "skipped weather", "verify local weather")
            if english
            else ("未进行天气校验", "未校验天气", "复查天气", "复查当地", "复核天气", "复核当地")
        )
        if not any(term in text.lower() for term in terms):
            note = (
                "The user skipped weather verification; re-check the local 24-hour and "
                "3-day forecast before formal construction."
                if english
                else "用户已跳过天气校验；正式施工前应复查当地 24 小时和 3 天天气。"
            )
            safety_notes.append(note)
            issues.append(
                _issue(
                    severity="info",
                    code="weather_skip_note_added",
                    message=_critic_message("weather_skip_note_added", state.get("locale")),
                    field_path="final_answer.safety_notes",
                )
            )
        return False

    if weather_context is not None and weather_context.status != "available":
        hallucinated = (
            re.search(r"next\s*24\s*hours.*suitable\s*for\s*formal\s*construction", text, re.I)
            if english
            else re.search(r"未来\s*24\s*小时.*适合正式施工", text)
        )
        if hallucinated:
            retryable_error = True
            issues.append(
                _issue(
                    severity="error",
                    code="weather_hallucination",
                    message=_critic_message("weather_hallucination", state.get("locale")),
                    field_path="final_answer.weather_advice",
                )
            )

    if weather_advice and weather_advice.next_24h_suitable is False:
        blocking_terms = (
            ("not suitable", "postpone", "rain", "drainage", "dry")
            if english
            else ("不适合正式施工", "暂缓", "降雨", "排水", "干燥")
        )
        if not any(term in text.lower() for term in blocking_terms):
            note = (
                "Weather advice indicates the next 24 hours are not suitable for formal "
                "construction; drain the site and wait for a dry work window."
                if english
                else "天气建议显示未来 24 小时不适合正式施工，应先排水并等待干燥窗口。"
            )
            safety_notes.append(note)
            issues.append(
                _issue(
                    severity="warning",
                    code="weather_blocking_note_added",
                    message=_critic_message("weather_blocking_note_added", state.get("locale")),
                    field_path="final_answer.safety_notes",
                )
            )
    return retryable_error


def safety_critic(state: AgentState) -> AgentState:
    issues: list[CriticIssue] = []
    safety_notes: list[str] = []
    evidence_gaps: list[str] = []
    retryable_error = False
    answer = state.get("final_answer")

    if answer is None:
        issues.append(
            _issue(
                severity="error",
                code="missing_final_answer",
                message=_critic_message("missing_final_answer", state.get("locale")),
                field_path="final_answer",
            )
        )
        retry_count = state.get("critic_retry_count", 0) + 1
        return {
            "safety_review": SafetyReview(passed=False, issues=issues),
            "critic_retry_count": retry_count,
            "phase": WorkflowPhase.SAFETY,
            "next_action": "rewrite_answer",
            "awaiting_user_input": False,
            "interrupt": None,
            "audit_log": [
                AuditEvent(
                    node_name="safety_critic",
                    message=_critic_message("missing_final_answer_audit", state.get("locale")),
                    metadata={"retry_count": retry_count},
                )
            ],
        }

    retryable_error = _check_citations(state, answer, issues) or retryable_error
    retryable_error = _check_method_consistency(state, answer, issues) or retryable_error
    force_review = _check_review_triggers(state, issues, evidence_gaps)
    retryable_error = _check_weather(state, answer, issues, safety_notes) or retryable_error

    if retryable_error:
        force_review = True
        safety_notes.append(
            (
                "The safety review found citation or treatment-consistency issues and "
                "triggered one rewrite; manual review is required before execution."
            )
            if state.get("locale") == "en-US"
            else "安全审查发现答案依据或方案一致性问题，已触发一次重写；落地前需人工复核。"
        )
        evidence_gaps.append("safety_critic_retryable_error")

    retry_count = (
        state.get("critic_retry_count", 0) + 1
        if retryable_error
        else state.get(
            "critic_retry_count",
            0,
        )
    )
    patched_answer = _with_review_updates(
        answer,
        force_review=force_review,
        safety_notes=safety_notes,
        evidence_gaps=evidence_gaps,
    )
    review = SafetyReview(
        passed=not retryable_error,
        issues=issues,
        forced_need_human_review=force_review,
    )
    final_message, references, presentation_audit, display = _presented_message(
        patched_answer,
        state,
        retryable_error,
        retry_count,
    )
    return {
        "final_answer": patched_answer,
        "final_answer_message": final_message,
        "final_answer_display": display or state.get("final_answer_display"),
        "reference_index": references,
        "safety_review": review,
        "critic_retry_count": retry_count,
        "phase": WorkflowPhase.SAFETY,
        "next_action": "rewrite_answer" if retryable_error else "write_memory",
        "awaiting_user_input": False,
        "interrupt": None,
        "audit_log": [
            *presentation_audit,
            AuditEvent(
                node_name="safety_critic",
                message=_critic_message(
                    "retryable_audit" if retryable_error else "completed_audit",
                    state.get("locale"),
                ),
                metadata={
                    "passed": review.passed,
                    "issue_count": len(issues),
                    "safety_issue_count": len(issues),
                    "safety_issue_codes": [issue.code for issue in issues],
                    "forced_need_human_review": force_review,
                    "retry_count": retry_count,
                    "min_candidate_confidence": _float_env(
                        "ROAD_DISTRESS_MIN_CANDIDATE_CONFIDENCE",
                        DEFAULT_MIN_CANDIDATE_CONFIDENCE,
                    ),
                    "min_evidence_similarity": _float_env(
                        "ROAD_DISTRESS_MIN_EVIDENCE_SIMILARITY",
                        DEFAULT_MIN_EVIDENCE_SIMILARITY,
                    ),
                },
            ),
        ],
    }
