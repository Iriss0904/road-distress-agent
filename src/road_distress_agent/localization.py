"""Locale helpers shared by API, graph nodes, delivery, and web snapshots."""

from __future__ import annotations

from typing import Literal

Locale = Literal["zh-CN", "en-US"]

DEFAULT_LOCALE: Locale = "zh-CN"
SUPPORTED_LOCALES = frozenset({"zh-CN", "en-US"})

_STAGE_LABELS_ZH: dict[str, str] = {
    "parallel_context_loader": "正在加载上下文",
    "top_router": "正在判断本轮意图",
    "vision_subgraph": "正在分析图片现场信息",
    "intent_router": "正在接收您的补充",
    "handle_off_intent": "请回到上一个问题",
    "off_topic_refuser": "正在返回边界说明",
    "diagnosis_reconcile": "正在更新现场事实",
    "disease_selection_handler": "正在处理病害候选选择",
    "disease_query_rewriter": "正在生成病害检索问题",
    "disease_retriever": "正在检索病害判定依据",
    "disease_continue_discriminator": "正在继续判别病害",
    "disease_discriminator": "正在判别病害类型",
    "disease_result_router": "正在处理病害判别结果",
    "method_selection_handler": "正在处理方法候选选择",
    "method_query_rewriter": "正在生成处治方案检索问题",
    "method_retriever": "正在检索处治方案依据",
    "method_discriminator": "正在判别处治方案",
    "method_result_router": "正在处理方法判别结果",
    "detail_retriever_v2": "正在检索施工细节依据",
    "kb_query_planner": "正在规划知识检索路径",
    "kb_query_rewriter": "正在生成知识检索问题",
    "kb_retriever": "正在检索道路养护知识",
    "kb_direct_meta_answer": "正在返回知识问答说明",
    "kb_answer_composer": "正在生成知识回答",
    "kb_hop_retriever": "正在分步检索道路养护知识",
    "kb_planned_answer_composer": "正在合成多证据知识回答",
    "kb_clarification_composer": "正在生成条件化知识回答",
    "answer_composer": "正在生成施工建议",
    "weather_location_handler": "正在解析施工安排输入",
    "safety_norm_rewriter": "正在生成安全规程检索问题",
    "safety_norm_retriever": "正在检索安全作业规范",
    "address_weather_loader": "正在加载天气数据",
    "construction_arrangement_advisor": "正在生成施工安排建议",
    "weather_advisor": "正在生成天气施工建议",
    "construction_tip_offer": "等待确认施工安排建议",
    "safety_critic": "正在进行安全审查",
    "memory_writer": "正在保存对话记忆",
    "archive_intake": "正在锁定巡查任务",
    "ledger_loader": "正在读取任务台账",
    "dedup_resolver": "正在生成归档预览",
    "dedup_confirm_gate": "正在确认归档清单",
    "delivery_supervisor": "正在分派交付专家",
    "cost_quantity_agent": "正在生成造价表",
    "report_agent": "正在生成巡查报告",
    "work_order_agent": "正在生成施工工单",
    "compliance_critic": "正在检查交付合规性",
    "delivery_packager": "正在打包交付物",
    "project_memory_writer": "正在写入项目记忆",
}

_STAGE_LABELS_EN: dict[str, str] = {
    "parallel_context_loader": "Loading context",
    "top_router": "Routing this turn",
    "vision_subgraph": "Analyzing the site image",
    "intent_router": "Reading your follow-up",
    "handle_off_intent": "Returning to the pending question",
    "off_topic_refuser": "Preparing boundary response",
    "diagnosis_reconcile": "Updating site facts",
    "disease_selection_handler": "Processing distress selection",
    "disease_query_rewriter": "Building Chinese distress-search query",
    "disease_retriever": "Retrieving distress evidence",
    "disease_continue_discriminator": "Continuing distress classification",
    "disease_discriminator": "Classifying road distress",
    "disease_result_router": "Preparing distress confirmation",
    "method_selection_handler": "Processing treatment selection",
    "method_query_rewriter": "Building Chinese treatment-search query",
    "method_retriever": "Retrieving treatment evidence",
    "method_discriminator": "Ranking treatment options",
    "method_result_router": "Preparing treatment confirmation",
    "detail_retriever_v2": "Retrieving construction details",
    "kb_query_planner": "Planning knowledge retrieval",
    "kb_query_rewriter": "Building Chinese knowledge-search query",
    "kb_retriever": "Retrieving maintenance knowledge",
    "kb_direct_meta_answer": "Returning knowledge-base meta answer",
    "kb_answer_composer": "Composing knowledge answer",
    "kb_hop_retriever": "Retrieving planned maintenance evidence",
    "kb_planned_answer_composer": "Composing planned knowledge answer",
    "kb_clarification_composer": "Composing conditional knowledge answer",
    "answer_composer": "Composing structured recommendation",
    "weather_location_handler": "Parsing construction arrangement input",
    "safety_norm_rewriter": "Building safety-norm query",
    "safety_norm_retriever": "Retrieving work-zone safety norms",
    "address_weather_loader": "Loading weather data",
    "construction_arrangement_advisor": "Composing construction arrangement advice",
    "weather_advisor": "Composing weather construction tips",
    "construction_tip_offer": "Waiting for construction-arrangement choice",
    "safety_critic": "Running safety review",
    "memory_writer": "Saving conversation memory",
    "archive_intake": "Locking the inspection project",
    "ledger_loader": "Loading the project ledger",
    "dedup_resolver": "Preparing archive preview",
    "dedup_confirm_gate": "Confirming archive list",
    "delivery_supervisor": "Dispatching delivery specialists",
    "cost_quantity_agent": "Generating cost workbook",
    "report_agent": "Generating inspection report",
    "work_order_agent": "Generating work order",
    "compliance_critic": "Checking delivery compliance",
    "delivery_packager": "Packaging deliverables",
    "project_memory_writer": "Writing project memory",
}

_DEFECT_EN = {
    "坑槽": "Pothole",
    "横向裂缝": "Transverse crack",
    "纵向裂缝": "Longitudinal crack",
    "龟裂": "Alligator cracking",
    "块裂": "Block cracking",
    "车辙": "Rutting",
    "沉陷": "Settlement",
    "松散": "Raveling",
    "板角断裂": "Corner break",
    "错台": "Faulting",
    "渗漏水": "Water leakage",
}

_METHOD_EN = {
    "清缝并灌缝": "Clean and seal cracks",
    "扩缝并灌缝": "Widen and seal cracks",
    "开槽灌缝": "Rout and seal cracks",
    "局部挖补法": "Localized patch repair",
    "坑槽热补": "Hot-mix pothole patching",
    "铣刨重铺": "Mill and overlay",
    "板角断裂专属维修流程": "Corner-break repair workflow",
}

_EN_TO_CANONICAL = {
    "pothole": "坑槽",
    "transverse crack": "横向裂缝",
    "longitudinal crack": "纵向裂缝",
    "alligator cracking": "龟裂",
    "block cracking": "块裂",
    "rutting": "车辙",
    "settlement": "沉陷",
    "raveling": "松散",
    "corner break": "板角断裂",
    "faulting": "错台",
    "clean and seal": "清缝并灌缝",
    "clean and seal cracks": "清缝并灌缝",
    "widen and seal": "扩缝并灌缝",
    "rout and seal": "开槽灌缝",
    "localized patch": "局部挖补法",
    "hot-mix pothole patching": "坑槽热补",
    "mill and overlay": "铣刨重铺",
}


def normalize_locale(locale: str | None) -> Locale:
    value = (locale or DEFAULT_LOCALE).strip()
    if value not in SUPPORTED_LOCALES:
        raise ValueError(f"unsupported locale: {value!r}")
    return value  # type: ignore[return-value]


def is_english(locale: str | None) -> bool:
    return normalize_locale(locale) == "en-US"


def stage_label(node_name: str, locale: str | None = None) -> str:
    labels = _STAGE_LABELS_EN if is_english(locale) else _STAGE_LABELS_ZH
    return labels.get(node_name, node_name)


def display_term(value: str | None, locale: str | None = None) -> str:
    if not value:
        return ""
    if not is_english(locale):
        return value
    return _DEFECT_EN.get(value) or _METHOD_EN.get(value) or value


def canonical_from_english_text(text: str) -> list[str]:
    normalized = " ".join(text.lower().replace("-", " ").split())
    return [value for key, value in _EN_TO_CANONICAL.items() if key in normalized]


def user_language_instruction(locale: str | None) -> str:
    if is_english(locale):
        return (
            "User-facing text must be in English. Keep Chinese standard excerpts, "
            "clause IDs, and canonical evidence names unchanged when cited."
        )
    return "面向用户的文字必须使用中文。"


def query_language_instruction() -> str:
    return "Regardless of the user's language, retrieval queries must be written in Chinese."
