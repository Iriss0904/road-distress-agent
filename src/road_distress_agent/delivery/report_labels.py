"""Localized labels for deterministic report rendering."""

from __future__ import annotations

_LABELS_ZH = {
    "title": "道路巡查报告",
    "unit_name": "单位名称/部门",
    "report_date": "报告日期",
    "overview": "一、巡查概况",
    "task": "巡查任务",
    "inspection_time": "巡查时间",
    "segment": "路段桩号",
    "inspection_method": "巡查方式",
    "participants": "参与人员",
    "crew": "责任班组",
    "defects_found": "发现病害",
    "results": "二、巡查内容与结果",
    "pavement_condition": "1. 路基路面状况",
    "overall": "总体状况评价",
    "features": "现场特征",
    "risk": "风险判断",
    "recommendation": "处治建议",
    "process": "1、工艺要求",
    "quality": "2、质量保证",
    "citations": "3、规范依据",
    "actions": "三、处置与建议",
    "maintenance_needed": "需安排小修保养的病害",
    "cost": "工程量及费用预估",
    "cost_sheet_ref": "造价明细见单独造价表。",
    "coordination": "其他需协调的问题",
    "attachments": "四、附件清单",
    "attachment_records": "巡查记录表（原始件）",
    "attachment_photos": "病害照片及位置示意图",
    "attachment_cost": "工程量与造价估算表",
}

_LABELS_EN = {
    "title": "Road Inspection Report",
    "unit_name": "Unit / Department",
    "report_date": "Report Date",
    "overview": "1. Inspection Overview",
    "task": "Inspection Task",
    "inspection_time": "Inspection Time",
    "segment": "Road Segment / Stake",
    "inspection_method": "Inspection Method",
    "participants": "Participants",
    "crew": "Responsible Crew",
    "defects_found": "Defects Found",
    "results": "2. Inspection Findings",
    "pavement_condition": "1. Pavement Condition",
    "overall": "Overall Assessment",
    "features": "Site Features",
    "risk": "Risk Assessment",
    "recommendation": "Treatment Recommendation",
    "process": "1. Process Requirements",
    "quality": "2. Quality Assurance",
    "citations": "3. Standards References",
    "actions": "3. Actions and Recommendations",
    "maintenance_needed": "Defects Requiring Maintenance",
    "cost": "Quantity and Cost Estimate",
    "cost_sheet_ref": "See the separate cost workbook for details.",
    "coordination": "Coordination Items",
    "attachments": "4. Attachment List",
    "attachment_records": "Inspection records (original)",
    "attachment_photos": "Distress photos and location sketch",
    "attachment_cost": "Quantity and cost estimate workbook",
}


def report_label(key: str, locale: str) -> str:
    labels = _LABELS_EN if locale == "en-US" else _LABELS_ZH
    return labels[key]
