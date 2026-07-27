# ─────────────────────────────────────────────────────────────
# REFACTOR NOTES (new road_rag_design.md)
# 保留：WorkflowPhase（用于状态机阶段标记）。
# 删除：DistressType（8 值枚举）。新设计明确反对跨文档枚举：
#   不同标准对"病害类型"的层级和命名不同，硬编码枚举会失效。
#   病害类别改为 LLM 自由文本字段 DistressInput.defect_category。
# 受影响：state.DistressInput / safety_critic / retrieval tools —
#         已同步去掉对 DistressType 的引用。
# ─────────────────────────────────────────────────────────────
"""Domain enums shared across graph state and tools."""

from enum import Enum


class WorkflowPhase(str, Enum):
    INPUT = "input"
    CONTEXT = "context"
    EVIDENCE = "evidence"
    CANDIDATE = "candidate"
    DETAIL = "detail"
    SAFETY = "safety"
    DONE = "done"
