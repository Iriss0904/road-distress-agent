"""Central model-tier policy for runtime LLM calls."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

MODEL_TIERING_ENV = "ROAD_DISTRESS_NONCRITICAL_MODEL_TIERING_ENABLED"
PRO_MODEL_ENV = "DEEPSEEK_MODEL"
DEFAULT_PRO_MODEL = "deepseek-v4-pro"
FLASH_MODEL = "deepseek-v4-flash"
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ModelTier(str, Enum):
    PRO = "pro"
    FLASH = "flash"


FLASH_NODE_NAMES = frozenset(
    {
        "top_router",
        "disease_query_rewriter",
        "method_query_rewriter",
        "kb_query_rewriter",
        "diagnosis_reconcile",
        "eval",
        "eval_baseline",
        "kb_hop_retriever",
        "kb_retriever",
        "detail_retriever_v2",
        "disease_retriever",
        "method_retriever",
    }
)
MODEL_TIER_POLICY: Mapping[str, ModelTier] = MappingProxyType(
    {node_name: ModelTier.FLASH for node_name in FLASH_NODE_NAMES}
)


@dataclass(frozen=True)
class ModelAssignment:
    node_name: str
    tier: ModelTier
    model: str


def model_tiering_enabled() -> bool:
    raw = os.environ.get(MODEL_TIERING_ENV)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid {MODEL_TIERING_ENV}={raw!r}; expected a boolean value.")


def configured_pro_model() -> str:
    raw = os.environ.get(PRO_MODEL_ENV)
    return raw.strip() if raw and raw.strip() else DEFAULT_PRO_MODEL


def model_assignment(
    node_name: str,
    *,
    policy: Mapping[str, ModelTier] = MODEL_TIER_POLICY,
) -> ModelAssignment:
    if not node_name.strip():
        raise ValueError("LLM model policy requires a non-empty node name.")
    configured_tier = policy.get(node_name, ModelTier.PRO)
    if not isinstance(configured_tier, ModelTier):
        raise ValueError(f"Invalid model tier for {node_name!r}: {configured_tier!r}.")
    tier = configured_tier if model_tiering_enabled() else ModelTier.PRO
    model = FLASH_MODEL if tier == ModelTier.FLASH else configured_pro_model()
    return ModelAssignment(node_name=node_name, tier=tier, model=model)


def model_policy_manifest() -> dict[str, Any]:
    enabled = model_tiering_enabled()
    return {
        "enabled": enabled,
        "switch_environment": MODEL_TIERING_ENV,
        "models": {"pro": configured_pro_model(), "flash": FLASH_MODEL},
        "node_assignments": {
            name: model_assignment(name).tier.value for name in sorted(MODEL_TIER_POLICY)
        },
        "unlisted_node_tier": ModelTier.PRO.value,
    }
