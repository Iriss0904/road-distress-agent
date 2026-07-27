"""Strict tool schemas for HITL disease and treatment discriminators."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from road_distress_agent.nodes.strict_tool_calling import StrictToolResult
from road_distress_agent.state import (
    DiseaseCandidate,
    DiseaseDiscriminatorOutput,
    MethodDiscriminatorOutput,
    MethodOption,
)

MAX_CANDIDATES = 3


class StrictToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiseaseCandidateTool(StrictToolModel):
    name: str
    description: str
    confidence: float

    @field_validator("name", "description")
    @classmethod
    def _text_required(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        return _confidence(value)


class MethodCandidateTool(StrictToolModel):
    name: str
    reason: str
    confidence: float

    @field_validator("name", "reason")
    @classmethod
    def _text_required(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        return _confidence(value)


class PresentDiseaseCandidates(StrictToolModel):
    """Return 1-3 ranked distress candidates for user confirmation."""

    candidates: list[DiseaseCandidateTool]

    @field_validator("candidates")
    @classmethod
    def _candidate_count(cls, value: list[DiseaseCandidateTool]) -> list[DiseaseCandidateTool]:
        return _candidate_list(value)


class AskDiseaseClarification(StrictToolModel):
    """Ask one high-value site question when first-turn evidence is insufficient."""

    missing_feature: str
    clarifying_question: str

    @field_validator("missing_feature", "clarifying_question")
    @classmethod
    def _text_required(cls, value: str) -> str:
        return _non_empty(value)


class PresentMethodCandidates(StrictToolModel):
    """Return 1-3 ranked treatment candidates for user confirmation."""

    candidates: list[MethodCandidateTool]

    @field_validator("candidates")
    @classmethod
    def _candidate_count(cls, value: list[MethodCandidateTool]) -> list[MethodCandidateTool]:
        return _candidate_list(value)


class AskMethodClarification(StrictToolModel):
    """Ask one high-value site question when first-turn evidence is insufficient."""

    missing_feature: str
    clarifying_question: str

    @field_validator("missing_feature", "clarifying_question")
    @classmethod
    def _text_required(cls, value: str) -> str:
        return _non_empty(value)


def disease_tools(attempts: int) -> tuple[type[BaseModel], ...]:
    if attempts >= 1:
        return (PresentDiseaseCandidates,)
    return (PresentDiseaseCandidates, AskDiseaseClarification)


def method_tools(attempts: int) -> tuple[type[BaseModel], ...]:
    if attempts >= 1:
        return (PresentMethodCandidates,)
    return (PresentMethodCandidates, AskMethodClarification)


def disease_output_from_tool(result: StrictToolResult) -> DiseaseDiscriminatorOutput:
    payload = result.parsed
    if isinstance(payload, PresentDiseaseCandidates):
        return DiseaseDiscriminatorOutput(
            sufficient=False,
            candidates=[
                DiseaseCandidate(
                    name=item.name,
                    description=item.description,
                    confidence=item.confidence,
                )
                for item in payload.candidates
            ],
        )
    if isinstance(payload, AskDiseaseClarification):
        return DiseaseDiscriminatorOutput(
            sufficient=False,
            missing_feature=payload.missing_feature,
            clarifying_question=payload.clarifying_question,
        )
    raise TypeError(f"Unsupported disease discriminator tool: {result.tool_name}")


def method_output_from_tool(result: StrictToolResult) -> MethodDiscriminatorOutput:
    payload = result.parsed
    if isinstance(payload, PresentMethodCandidates):
        return MethodDiscriminatorOutput(
            sufficient=False,
            candidates=[
                MethodOption(
                    name=item.name,
                    reason=item.reason,
                    confidence=item.confidence,
                )
                for item in payload.candidates
            ],
        )
    if isinstance(payload, AskMethodClarification):
        return MethodDiscriminatorOutput(
            sufficient=False,
            missing_feature=payload.missing_feature,
            clarifying_question=payload.clarifying_question,
        )
    raise TypeError(f"Unsupported method discriminator tool: {result.tool_name}")


def _non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("Text fields must not be empty.")
    return value


def _confidence(value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0.")
    return value


def _candidate_list(value: list) -> list:
    if len(value) < 1 or len(value) > MAX_CANDIDATES:
        raise ValueError(f"candidates must contain 1-{MAX_CANDIDATES} items.")
    return value
