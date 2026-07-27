"""Prepare and bind one Web API turn before SSE headers are sent."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from road_distress_agent.api.turn_results import (
    TurnResultClaim,
    claim_turn_result,
    input_fingerprint,
)
from road_distress_agent.state import AttachmentRef

MAX_REQUEST_ID_LENGTH = 128


class InvalidRequestIdError(ValueError):
    """Raised for an empty or oversized caller-supplied request ID."""


@dataclass(frozen=True)
class PreparedTurn:
    request_id: str
    fingerprint: str
    claim: TurnResultClaim


def prepare_turn_request(
    db_path: str,
    *,
    request_id: str | None,
    thread_id: str,
    user_id: str,
    locale: str,
    text: str,
    attachments: list[AttachmentRef],
) -> PreparedTurn:
    active_request_id = _request_id(request_id)
    fingerprint = input_fingerprint(
        thread_id=thread_id,
        user_id=user_id,
        locale=locale,
        text=text,
        attachments=attachments,
    )
    claim = claim_turn_result(
        db_path,
        request_id=active_request_id,
        fingerprint=fingerprint,
        thread_id=thread_id,
    )
    return PreparedTurn(active_request_id, fingerprint, claim)


def _request_id(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    cleaned = value.strip()
    if not cleaned or len(cleaned) > MAX_REQUEST_ID_LENGTH:
        raise InvalidRequestIdError("request_id 必须为 1 到 128 个字符。")
    return cleaned
