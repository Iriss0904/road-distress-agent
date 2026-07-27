"""Vision subgraph scaffold."""

from __future__ import annotations

from typing import Any

from road_distress_agent.errors import BoundaryError, ErrorCategory, make_error_info
from road_distress_agent.state import AgentState, AuditEvent, ErrorEvent, SceneContext
from road_distress_agent.tools.vlm import make_vision_scene_describer
from road_distress_agent.tracing import trace_event


def _image_attachments(state: AgentState) -> list[Any]:
    return [
        attachment
        for attachment in state.get("latest_attachments", [])
        if attachment.media_type.startswith("image/")
    ]


def _no_image_delta() -> AgentState:
    return {
        "scene_description": None,
        "scene_context": None,
        "audit_log": [
            AuditEvent(node_name="vision_subgraph", message="No image attachment to inspect.")
        ],
    }


def _error_delta(attachment: Any, exc: Exception) -> AgentState:
    info = _error_info(exc)
    return {
        "scene_description": None,
        "scene_context": None,
        "errors": [
            ErrorEvent.from_info(
                node_name="vision_subgraph",
                recoverable=True,
                info=info,
                surface_to_user=True,
            )
        ],
        "audit_log": [
            AuditEvent(
                node_name="vision_subgraph",
                message="Vision scene description failed; continuing without image context.",
                metadata={"image_uri": attachment.uri},
            )
        ],
    }


def _error_info(exc: Exception):
    if isinstance(exc, BoundaryError):
        return exc.info
    return make_error_info(
        domain="VLM",
        step="CALL",
        category=ErrorCategory.INTERNAL,
        responsibility="图像识别失败",
        reason=f"{exc.__class__.__name__}: {exc}",
        hint="查看 raw 定位 VLM 调用或图片处理异常。",
        raw=f"{exc.__class__.__name__}: {exc}",
        retriable=False,
    )


def _success_delta(
    *,
    state: AgentState,
    attachment: Any,
    describer: Any,
    scene: str,
) -> AgentState:
    return {
        "scene_description": scene,
        "scene_context": SceneContext(
            scene_description=scene,
            source_attachment_id=attachment.attachment_id,
            source_uri=attachment.uri,
        ),
        "audit_log": [
            trace_event(
                node_name="vision_subgraph",
                kind="llm_call",
                title="Vision scene description",
                inputs={
                    "image_uri": attachment.uri,
                    "media_type": attachment.media_type,
                    "latest_user_text": state.get("latest_user_text") or "",
                },
                output={"scene_description": scene},
                metadata={"describer": describer.__class__.__name__},
            ),
            AuditEvent(
                node_name="vision_subgraph",
                message="Vision scene description produced.",
                metadata={
                    "image_uri": attachment.uri,
                    "media_type": attachment.media_type,
                    "attachment_id": attachment.attachment_id,
                },
            ),
        ],
    }


def vision_subgraph(state: AgentState) -> AgentState:
    attachments = _image_attachments(state)
    if not attachments:
        return _no_image_delta()

    attachment = attachments[0]
    describer = make_vision_scene_describer()
    try:
        scene = describer.describe(
            attachment.uri,
            user_text=state.get("latest_user_text"),
            locale=state.get("locale") or "zh-CN",
        )
    except Exception as exc:
        return _error_delta(attachment, exc)
    return _success_delta(
        state=state,
        attachment=attachment,
        describer=describer,
        scene=scene,
    )
