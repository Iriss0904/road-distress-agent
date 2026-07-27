"""FastAPI server exposing the road-distress workflow to the web chat UI.

The graph itself is unchanged; this layer only wraps it in HTTP + SSE so the
single-page frontend can drive new/resume turns the same way the CLI does.
"""

from __future__ import annotations

import hashlib
import os
from mimetypes import guess_type
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from road_distress_agent.api import knowledge_routes, ledger_routes
from road_distress_agent.api.delivery_routes import router as delivery_router
from road_distress_agent.api.paths import PROJECT_ROOT, data_dir, web_db_path
from road_distress_agent.api.projects_routes import router as projects_router
from road_distress_agent.api.serialization import serialize_snapshot
from road_distress_agent.api.thread_history import (
    get_thread,
    rename_thread,
    soft_delete_thread,
)
from road_distress_agent.api.thread_status_service import threads_with_status
from road_distress_agent.api.turn_request import InvalidRequestIdError, prepare_turn_request
from road_distress_agent.api.turn_results import RequestIdConflictError
from road_distress_agent.api.turn_stream import turn_event_stream
from road_distress_agent.api.web_static import WebStaticFiles, index_response
from road_distress_agent.checkpointing import sqlite_checkpointer
from road_distress_agent.graph import build_graph
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.store import _project_db_path
from road_distress_agent.settings import apply_runtime_mode, runtime_profile
from road_distress_agent.state import AttachmentRef

load_dotenv()


WEB_DIR = PROJECT_ROOT / "frontend" / "dist"
DB_PATH = str(web_db_path())
UPLOAD_DIR = Path(os.environ.get("ROAD_DISTRESS_WEB_UPLOADS", data_dir() / "uploads"))
DEFAULT_USER_ID = "web_user_001"
MAX_IMAGE_BYTES = 12 * 1024 * 1024

app = FastAPI(title="Road Distress Agent Web UI")
app.include_router(projects_router)
app.include_router(delivery_router)
app.include_router(knowledge_routes.router)
app.include_router(ledger_routes.router)


class RenameThreadRequest(BaseModel):
    title: str


@app.on_event("startup")
def _startup() -> None:
    apply_runtime_mode(os.environ.get("ROAD_DISTRESS_RUN_MODE"))
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(image: UploadFile, locale: str) -> AttachmentRef:
    media_type = image.content_type or guess_type(image.filename or "")[0]
    if not media_type or not media_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=_api_message("image_type", locale))
    data = image.file.read()
    if not data:
        raise HTTPException(status_code=400, detail=_api_message("image_empty", locale))
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=_api_message("image_too_large", locale))

    suffix = Path(image.filename or "").suffix or ".img"
    stored = UPLOAD_DIR / f"upload-{uuid4().hex[:12]}{suffix}"
    stored.write_bytes(data)
    return AttachmentRef(
        uri=str(stored.resolve()),
        media_type=media_type,
        sha256=hashlib.sha256(data).hexdigest(),
    )


@app.post("/api/turn")
def post_turn(
    text: str = Form(...),
    thread_id: str | None = Form(None),
    user_id: str = Form(DEFAULT_USER_ID),
    locale: str = Form(DEFAULT_LOCALE),
    image: UploadFile | None = File(None),
    request_id: str | None = Form(None),
) -> StreamingResponse:
    """Run or resume one workflow turn, streaming node progress then a snapshot."""
    try:
        active_locale = normalize_locale(locale)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=_api_message("empty_text", active_locale),
        )

    attachments = [_save_upload(image, active_locale)] if image is not None else []
    active_thread_id = thread_id or f"thread-{uuid4().hex[:8]}"
    try:
        prepared = prepare_turn_request(
            DB_PATH,
            request_id=request_id,
            thread_id=active_thread_id,
            user_id=user_id,
            locale=active_locale,
            text=cleaned,
            attachments=attachments,
        )
    except (InvalidRequestIdError, RequestIdConflictError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return StreamingResponse(
        turn_event_stream(
            thread_id=active_thread_id,
            text=cleaned,
            attachments=attachments,
            user_id=user_id,
            locale=active_locale,
            request_id=prepared.request_id,
            fingerprint=prepared.fingerprint,
            record_user_input=prepared.claim.is_new,
            replay_snapshot=prepared.claim.snapshot,
            db_path=DB_PATH,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": prepared.request_id,
        },
    )


@app.get("/api/state/{thread_id}")
def get_state(thread_id: str) -> dict[str, Any]:
    """Return the current snapshot for a conversation (used on reconnect)."""
    with sqlite_checkpointer(DB_PATH) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="未找到该会话。")
    return serialize_snapshot(snapshot, thread_id)


@app.get("/api/threads")
def list_thread_history(
    user_id: str = DEFAULT_USER_ID,
    q: str | None = None,
) -> list[dict[str, Any]]:
    """List chat threads for the collapsible history sidebar."""
    return threads_with_status(DB_PATH, _project_db_path(), user_id=user_id, q=q)


@app.get("/api/threads/{thread_id}")
def get_thread_history(thread_id: str) -> dict[str, Any]:
    """Return transcript plus the current graph snapshot for one thread."""
    thread = get_thread(DB_PATH, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="未找到该会话。")
    with sqlite_checkpointer(DB_PATH) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if snapshot.values:
        return {
            **thread,
            "snapshot": serialize_snapshot(snapshot, thread_id),
            "snapshot_source": "checkpoint",
        }
    stored = thread.get("stored_snapshot")
    if stored:
        return {**thread, "snapshot": stored, "snapshot_source": "history"}
    raise HTTPException(status_code=404, detail="未找到该会话状态。")


@app.patch("/api/threads/{thread_id}")
def patch_thread(thread_id: str, req: RenameThreadRequest) -> dict[str, Any]:
    try:
        return rename_thread(DB_PATH, thread_id=thread_id, title=req.title)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="未找到该会话。") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str) -> dict[str, bool]:
    soft_delete_thread(DB_PATH, thread_id=thread_id)
    return {"ok": True}


@app.get("/api/profile")
def get_profile() -> dict[str, Any]:
    """Runtime mode and provider summary for the status bar."""
    profile = runtime_profile()
    return {
        "mode": profile.mode,
        "llm": profile.llm,
        "rag": profile.rag,
        "vision": profile.vision,
        "weather": profile.weather,
    }


@app.get("/")
def index() -> FileResponse:
    return index_response(WEB_DIR / "index.html")


app.mount("/", WebStaticFiles(directory=WEB_DIR), name="web")


def main() -> None:
    """Console-script entrypoint: ``road-distress-web``."""
    import uvicorn

    host = os.environ.get("ROAD_DISTRESS_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("ROAD_DISTRESS_WEB_PORT", "8010"))
    uvicorn.run("road_distress_agent.api.server:app", host=host, port=port, reload=False)


def _api_message(key: str, locale: str) -> str:
    messages = _API_MESSAGES_EN if locale == "en-US" else _API_MESSAGES_ZH
    return messages[key]


_API_MESSAGES_ZH = {
    "image_type": "仅支持图片文件作为补充证据。",
    "image_empty": "上传的图片为空。",
    "image_too_large": "图片过大，请压缩后再上传。",
    "empty_text": "请先输入现场问题或病害描述，图片只能作为补充信息。",
}
_API_MESSAGES_EN = {
    "image_type": "Only image files can be uploaded as supplemental evidence.",
    "image_empty": "The uploaded image is empty.",
    "image_too_large": "The image is too large; compress it and try again.",
    "empty_text": (
        "Enter a site question or distress description first; images are supplemental evidence."
    ),
}


if __name__ == "__main__":
    main()
