"""Static-file cache policy for the WebUI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

NO_STORE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
}


class WebStaticFiles(StaticFiles):
    """Serve WebUI assets while keeping debug trace assets uncached."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and _is_spa_history_path(path, scope):
                return index_response(Path(self.directory) / "index.html")
            raise

    def file_response(
        self,
        full_path: os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        if _is_no_store_path(scope.get("path", "")):
            response.headers.update(NO_STORE_HEADERS)
        return response


def index_response(path: os.PathLike[str]) -> FileResponse:
    return FileResponse(path, headers=NO_STORE_HEADERS)


def _is_no_store_path(path: Any) -> bool:
    if not isinstance(path, str):
        return False
    name = path.rsplit("/", 1)[-1]
    return (
        name == "citations.js"
        or name == "i18n.js"
        or name == "index.html"
        or name.startswith("debug_trace")
    )


def _is_spa_history_path(path: str, scope: Scope) -> bool:
    if scope.get("method") not in {"GET", "HEAD"}:
        return False
    normalized = path.lstrip("/")
    if normalized == "api" or normalized.startswith("api/"):
        return False
    if normalized.startswith("assets/"):
        return False
    return Path(normalized).suffix == ""
