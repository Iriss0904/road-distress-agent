"""Capture generated delivery files into the delivery archive index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from road_distress_agent.delivery.artifacts import deliverable_dir
from road_distress_agent.delivery.store import DeliveryStore, make_delivery_store
from road_distress_agent.delivery.store_models import FileFormat
from road_distress_agent.error_classifiers import classify_delivery_error
from road_distress_agent.errors import BoundaryError
from road_distress_agent.projects.store import ProjectStore, make_project_store

SUPPORTED_FORMATS: set[FileFormat] = {"docx", "pdf", "xlsx"}


def capture_finished_package(
    project_id: str,
    values: dict[str, Any],
    *,
    project_store: ProjectStore | None = None,
    delivery_store: DeliveryStore | None = None,
) -> None:
    package_files = _package_files(values)
    if not package_files:
        return
    projects = project_store or make_project_store()
    project = projects.get_project(project_id)
    if project is None:
        raise KeyError(f"unknown project_id: {project_id}")
    records = projects.list_records(project_id, status="active")
    capture_delivery_version(
        project_id=project_id,
        user_id=project.user_id,
        title=project.name,
        record_ids=[record.record_id for record in records],
        package_files=package_files,
        delivery_store=delivery_store,
    )


def capture_delivery_version(
    *,
    project_id: str,
    user_id: str,
    title: str,
    record_ids: list[str],
    package_files: list[str] | None = None,
    delivery_store: DeliveryStore | None = None,
) -> None:
    artifact = _newest_supported_deliverable(project_id, package_files)
    store = delivery_store or make_delivery_store()
    existing = store.list_deliveries(project_id, user_id=user_id)
    delivery = (
        existing[0]
        if existing
        else store.create_delivery(
            project_id=project_id,
            user_id=user_id,
            title=title,
        )
    )
    store.add_version(
        delivery_id=delivery.delivery_id,
        file_path=str(artifact),
        file_format=_file_format(artifact),
        generated_by="auto",
        record_ids=record_ids,
    )
    store.set_status(delivery.delivery_id, "final")


def _package_files(values: dict[str, Any]) -> list[str]:
    package = values.get("delivery_package")
    files = package.get("files") if isinstance(package, dict) else None
    if not isinstance(files, list):
        return []
    return [str(file_path) for file_path in files]


def _newest_supported_deliverable(project_id: str, package_files: list[str] | None) -> Path:
    candidates = _candidate_paths(project_id, package_files)
    supported = [
        path
        for path in candidates
        if path.is_file() and _is_supported(path) and _inside_deliverable_dir(project_id, path)
    ]
    if not supported:
        error = FileNotFoundError("no supported delivery artifact produced")
        raise BoundaryError(classify_delivery_error(error, step="ARCHIVE"))
    return max(supported, key=lambda path: path.stat().st_mtime)


def _candidate_paths(project_id: str, package_files: list[str] | None) -> list[Path]:
    if package_files:
        return [Path(file_path) for file_path in package_files]
    return [path for path in deliverable_dir(project_id).glob("*") if path.is_file()]


def _inside_deliverable_dir(project_id: str, path: Path) -> bool:
    base = deliverable_dir(project_id).resolve()
    target = path.resolve()
    return target == base or base in target.parents


def _is_supported(path: Path) -> bool:
    return path.suffix.lstrip(".").lower() in SUPPORTED_FORMATS


def _file_format(path: Path) -> FileFormat:
    suffix = path.suffix.lstrip(".").lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported delivery artifact format: {suffix!r}")
    return suffix  # type: ignore[return-value]
