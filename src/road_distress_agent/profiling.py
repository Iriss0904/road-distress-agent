"""Helpers for profiling one road-distress graph turn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.table import Table

from road_distress_agent.checkpointing import sqlite_checkpointer
from road_distress_agent.graph import build_graph
from road_distress_agent.settings import (
    apply_runtime_mode,
    missing_runtime_requirements,
    runtime_profile,
)
from road_distress_agent.tracing import extract_trace_events
from road_distress_agent.turns import prepare_user_turn

MS_PER_SECOND = 1000.0
TIMING_DECIMALS = 2
DEFAULT_PROFILE_DB = Path("data/runtime/profile-checkpoints.sqlite3")


@dataclass(frozen=True)
class ProfileConfig:
    text: str
    user_id: str
    checkpoint_db: Path = DEFAULT_PROFILE_DB
    mode: str | None = None


@dataclass(frozen=True)
class TimingRow:
    sequence: int
    node: str
    duration_ms: float


@dataclass(frozen=True)
class TurnProfileResult:
    thread_id: str
    status: str
    mode: str
    provider_summary: str
    total_ms: float
    rows: tuple[TimingRow, ...]
    next_nodes: tuple[str, ...]
    response_preview: str | None
    failure_message: str | None = None


class TurnProfileError(RuntimeError):
    def __init__(self, partial_result: TurnProfileResult, cause: Exception) -> None:
        super().__init__(f"profiled turn failed: {cause.__class__.__name__}: {cause}")
        self.partial_result = partial_result


def run_turn_profile(config: ProfileConfig) -> TurnProfileResult:
    apply_runtime_mode(config.mode)
    missing = missing_runtime_requirements(has_image=False)
    if missing:
        raise ValueError("Live mode is missing required configuration: " + ", ".join(missing))

    thread_id = f"profile-{uuid4().hex[:8]}"
    graph_config = {"configurable": {"thread_id": thread_id}}
    rows: list[TimingRow] = []

    with sqlite_checkpointer(str(config.checkpoint_db)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        stream_input = prepare_user_turn(
            graph=graph,
            config=graph_config,
            existing=graph.get_state(graph_config),
            user_id=config.user_id,
            thread_id=thread_id,
            text=config.text,
            attachments=[],
        )
        started = perf_counter()
        try:
            for chunk in graph.stream(stream_input, config=graph_config, stream_mode="updates"):
                _append_timing_rows(rows, chunk)
        except Exception as exc:
            raise TurnProfileError(
                _partial_result(
                    thread_id=thread_id,
                    started=started,
                    rows=rows,
                    cause=exc,
                ),
                exc,
            ) from exc
        total_ms = round((perf_counter() - started) * MS_PER_SECOND, TIMING_DECIMALS)
        snapshot = graph.get_state(graph_config)

    profile = runtime_profile()
    return TurnProfileResult(
        thread_id=thread_id,
        status="complete",
        mode=profile.mode,
        provider_summary=f"LLM={profile.llm}; RAG={profile.rag}",
        total_ms=total_ms,
        rows=tuple(rows),
        next_nodes=tuple(snapshot.next or ()),
        response_preview=_response_preview(snapshot.values or {}),
    )


def _partial_result(
    *,
    thread_id: str,
    started: float,
    rows: list[TimingRow],
    cause: Exception,
) -> TurnProfileResult:
    profile = runtime_profile()
    elapsed_ms = round((perf_counter() - started) * MS_PER_SECOND, TIMING_DECIMALS)
    return TurnProfileResult(
        thread_id=thread_id,
        status="failed",
        mode=profile.mode,
        provider_summary=f"LLM={profile.llm}; RAG={profile.rag}",
        total_ms=elapsed_ms,
        rows=tuple(rows),
        next_nodes=(),
        response_preview=None,
        failure_message=f"{cause.__class__.__name__}: {cause}",
    )


def print_profile_result(result: TurnProfileResult, console: Console | None = None) -> None:
    active_console = console or Console()
    table = Table(title="节点耗时")
    table.add_column("#", justify="right")
    table.add_column("节点")
    table.add_column("耗时 ms", justify="right")
    for row in result.rows:
        table.add_row(str(row.sequence), row.node, f"{row.duration_ms:.2f}")

    active_console.print(f"[bold]Thread:[/bold] {result.thread_id}")
    active_console.print(f"[bold]Status:[/bold] {result.status}")
    active_console.print(f"[bold]Mode:[/bold] {result.mode} ({result.provider_summary})")
    active_console.print(f"[bold]Total:[/bold] {result.total_ms:.2f} ms")
    if result.status == "complete":
        active_console.print(f"[bold]Next:[/bold] {result.next_nodes or 'END'}")
    if result.failure_message:
        active_console.print(f"[bold red]Failure:[/bold red] {result.failure_message}")
    active_console.print(table)
    if result.response_preview:
        active_console.print("[bold]Response preview:[/bold]")
        active_console.print(result.response_preview)


def _append_timing_rows(rows: list[TimingRow], chunk: dict[str, Any]) -> None:
    for node_delta in chunk.values():
        if not isinstance(node_delta, dict):
            continue
        for trace in extract_trace_events(node_delta.get("audit_log") or []):
            if trace.get("kind") != "node_timing":
                continue
            rows.append(_timing_row(len(rows) + 1, trace))


def _timing_row(sequence: int, trace: dict[str, Any]) -> TimingRow:
    duration = trace.get("metadata", {}).get("duration_ms")
    if not isinstance(duration, int | float):
        raise ValueError(f"node_timing trace missing duration_ms: {trace}")
    return TimingRow(
        sequence=sequence,
        node=str(trace.get("node") or "unknown"),
        duration_ms=float(duration),
    )


def _response_preview(values: dict[str, Any]) -> str | None:
    final_message = values.get("final_answer_message")
    if final_message:
        return str(final_message)
    final_answer = values.get("final_answer")
    summary = getattr(final_answer, "summary", None)
    if summary:
        return str(summary)
    direct_message = values.get("direct_message")
    return str(direct_message) if direct_message else None
