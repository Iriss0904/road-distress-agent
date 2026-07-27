"""Typer CLI entrypoint."""

from __future__ import annotations

import hashlib
from mimetypes import guess_type
from pathlib import Path
from uuid import uuid4

import typer
from dotenv import load_dotenv
from rich import print

from road_distress_agent.checkpointing import sqlite_checkpointer
from road_distress_agent.factories import make_initial_state
from road_distress_agent.graph import build_graph
from road_distress_agent.profiling import (
    ProfileConfig,
    TurnProfileError,
    print_profile_result,
    run_turn_profile,
)
from road_distress_agent.settings import (
    apply_runtime_mode,
    missing_runtime_requirements,
    runtime_profile,
)
from road_distress_agent.state import AttachmentRef, FinalAnswer
from road_distress_agent.turns import prepare_user_turn

load_dotenv()

app = typer.Typer(help="Road distress LangGraph agent CLI.")
DEFAULT_USER_ID = "virtual_001"


def _attachment_from_image_path(image: Path) -> AttachmentRef:
    path = image.expanduser()
    if not path.exists():
        raise typer.BadParameter(f"Image path does not exist: {image}")
    if not path.is_file():
        raise typer.BadParameter(f"Image path is not a file: {image}")

    media_type = guess_type(path.name)[0]
    if media_type is None or not media_type.startswith("image/"):
        raise typer.BadParameter(f"Unsupported image file type: {image}")

    data = path.read_bytes()
    return AttachmentRef(
        uri=str(path.resolve()),
        media_type=media_type,
        sha256=hashlib.sha256(data).hexdigest(),
    )


@app.callback()
def main() -> None:
    """Road distress LangGraph agent CLI."""


@app.command()
def init_state(text: str, user_id: str = DEFAULT_USER_ID) -> None:
    """Print a minimal initial state for debugging scaffold wiring."""
    state = make_initial_state(user_id=user_id, latest_user_text=text)
    typer.echo(state)


@app.command()
def profile(
    text: str = typer.Argument(..., help="User text for one profiled graph turn."),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Runtime mode: live for real APIs, dev for local debugging.",
    ),
) -> None:
    """Run one turn and print node-level runtime timings."""
    try:
        result = run_turn_profile(ProfileConfig(text=text, user_id=DEFAULT_USER_ID, mode=mode))
    except TurnProfileError as exc:
        print_profile_result(exc.partial_result)
        raise
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_profile_result(result)


@app.command()
def run(
    text: str = typer.Argument(..., help="User text for this road-distress turn."),
    thread_id: str | None = typer.Option(
        None,
        "--thread-id",
        "-t",
        help="Existing distress-scenario thread to resume. Omit for a new distress.",
    ),
    image: Path | None = typer.Option(
        None,
        "--image",
        "-i",
        help="Optional single image path for the current turn.",
    ),
    user_id: str = typer.Option(
        DEFAULT_USER_ID,
        "--user-id",
        help="Stable user identity for long-term memory and analytics.",
    ),
    checkpoint_db: Path = typer.Option(
        Path(".checkpoints.db"),
        "--checkpoint-db",
        help="SQLite checkpoint database path.",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Runtime mode: live for real APIs, dev for offline mocks/fallbacks. Defaults to live.",
    ),
) -> None:
    """Run or resume one graph turn from the command line."""
    if not text.strip():
        raise typer.BadParameter("Text is required; pure image input is not supported.")

    try:
        apply_runtime_mode(mode)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    active_thread_id = thread_id or f"thread-{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": active_thread_id}}
    attachments = [_attachment_from_image_path(image)] if image else []
    missing = missing_runtime_requirements(has_image=bool(attachments))
    if missing:
        raise typer.BadParameter(
            "Live mode is missing required configuration: " + ", ".join(missing)
        )
    profile = runtime_profile()

    with sqlite_checkpointer(str(checkpoint_db)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        existing = graph.get_state(config)

        stream_input = prepare_user_turn(
            graph=graph,
            config=config,
            existing=existing,
            user_id=user_id,
            thread_id=active_thread_id,
            text=text,
            attachments=attachments,
        )

        for chunk in graph.stream(stream_input, config=config, stream_mode="updates"):
            print(chunk)

        final = graph.get_state(config)
        values = final.values
        print(f"\n[bold]Runtime mode:[/bold] {profile.mode}")
        print(
            "[bold]Providers:[/bold] "
            f"LLM={profile.llm}; RAG={profile.rag}; "
            f"Vision={profile.vision}; Weather={profile.weather}"
        )
        print(f"\n[bold]User ID:[/bold] {values.get('user_id', user_id)}")
        print(f"[bold]Thread ID:[/bold] {active_thread_id}")
        print(f"[bold]Awaiting input?[/bold] {values.get('awaiting_user_input', False)}")
        if values.get("scene_description"):
            print(f"[bold]Vision scene:[/bold] {values['scene_description']}")
        if final.next:
            print(f"[bold]Next node(s):[/bold] {final.next}")
        if values.get("interrupt"):
            interrupt = values["interrupt"]
            print(f"[yellow]Interrupt:[/yellow] {interrupt.prompt}")
        if values.get("solution_candidates"):
            print("[bold]Candidates:[/bold]")
            for candidate in values["solution_candidates"]:
                print(
                    f"  {candidate.candidate_id}. {candidate.name} "
                    f"({candidate.rough_confidence:.2f})"
                )
        final_answer = values.get("final_answer")
        if isinstance(final_answer, FinalAnswer):
            print(f"[bold]Final answer:[/bold] {final_answer.summary}")
            if final_answer.diagnosis:
                print(f"[bold]Diagnosis:[/bold] {final_answer.diagnosis}")
            if final_answer.method_selection:
                print(f"[bold]Method:[/bold] {final_answer.method_selection}")
            if final_answer.steps:
                print("[bold]Construction steps:[/bold]")
                for index, step in enumerate(final_answer.steps, start=1):
                    print(f"  {index}. {step}")
            if final_answer.materials:
                print("[bold]Materials:[/bold]")
                for item in final_answer.materials:
                    print(f"  - {item}")
            if final_answer.acceptance_criteria:
                print("[bold]Acceptance criteria:[/bold]")
                for item in final_answer.acceptance_criteria:
                    print(f"  - {item}")
            if final_answer.safety_notes:
                print("[bold]Safety tips:[/bold]")
                for item in final_answer.safety_notes:
                    print(f"  - {item}")
            if final_answer.weather_advice and final_answer.weather_advice.summary:
                print(
                    f"[bold]{final_answer.weather_advice.title}:[/bold] "
                    f"{final_answer.weather_advice.summary}"
                )
            if final_answer.evidence_gaps:
                print("[bold]Evidence gaps:[/bold]")
                for item in final_answer.evidence_gaps:
                    print(f"  - {item}")
            print(f"[bold]Need human review?[/bold] {final_answer.need_human_review}")
            print(f"[bold]Citations:[/bold] {len(final_answer.citations)}")


if __name__ == "__main__":
    app()
