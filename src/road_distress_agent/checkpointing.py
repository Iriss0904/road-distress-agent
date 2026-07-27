"""Checkpoint helpers for local LangGraph runs."""

from __future__ import annotations

import sqlite3
import warnings
from collections.abc import Iterator
from contextlib import closing, contextmanager

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r"The default value of `allowed_objects` will change in a future version\..*",
)

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

ALLOWED_MSGPACK_MODULES: tuple[tuple[str, str], ...] = (
    ("road_distress_agent.errors", "ErrorCategory"),
    ("road_distress_agent.enums", "WorkflowPhase"),
    ("road_distress_agent.state", "AddressContext"),
    ("road_distress_agent.state", "AttachmentRef"),
    ("road_distress_agent.state", "AuditEvent"),
    ("road_distress_agent.state", "CandidateSelection"),
    ("road_distress_agent.state", "Citation"),
    ("road_distress_agent.state", "CriticIssue"),
    ("road_distress_agent.state", "DiseaseCandidate"),
    ("road_distress_agent.state", "DiseaseDiscriminatorOutput"),
    ("road_distress_agent.state", "DiscriminatorOutput"),
    ("road_distress_agent.state", "DistressInput"),
    ("road_distress_agent.state", "ErrorEvent"),
    ("road_distress_agent.state", "EvidenceBundle"),
    ("road_distress_agent.state", "FieldChange"),
    ("road_distress_agent.state", "FinalAnswer"),
    ("road_distress_agent.state", "FinalAnswerDisplay"),
    ("road_distress_agent.state", "InterruptState"),
    ("road_distress_agent.state", "KbHop"),
    ("road_distress_agent.state", "KbHopResult"),
    ("road_distress_agent.state", "KbQueryPlan"),
    ("road_distress_agent.state", "KbSubQuestion"),
    ("road_distress_agent.state", "LoadedMemory"),
    ("road_distress_agent.state", "MethodDiscriminatorOutput"),
    ("road_distress_agent.state", "MethodOption"),
    ("road_distress_agent.state", "QueryPlan"),
    ("road_distress_agent.state", "RankedMethod"),
    ("road_distress_agent.state", "ReferenceItem"),
    ("road_distress_agent.state", "RetrievedChunk"),
    ("road_distress_agent.state", "RetrievalAttempt"),
    ("road_distress_agent.state", "SafetyReview"),
    ("road_distress_agent.state", "SceneContext"),
    ("road_distress_agent.state", "SolutionCandidate"),
    ("road_distress_agent.state", "WeatherAdvice"),
    ("road_distress_agent.state", "WeatherContext"),
    ("road_distress_agent.state", "WeatherForecastDay"),
    ("road_distress_agent.state", "WeatherForecastHour"),
)


@contextmanager
def sqlite_checkpointer(conn_string: str) -> Iterator[SqliteSaver]:
    """Open a SQLite checkpointer with explicit project type allowlisting."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
    with closing(sqlite3.connect(conn_string, check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=serde)
