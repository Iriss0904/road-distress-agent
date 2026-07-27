"""Thread-safe, bounded future registry for B-09."""

from __future__ import annotations

import atexit
import logging
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import BoundedSemaphore, RLock
from time import monotonic

from road_distress_agent.state import AgentState

MAX_PREFETCH_WORKERS = 4
MAX_PREFETCH_ENTRIES = 64
COMPLETED_ENTRY_TTL_SECONDS = 300.0
_LOGGER = logging.getLogger(__name__)


class SpeculativePrefetchCapacityError(RuntimeError):
    """Raised when the explicit bounded prefetch capacity is exhausted."""


@dataclass(frozen=True)
class PrefetchRequest:
    thread_id: str
    fingerprint: str
    top_method: str


@dataclass(frozen=True)
class PrefetchEntry:
    thread_id: str
    fingerprint: str
    top_method: str
    future: Future[AgentState]
    started_at: float
    completed_at: float | None = None


class SpeculativePrefetchRegistry:
    """Keep futures out of checkpoints and retain active work until completion."""

    def __init__(
        self,
        executor: Executor,
        *,
        clock: Callable[[], float] = monotonic,
        capacity: int = MAX_PREFETCH_ENTRIES,
        completed_ttl: float = COMPLETED_ENTRY_TTL_SECONDS,
    ) -> None:
        self._executor = executor
        self._clock = clock
        self._completed_ttl = completed_ttl
        self._capacity_limit = capacity
        self._capacity = BoundedSemaphore(capacity)
        self._entries: dict[str, PrefetchEntry] = {}
        self._lock = RLock()

    def start(
        self,
        request: PrefetchRequest,
        work: Callable[[], AgentState],
    ) -> PrefetchEntry:
        with self._lock:
            self._cleanup_locked()
            existing = self._entries.get(request.thread_id)
            if existing and _same_request(existing, request):
                return existing
            if existing:
                self._discard_locked(existing)
            self._acquire_capacity()
            try:
                future = self._executor.submit(work)
            except BaseException:
                self._capacity.release()
                raise
            entry = _entry(request, future, self._clock())
            self._entries[request.thread_id] = entry
            future.add_done_callback(lambda done: self._complete(request.thread_id, done))
            return entry

    def pop(self, thread_id: str) -> PrefetchEntry | None:
        with self._lock:
            self._cleanup_locked()
            return self._entries.pop(thread_id, None)

    def discard(self, thread_id: str) -> PrefetchEntry | None:
        with self._lock:
            self._cleanup_locked()
            entry = self._entries.get(thread_id)
            if entry:
                self._discard_locked(entry)
            return entry

    def close(self) -> None:
        shutdown = getattr(self._executor, "shutdown", None)
        if shutdown:
            shutdown(wait=True, cancel_futures=True)

    def _acquire_capacity(self) -> None:
        if self._capacity.acquire(blocking=False):
            return
        raise SpeculativePrefetchCapacityError(
            f"B-09 prefetch capacity exhausted ({self._capacity_limit} entries)."
        )

    def _discard_locked(self, entry: PrefetchEntry) -> None:
        self._entries.pop(entry.thread_id, None)
        entry.future.cancel()

    def _complete(self, thread_id: str, future: Future[AgentState]) -> None:
        self._capacity.release()
        with self._lock:
            entry = self._entries.get(thread_id)
            if entry and entry.future is future:
                self._entries[thread_id] = replace(entry, completed_at=self._clock())
        exception = future.exception() if not future.cancelled() else None
        if exception:
            error_info = (type(exception), exception, exception.__traceback__)
            _LOGGER.error("B-09 background prefetch failed", exc_info=error_info)

    def _cleanup_locked(self) -> None:
        cutoff = self._clock() - self._completed_ttl
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.completed_at is not None and entry.completed_at <= cutoff
        ]
        for key in expired:
            self._entries.pop(key, None)


def _entry(request: PrefetchRequest, future: Future[AgentState], now: float) -> PrefetchEntry:
    return PrefetchEntry(
        thread_id=request.thread_id,
        fingerprint=request.fingerprint,
        top_method=request.top_method,
        future=future,
        started_at=now,
    )


def _same_request(entry: PrefetchEntry, request: PrefetchRequest) -> bool:
    return entry.fingerprint == request.fingerprint and entry.top_method == request.top_method


_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_PREFETCH_WORKERS,
    thread_name_prefix="b09-detail-prefetch",
)
DEFAULT_REGISTRY = SpeculativePrefetchRegistry(_EXECUTOR)
atexit.register(DEFAULT_REGISTRY.close)
