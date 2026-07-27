"""Thread-safe lazy initialization for heavyweight process-local tools."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Generic, TypeVar, cast

Value = TypeVar("Value")
_UNINITIALIZED = object()


class LockedLazy(Generic[Value]):
    """Initialize one value under lock while keeping warm reads lock-free."""

    def __init__(self) -> None:
        self._value: Value | object = _UNINITIALIZED
        self._lock = Lock()

    def get(self, factory: Callable[[], Value]) -> Value:
        current = self._value
        if current is not _UNINITIALIZED:
            return cast(Value, current)
        with self._lock:
            current = self._value
            if current is _UNINITIALIZED:
                current = factory()
                self._value = current
        return cast(Value, current)
