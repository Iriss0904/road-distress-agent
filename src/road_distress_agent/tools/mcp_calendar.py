"""Calendar action adapter for the work-order agent.

Dry-run renders a real iCalendar VEVENT (inspectable .ics) and reports
``inserted=False`` — it never writes to a live calendar. A live MCP client drops
in behind the same Protocol once credentials are provided.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from pydantic import BaseModel

from road_distress_agent.error_classifiers import not_implemented_error


class CalendarEvent(BaseModel):
    title: str
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    location: str | None = None
    description: str | None = None


class CalendarClient(Protocol):
    def preview_event(self, event: CalendarEvent) -> dict[str, Any]: ...


class DryRunCalendarClient:
    """Renders an .ics preview and reports it as not inserted."""

    def preview_event(self, event: CalendarEvent) -> dict[str, Any]:
        return {
            "action": "preview_event",
            "provider": "google_calendar",
            "inserted": False,
            "title": event.title,
            "window": f"{event.start_date}~{event.end_date}",
            "ics": _render_ics(event),
        }


def make_calendar_client() -> CalendarClient:
    if _dry_run():
        return DryRunCalendarClient()
    raise not_implemented_error(
        domain="MCP",
        step="CALENDAR",
        responsibility="日历功能未接入",
    )


def _render_ics(event: CalendarEvent) -> str:
    start = event.start_date.replace("-", "")
    end = event.end_date.replace("-", "")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "BEGIN:VEVENT",
        f"SUMMARY:{event.title}",
        f"DTSTART;VALUE=DATE:{start}",
        f"DTEND;VALUE=DATE:{end}",
    ]
    if event.location:
        lines.append(f"LOCATION:{event.location}")
    if event.description:
        lines.append(f"DESCRIPTION:{event.description}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\n".join(lines)


def _dry_run() -> bool:
    return os.environ.get("DELIVERY_DRY_RUN", "true").lower() != "false"
