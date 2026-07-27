"""Email action adapter for the work-order agent.

Dry-run is the user-chosen, documented boundary: it renders a real RFC822 draft
(inspectable .eml) and reports ``sent=False`` — it never sends and never fakes a
send. A live MCP client (e.g. langchain-mcp-adapters → Gmail ``create_draft``)
drops in behind the same Protocol once server credentials are provided.
"""

from __future__ import annotations

import os
from email.message import EmailMessage
from typing import Any, Protocol

from pydantic import BaseModel

from road_distress_agent.error_classifiers import not_implemented_error


class EmailDraft(BaseModel):
    to: str
    subject: str
    body: str


class EmailClient(Protocol):
    def create_draft(self, draft: EmailDraft) -> dict[str, Any]: ...


class DryRunEmailClient:
    """Renders a draft .eml and reports it as an unsent Gmail draft."""

    def create_draft(self, draft: EmailDraft) -> dict[str, Any]:
        message = EmailMessage()
        message["To"] = draft.to
        message["Subject"] = draft.subject
        message["X-Delivery-Mode"] = "dry-run-draft"
        message.set_content(draft.body)
        return {
            "action": "create_draft",
            "provider": "gmail",
            "sent": False,
            "to": draft.to,
            "subject": draft.subject,
            "raw": message.as_string(),
        }


def make_email_client() -> EmailClient:
    if _dry_run():
        return DryRunEmailClient()
    raise not_implemented_error(
        domain="MCP",
        step="EMAIL",
        responsibility="邮件功能未接入",
    )


def _dry_run() -> bool:
    return os.environ.get("DELIVERY_DRY_RUN", "true").lower() != "false"
