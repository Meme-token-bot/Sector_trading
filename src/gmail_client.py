"""Gmail client (Gmail REST API + OAuth 2.0).

Pulls unread mail matching the configured filter address, returns
parsed FetchedMessage objects with body HTML/text + PDF attachments.

Auth is OAuth 2.0: a one-time `scripts/gmail_oauth_setup.py` run writes a
token (with refresh_token) to GMAIL_TOKEN_FILE; at runtime we load and, if
needed, refresh it silently — never an interactive flow. imaplib is no
longer used (the old App Password path was getting flagged by Google).
"""
from __future__ import annotations

import base64
import email
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from email.header import decode_header
from email.utils import parsedate_to_datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from config.settings import (
    GMAIL_ADDRESS, GMAIL_FILTER_ADDRESS, GMAIL_TOKEN_FILE, gmail_configured,
)

log = logging.getLogger(__name__)

# Read + modify labels (to flip UNREAD). Must match the scope the setup
# script authorized, or token refresh will succeed but calls will 403.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


@dataclass
class FetchedMessage:
    gmail_uid: str         # Gmail API message id, stable dedupe key
    message_id: str        # RFC822 Message-ID — preferred dedupe key
    from_addr: str         # display name <email>
    from_email: str        # bare address
    subject: str
    sent_date: date
    body_html: str | None
    body_text: str | None
    pdf_attachments: list[tuple[str, bytes]] = field(default_factory=list)


def _decode_header_value(raw: str | None) -> str:
    if not raw:
        return ""
    parts = decode_header(raw)
    out = []
    for chunk, enc in parts:
        if isinstance(chunk, bytes):
            try:
                out.append(chunk.decode(enc or "utf-8", errors="replace"))
            except LookupError:
                out.append(chunk.decode("utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out).strip()


def _parse_from_addr(raw: str) -> tuple[str, str]:
    decoded = _decode_header_value(raw)
    name, addr = email.utils.parseaddr(decoded)
    return decoded, addr.lower()


def _parse_date(raw: str | None) -> date:
    if not raw:
        return date.today()
    try:
        dt = parsedate_to_datetime(raw)
        return dt.date() if isinstance(dt, datetime) else date.today()
    except Exception:
        return date.today()


def _walk_for_bodies(msg) -> tuple[str | None, str | None, list[tuple[str, bytes]]]:
    body_html: str | None = None
    body_text: str | None = None
    attachments: list[tuple[str, bytes]] = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdisp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in cdisp:
                if ctype == "application/pdf" or (part.get_filename() or "").lower().endswith(".pdf"):
                    fname = _decode_header_value(part.get_filename()) or "attachment.pdf"
                    payload = part.get_payload(decode=True)
                    if payload:
                        attachments.append((fname, payload))
                continue
            if ctype == "text/html" and body_html is None:
                payload = part.get_payload(decode=True)
                if payload:
                    body_html = payload.decode(part.get_content_charset() or "utf-8",
                                               errors="replace")
            elif ctype == "text/plain" and body_text is None:
                payload = part.get_payload(decode=True)
                if payload:
                    body_text = payload.decode(part.get_content_charset() or "utf-8",
                                               errors="replace")
    else:
        ctype = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            decoded = payload.decode(msg.get_content_charset() or "utf-8",
                                     errors="replace")
            if ctype == "text/html":
                body_html = decoded
            else:
                body_text = decoded

    return body_html, body_text, attachments


def _build_search_query() -> str:
    """Gmail search query. is:unread + (optional) to: filter."""
    if GMAIL_FILTER_ADDRESS:
        return f"is:unread to:{GMAIL_FILTER_ADDRESS}"
    return "is:unread"


def _build_service():
    """Load the OAuth token, refresh silently if expired, return a Gmail
    service. Never starts an interactive flow — that's the setup script's job.
    """
    creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Persist the rotated access token so the next run starts valid.
            with open(GMAIL_TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        else:
            raise RuntimeError(
                f"Gmail token at {GMAIL_TOKEN_FILE} is invalid and cannot be "
                "refreshed. Re-run: PYTHONPATH=. python scripts/gmail_oauth_setup.py"
            )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def fetch_unread(mark_seen: bool = False) -> list[FetchedMessage]:
    """Fetch all unread messages matching the filter. Returns parsed messages.

    `mark_seen=True` removes the UNREAD label so re-runs don't re-pull the
    same mail. Defaults to False so you can iterate safely while testing.
    """
    if not gmail_configured():
        raise RuntimeError(
            "Gmail not configured. Set GMAIL_ADDRESS in .env and generate a "
            "token via: PYTHONPATH=. python scripts/gmail_oauth_setup.py"
        )

    svc = _build_service()
    out: list[FetchedMessage] = []
    query = _build_search_query()

    # users.messages.list is paginated; walk every page of matching ids.
    msg_ids: list[str] = []
    page_token = None
    while True:
        resp = svc.users().messages().list(
            userId="me", q=query, pageToken=page_token,
        ).execute()
        msg_ids.extend(m["id"] for m in resp.get("messages", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    for msg_id in msg_ids:
        full = svc.users().messages().get(
            userId="me", id=msg_id, format="raw",
        ).execute()
        raw = base64.urlsafe_b64decode(full["raw"])
        msg = email.message_from_bytes(raw)

        message_id = (msg.get("Message-ID") or f"<no-id-{msg_id}@local>").strip()
        from_disp, from_email = _parse_from_addr(msg.get("From"))
        subject = _decode_header_value(msg.get("Subject"))
        sent = _parse_date(msg.get("Date"))
        body_html, body_text, attachments = _walk_for_bodies(msg)

        out.append(FetchedMessage(
            gmail_uid=msg_id,
            message_id=message_id,
            from_addr=from_disp,
            from_email=from_email,
            subject=subject,
            sent_date=sent,
            body_html=body_html,
            body_text=body_text,
            pdf_attachments=attachments,
        ))

        if mark_seen:
            svc.users().messages().modify(
                userId="me", id=msg_id,
                body={"removeLabelIds": ["UNREAD"]},
            ).execute()

    return out


def test_connection() -> tuple[bool, str]:
    """One-shot connectivity check. Returns (ok, message)."""
    if not gmail_configured():
        return False, f"GMAIL_ADDRESS missing or no token at {GMAIL_TOKEN_FILE}"
    try:
        svc = _build_service()
        profile = svc.users().getProfile(userId="me").execute()
        return True, f"Connected as {profile.get('emailAddress', GMAIL_ADDRESS)}"
    except Exception as e:
        return False, f"Connection error: {e}"
