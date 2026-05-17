"""Gmail IMAP client (App Password auth).

Pulls unread mail matching the configured filter address, returns
parsed FetchedMessage objects with body HTML/text + PDF attachments.
"""
from __future__ import annotations

import email
import imaplib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from email.header import decode_header
from email.utils import parsedate_to_datetime

from config.settings import (
    GMAIL_ADDRESS, GMAIL_APP_PASSWORD, GMAIL_FILTER_ADDRESS,
    GMAIL_IMAP_HOST, GMAIL_IMAP_PORT, gmail_configured,
)

log = logging.getLogger(__name__)


@dataclass
class FetchedMessage:
    gmail_uid: str         # IMAP UID, stable across reads in same folder
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
    """IMAP search query. UNSEEN + (optional) TO filter."""
    if GMAIL_FILTER_ADDRESS:
        return f'(UNSEEN TO "{GMAIL_FILTER_ADDRESS}")'
    return "(UNSEEN)"


def fetch_unread(mark_seen: bool = False) -> list[FetchedMessage]:
    """Fetch all unread messages matching the filter. Returns parsed messages.

    `mark_seen=True` flips the IMAP \\Seen flag so re-runs don't re-pull
    the same mail. Defaults to False so you can iterate safely while testing.
    """
    if not gmail_configured():
        raise RuntimeError(
            "Gmail not configured. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env. "
            "Generate the password at myaccount.google.com/apppasswords (2FA required)."
        )

    out: list[FetchedMessage] = []
    M = imaplib.IMAP4_SSL(GMAIL_IMAP_HOST, GMAIL_IMAP_PORT)
    try:
        M.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        M.select("INBOX", readonly=not mark_seen)
        typ, data = M.uid("SEARCH", None, _build_search_query())
        if typ != "OK" or not data or not data[0]:
            return out
        uids = data[0].split()
        for uid in uids:
            uid_str = uid.decode()
            typ, msg_data = M.uid("FETCH", uid, "(RFC822)")
            if typ != "OK" or not msg_data:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)

            message_id = (msg.get("Message-ID") or f"<no-id-{uid_str}@local>").strip()
            from_disp, from_email = _parse_from_addr(msg.get("From"))
            subject = _decode_header_value(msg.get("Subject"))
            sent = _parse_date(msg.get("Date"))
            body_html, body_text, attachments = _walk_for_bodies(msg)

            out.append(FetchedMessage(
                gmail_uid=uid_str,
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
                M.uid("STORE", uid, "+FLAGS", "(\\Seen)")
    finally:
        try:
            M.logout()
        except Exception:
            pass

    return out


def test_connection() -> tuple[bool, str]:
    """One-shot connectivity check. Returns (ok, message)."""
    if not gmail_configured():
        return False, "GMAIL_ADDRESS or GMAIL_APP_PASSWORD missing in .env"
    try:
        M = imaplib.IMAP4_SSL(GMAIL_IMAP_HOST, GMAIL_IMAP_PORT)
        M.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        typ, data = M.select("INBOX", readonly=True)
        M.logout()
        if typ != "OK":
            return False, f"INBOX select failed: {data!r}"
        count = int(data[0]) if data and data[0] else 0
        return True, f"Connected. INBOX has {count} messages."
    except imaplib.IMAP4.error as e:
        return False, f"IMAP login failed: {e}"
    except Exception as e:
        return False, f"Connection error: {e}"
