"""HTML stripping, link extraction, PDF extraction, whitelist fetching.

Assembles a "context bundle" from one email: the body text plus extracted
text from any whitelisted links and PDF attachments. Capped at
EXTRACTION.max_total_chars to keep gpt-4o-mini token cost predictable.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from config.settings import EXTRACTION
from config.whitelist import is_tracker, is_whitelisted

log = logging.getLogger(__name__)

_USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) sector-rotation/1.0 "
               "(personal research tool)")
_MAX_FETCH_BYTES = 5_000_000  # 5 MB cap per URL
_MAX_TRACKER_RESOLUTIONS = 15  # bound HEAD calls per email


@dataclass
class ExtractedContent:
    body_text: str
    enriched_links: list[tuple[str, str]] = field(default_factory=list)  # (url, text)
    enriched_pdfs: list[tuple[str, str]] = field(default_factory=list)   # (label, text)
    truncated: bool = False

    def assemble(self, max_chars: int | None = None) -> str:
        cap = max_chars or EXTRACTION.max_total_chars
        parts: list[str] = [self.body_text.strip()]
        for url, text in self.enriched_links:
            parts.append(f"\n\n--- LINKED ARTICLE: {url} ---\n{text.strip()}")
        for label, text in self.enriched_pdfs:
            parts.append(f"\n\n--- PDF: {label} ---\n{text.strip()}")
        combined = "\n".join(parts)
        if len(combined) > cap:
            self.truncated = True
            combined = combined[:cap] + "\n\n[... truncated ...]"
        return combined


def html_to_text(html: str) -> str:
    """Strip HTML to readable plain text. Drops nav/footer/scripts."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_links(html: str) -> list[str]:
    """Extract unique http(s) hrefs from email HTML, in document order."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.lower().startswith(("http://", "https://")):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append(href)
    return out


def _resolve_redirects(url: str) -> str | None:
    """Follow redirects on a tracker/wrapper URL and return the final URL.

    Uses streaming GET with an immediate close so we pay for headers only,
    not the body. HEAD is unreliable — some trackers 405 it or refuse to
    redirect on HEAD.
    """
    try:
        with requests.get(
            url,
            headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
            timeout=EXTRACTION.fetch_timeout_seconds,
            allow_redirects=True,
            stream=True,
        ) as resp:
            return resp.url
    except Exception as e:
        log.info("redirect resolve failed %s: %s", url, e)
        return None


def _fetch_url(url: str) -> tuple[bytes, str] | None:
    """GET url, return (body_bytes, content_type) or None on error."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
            timeout=EXTRACTION.fetch_timeout_seconds,
            allow_redirects=True,
            stream=True,
        )
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        body = b""
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            body += chunk
            if len(body) > _MAX_FETCH_BYTES:
                break
        return body, ctype
    except Exception as e:
        log.info("fetch failed %s: %s", url, e)
        return None


def _extract_pdf_text(blob: bytes, label: str) -> str | None:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(blob))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(p.strip() for p in pages if p.strip())
        return text or None
    except Exception as e:
        log.info("pdf parse failed for %s: %s", label, e)
        return None


def _extract_html_text(blob: bytes, url: str) -> str | None:
    try:
        import trafilatura
        text = trafilatura.extract(
            blob.decode("utf-8", errors="replace"),
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        return text or None
    except Exception as e:
        log.info("trafilatura failed for %s: %s", url, e)
        return None


def fetch_link(url: str) -> tuple[str, str] | None:
    """Fetch a whitelisted URL, return (url, extracted_text) or None."""
    result = _fetch_url(url)
    if not result:
        return None
    body, ctype = result
    if "pdf" in ctype:
        text = _extract_pdf_text(body, url)
    else:
        text = _extract_html_text(body, url)
    if not text:
        return None
    return url, text


def extract_pdf_attachment(blob: bytes, filename: str) -> tuple[str, str] | None:
    text = _extract_pdf_text(blob, filename)
    if not text:
        return None
    return filename, text


def build_context(
    body_html: str | None,
    body_text: str | None,
    pdf_attachments: list[tuple[str, bytes]] | None = None,
    follow_links: bool = True,
) -> ExtractedContent:
    """Assemble the full LLM-ready context for one email.

    pdf_attachments: list of (filename, raw_bytes) extracted from the email.
    """
    if body_html:
        text_body = html_to_text(body_html)
        link_candidates = extract_links(body_html)
    else:
        text_body = body_text or ""
        link_candidates = re.findall(r"https?://\S+", text_body)

    out = ExtractedContent(body_text=text_body)

    for filename, blob in pdf_attachments or []:
        result = extract_pdf_attachment(blob, filename)
        if result:
            out.enriched_pdfs.append(result)

    if follow_links:
        # Walk candidate links in order. A URL is kept if its host is
        # already whitelisted, OR if it looks like a tracker/wrapper and
        # resolving its redirect chain lands on a whitelisted host.
        seen: set[str] = set()
        deduped: list[str] = []
        resolutions = 0
        for raw in link_candidates:
            final = raw
            if not is_whitelisted(final):
                if not is_tracker(final):
                    continue
                if resolutions >= _MAX_TRACKER_RESOLUTIONS:
                    continue
                resolutions += 1
                resolved = _resolve_redirects(final)
                if not resolved or not is_whitelisted(resolved):
                    continue
                final = resolved
            if final in seen:
                continue
            seen.add(final)
            deduped.append(final)
            if len(deduped) >= EXTRACTION.max_links_per_newsletter:
                break

        for url in deduped:
            result = fetch_link(url)
            if result:
                if result[0].lower().endswith(".pdf"):
                    out.enriched_pdfs.append((url, result[1]))
                else:
                    out.enriched_links.append(result)

    return out
