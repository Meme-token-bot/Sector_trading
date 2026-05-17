"""Module A: Unstructured newsletter -> NewsletterAnalysis -> SQLite."""
from __future__ import annotations

from datetime import date

from openai import OpenAI

from config.settings import OPENAI_API_KEY, OPENAI_MODEL
from src.db import save_analysis
from src.schemas import NewsletterAnalysis

_SYSTEM_PROMPT = """You are a financial analyst extracting structured views \
from a macroeconomic newsletter for a US sector-rotation model.

Rules:
1. `author` should be the byline or publication name as written.
2. `publication_date` must be the date the piece was published. If not stated, \
infer from the text or use today.
3. `overall_macro_bias`:
   - "Defensive" if the author favors safety (utilities, staples, healthcare, cash, gold, bonds).
   - "Expansionary" if the author favors risk-on / cyclical / growth.
   - "Neutral" if mixed or no clear regime call.
4. `sector_ratings`: ONLY include the 11 US SPDR Select Sector ETFs the author \
explicitly discusses. Do not list a sector with score 0 just because it wasn't \
mentioned — omit it entirely. Map general themes to the right ETF, e.g. \
"semiconductors" → XLK, "banks" → XLF, "oil majors" → XLE, "REITs" → XLRE, \
"media/telecom" → XLC, "homebuilders/autos/retail" → XLY, "gold miners" → XLB, \
"copper miners" → XLB, "biotech" → XLV.
5. `sentiment_score`: integer -5..+5. Use -5/+5 only for unambiguous strong calls.
6. `reasoning`: quote or paraphrase the author's actual stated reason. Do not invent.
7. `summary`: 2-3 sentences, neutral, capturing the macro thesis.

If the input is not a financial/macro piece, return an empty `sector_ratings` \
list and a brief `summary` noting that."""


_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def parse_newsletter(raw_text: str,
                     author_hint: str | None = None,
                     date_hint: date | None = None) -> NewsletterAnalysis:
    client = _get_client()

    user_content = raw_text
    if author_hint or date_hint:
        hints = []
        if author_hint:
            hints.append(f"Author hint: {author_hint}")
        if date_hint:
            hints.append(f"Publication date hint: {date_hint.isoformat()}")
        user_content = "\n".join(hints) + "\n\n---\n\n" + raw_text

    completion = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=NewsletterAnalysis,
        temperature=0.1,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(
            f"OpenAI refused or failed to parse: "
            f"{completion.choices[0].message.refusal!r}"
        )
    return parsed


def ingest(raw_text: str,
           author_hint: str | None = None,
           date_hint: date | None = None) -> tuple[NewsletterAnalysis, int | None]:
    analysis = parse_newsletter(raw_text, author_hint, date_hint)
    nid = save_analysis(analysis, raw_text)
    return analysis, nid


def fetch_and_ingest(mark_seen: bool = True,
                     follow_links: bool = True) -> list[dict]:
    """End-to-end: pull unread Gmail, enrich with whitelisted links/PDFs,
    parse with the LLM, persist. Returns a per-message report.
    """
    from src.content_extractor import build_context
    from src.db import attach_gmail_message_id, gmail_message_already_ingested
    from src.gmail_client import fetch_unread

    messages = fetch_unread(mark_seen=mark_seen)
    report: list[dict] = []
    for m in messages:
        entry: dict = {
            "subject": m.subject,
            "from": m.from_addr,
            "date": m.sent_date.isoformat(),
            "message_id": m.message_id,
            "status": "pending",
        }
        if gmail_message_already_ingested(m.message_id):
            entry["status"] = "skipped_dupe_message_id"
            report.append(entry)
            continue

        try:
            ctx = build_context(
                body_html=m.body_html,
                body_text=m.body_text,
                pdf_attachments=m.pdf_attachments,
                follow_links=follow_links,
            )
            assembled = ctx.assemble()
            entry["chars"] = len(assembled)
            entry["links_used"] = len(ctx.enriched_links)
            entry["pdfs_used"] = len(ctx.enriched_pdfs)
            entry["truncated"] = ctx.truncated

            analysis, nid = ingest(
                raw_text=assembled,
                author_hint=m.from_addr or m.from_email or None,
                date_hint=m.sent_date,
            )
            if nid is None:
                entry["status"] = "skipped_dupe_content_hash"
            else:
                attach_gmail_message_id(nid, m.message_id)
                entry["status"] = "ingested"
                entry["newsletter_id"] = nid
                entry["sectors"] = [r.ticker for r in analysis.sector_ratings]
                entry["bias"] = analysis.overall_macro_bias.value
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
        report.append(entry)
    return report
