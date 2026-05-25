"""Automated theme-news leg.

For each theme in config.themes, pull recent headlines from Google News RSS
(free, no API key), then score the lot in ONE batched OpenAI call into a
per-theme news sentiment. Results are persisted to the `theme_news` table via
db.save_theme_news.

Two seams keep this testable without network or OpenAI:
  * `http_get(url) -> str` — swap for a fixture in tests / a different source
    (yfinance, a keyed API) in prod without touching scoring.
  * `_get_openai_client()` — monkeypatch to a stub (mirrors weekly_recap).
"""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date

from config.settings import OPENAI_API_KEY
from config.themes import THEMES, Theme
from src.schemas import ThemeNewsBatch

_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
_DEFAULT_LIMIT = 10
_HTTP_TIMEOUT = 15


def _default_http_get(url: str) -> str:
    """Fetch a URL as text. Stdlib only — no requests dependency."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_headlines(query: str, limit: int = _DEFAULT_LIMIT,
                    http_get=_default_http_get) -> list[str]:
    """Recent headline titles for a Google News query, newest first, deduped.

    Any fetch/parse failure returns [] — the news leg degrades gracefully and
    the recommendation falls back to newsletters + price.
    """
    url = _GOOGLE_NEWS_RSS.format(q=urllib.parse.quote(query))
    try:
        xml = http_get(url)
        root = ET.fromstring(xml)
    except Exception:
        return []
    titles: list[str] = []
    seen: set[str] = set()
    for item in root.iterfind(".//item/title"):
        t = (item.text or "").strip()
        if t and t not in seen:
            seen.add(t)
            titles.append(t)
        if len(titles) >= limit:
            break
    return titles


def _get_openai_client():
    """Resolve the OpenAI client. Wrapped so tests can monkeypatch a stub."""
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=OPENAI_API_KEY)


_SCORING_SYSTEM_PROMPT = """You score the tone of recent financial news \
headlines for sub-sector investment themes (e.g. semiconductors, uranium, \
biotech). For EACH theme block supplied, return one score:
- `score`: integer -5..+5. Net market-relevant tone of that theme's headlines. \
-5 strongly negative, 0 neutral/mixed/irrelevant, +5 strongly positive. Judge \
investment impact, not vibes — an earnings beat or supply deficit is positive; \
a guidance cut, fraud, or demand collapse is negative.
- `n_headlines`: how many supplied headlines you actually used.
- `top_headline`: the single most market-relevant headline, copied verbatim \
(or "" if none are relevant).
Only emit themes present in the input. Do not invent headlines."""


def _format_headlines_md(headlines_by_theme: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for key, heads in headlines_by_theme.items():
        label = THEMES[key].label if key in THEMES else key
        lines.append(f"## {key} — {label}")
        if heads:
            lines.extend(f"- {h}" for h in heads)
        else:
            lines.append("- (no headlines)")
        lines.append("")
    return "\n".join(lines)


def score_themes(headlines_by_theme: dict[str, list[str]],
                 llm_client=None) -> dict[str, dict]:
    """Batched LLM scoring. Returns {theme_key: {score, n_headlines, top_headline}}.

    Only themes with at least one headline are sent. Themes the model omits are
    simply absent from the result.
    """
    populated = {k: v for k, v in headlines_by_theme.items() if v}
    if not populated:
        return {}
    client = llm_client or _get_openai_client()
    from config.settings import OPENAI_MODEL
    completion = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": _format_headlines_md(populated)},
        ],
        response_format=ThemeNewsBatch,
        temperature=0.1,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(
            "OpenAI refused or failed to score theme news: "
            f"{completion.choices[0].message.refusal!r}"
        )
    out: dict[str, dict] = {}
    for s in parsed.scores:
        out[s.theme_key] = {
            "score": s.score,
            "n_headlines": s.n_headlines,
            "top_headline": s.top_headline,
        }
    return out


def refresh_theme_news(as_of: date | None = None,
                       themes: list[Theme] | None = None,
                       http_get=_default_http_get,
                       llm_client=None) -> list[dict]:
    """Fetch headlines for every theme, score them, persist, and return the rows.

    Pure orchestration over the two seams; the DB write is the only side effect.
    """
    from src.db import save_theme_news

    as_of = as_of or date.today()
    themes = themes if themes is not None else list(THEMES.values())

    headlines_by_theme = {
        t.key: fetch_headlines(t.news_query, http_get=http_get) for t in themes
    }
    scored = score_themes(headlines_by_theme, llm_client=llm_client)

    rows = [
        {"theme_key": key,
         "score": v["score"],
         "n_headlines": v["n_headlines"],
         "top_headline": v.get("top_headline", "")}
        for key, v in scored.items()
    ]
    save_theme_news(as_of, rows)
    return rows
