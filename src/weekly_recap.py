"""Weekly recap synthesis — newsletters + live macro -> plain-language brief.

Pure-ish: gather_context() reads SQLite + yfinance/FRED for inputs, but does
no IO beyond what those getters already cache. generate_recap() makes one
OpenAI call. Both are unit-testable: gather_context against a seeded DB,
generate_recap by monkey-patching _get_openai_client to return a stub.

Band tables are duplicated from app.py (the macro tab's `_VIX_BANDS` etc.).
This is deliberate — embedding them keeps this module self-contained and
avoids touching app.py during the recap feature. If the band thresholds
move, update both files.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from config.settings import OPENAI_API_KEY, SECTOR_ETFS
from src.db import _conn, init_db
from src.market_engine import (
    copper_gold_ratio, dxy_level, fetch_fred_indicators,
    fetch_macro_prices, gold_oil_ratio, vix_level, yield_curve_spread,
)
from src.schemas import (
    MacroSnapshot, NewsletterExcerpt, SectorRollup,
    WeeklyRecap, WeeklyRecapContext,
)


# ---------------------------------------------------------------------------
# Regime bands — duplicated from app.py macro tab. Keep in sync.
# Each entry is (label, emoji, range_label, predicate).
# ---------------------------------------------------------------------------

_VIX_BANDS = [
    ("Complacent", "🟢", "< 12",  lambda v: v < 12),
    ("Normal",     "🟡", "12–20", lambda v: 12 <= v < 20),
    ("Stressed",   "🟠", "20–30", lambda v: 20 <= v < 30),
    ("Crisis",     "🔴", "≥ 30",  lambda v: v >= 30),
]
_HY_OAS_BANDS = [
    ("Tight (risk-on)", "🟢", "< 3.5%",   lambda v: v < 3.5),
    ("Normal",          "🟡", "3.5–5.0%", lambda v: 3.5 <= v < 5.0),
    ("Stress building", "🟠", "5.0–7.0%", lambda v: 5.0 <= v < 7.0),
    ("Credit crisis",   "🔴", "≥ 7.0%",   lambda v: v >= 7.0),
]
_GOLD_OIL_BANDS = [
    ("Oil-rich / pro-cyclical", "🟢", "< 15",  lambda v: v < 15),
    ("Normal",                  "🟡", "15–25", lambda v: 15 <= v < 25),
    ("Risk-off bid",            "🟠", "25–35", lambda v: 25 <= v < 35),
    ("Recession-likely",        "🔴", "≥ 35",  lambda v: v >= 35),
]
_COPPER_GOLD_Z_BANDS = [
    ("Deflationary",     "🔴", "z < -1",   lambda v: v < -1),
    ("Softening",        "🟠", "-1 to 0",  lambda v: -1 <= v < 0),
    ("Reflation",        "🟢", "0 to +1",  lambda v: 0 <= v < 1),
    ("Strong reflation", "🟢", "z ≥ +1",   lambda v: v >= 1),
]
_DXY_BANDS = [
    ("Weak dollar",   "🟢", "< 95",    lambda v: v < 95),
    ("Normal-weak",   "🟡", "95–100",  lambda v: 95 <= v < 100),
    ("Normal-strong", "🟠", "100–105", lambda v: 100 <= v < 105),
    ("Strong dollar", "🔴", "≥ 105",   lambda v: v >= 105),
]
_T10Y2Y_BANDS = [
    ("Inverted (recession warning)", "🔴", "< 0",       lambda v: v < 0),
    ("Flat",                          "🟠", "0–0.5%",    lambda v: 0 <= v < 0.5),
    ("Normal",                        "🟡", "0.5–1.5%",  lambda v: 0.5 <= v < 1.5),
    ("Steep",                         "🟢", "≥ 1.5%",    lambda v: v >= 1.5),
]
_UST10_BANDS = [
    ("Easy",        "🟢", "< 3%", lambda v: v < 3),
    ("Normal",      "🟡", "3–4%", lambda v: 3 <= v < 4),
    ("Restrictive", "🟠", "4–5%", lambda v: 4 <= v < 5),
    ("Tight",       "🔴", "≥ 5%", lambda v: v >= 5),
]
_REAL_10Y_BANDS = [
    ("Financial repression", "🟢", "< 0",  lambda v: v < 0),
    ("Normal",               "🟡", "0–1%", lambda v: 0 <= v < 1),
    ("Restrictive",          "🟠", "1–2%", lambda v: 1 <= v < 2),
    ("Tight",                "🔴", "≥ 2%", lambda v: v >= 2),
]
_BREAKEVEN_BANDS = [
    ("Deflationary fears", "🔴", "< 1.8%",   lambda v: v < 1.8),
    ("Anchored",           "🟢", "1.8–2.5%", lambda v: 1.8 <= v < 2.5),
    ("Unanchored",         "🟠", "≥ 2.5%",   lambda v: v >= 2.5),
]


def _band_label(value, bands) -> tuple[str, str]:
    """Return (label, emoji) for the first matching band, or ('—', '⚪')."""
    if value is None or (isinstance(value, float) and value != value):
        return "—", "⚪"
    for entry in bands:
        if entry[3](value):
            return entry[0], entry[1]
    return "—", "⚪"


def _macro_snapshot(name: str, level, z_or_slope, kind: str, bands,
                    band_input) -> MacroSnapshot:
    """Build a MacroSnapshot, picking the band off `band_input` (level or z)."""
    label, emoji = _band_label(band_input, bands)
    return MacroSnapshot(
        name=name,
        level=float(level) if level is not None and level == level else None,
        z_or_slope=(float(z_or_slope) if z_or_slope is not None
                    and z_or_slope == z_or_slope else None),
        z_or_slope_kind=kind,
        band_label=label,
        band_emoji=emoji,
    )


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _query_newsletters_in_window(as_of: date,
                                 lookback_days: int) -> pd.DataFrame:
    """Pull newsletters + their sector_ratings for the lookback window."""
    cutoff = as_of - timedelta(days=lookback_days)
    init_db()
    with _conn() as c:
        df = pd.read_sql_query(
            """
            SELECT n.id              AS newsletter_id,
                   n.author          AS author,
                   n.publication_date AS publication_date,
                   n.overall_macro_bias AS overall_macro_bias,
                   n.summary         AS summary,
                   sr.ticker         AS ticker,
                   sr.sentiment_score AS sentiment_score,
                   sr.reasoning      AS reasoning
            FROM newsletters n
            LEFT JOIN sector_ratings sr ON sr.newsletter_id = n.id
            WHERE n.publication_date >= ?
              AND n.publication_date <= ?
            ORDER BY n.publication_date DESC, n.id DESC
            """,
            c, params=(cutoff.isoformat(), as_of.isoformat()),
        )
    return df


def _build_newsletter_excerpts(df: pd.DataFrame) -> list[NewsletterExcerpt]:
    """Roll the long-form join back up into one row per newsletter."""
    if df.empty:
        return []
    out: list[NewsletterExcerpt] = []
    for nid, group in df.groupby("newsletter_id", sort=False):
        head = group.iloc[0]
        ratings: list[dict] = []
        for _, r in group.iterrows():
            if pd.isna(r["ticker"]):
                continue
            ratings.append({
                "ticker": r["ticker"],
                "sentiment_score": int(r["sentiment_score"]),
                "reasoning": (r["reasoning"] or "").strip(),
            })
        pd_str = head["publication_date"]
        pd_val = (date.fromisoformat(pd_str) if isinstance(pd_str, str)
                  else pd_str)
        out.append(NewsletterExcerpt(
            author=head["author"],
            publication_date=pd_val,
            overall_macro_bias=head["overall_macro_bias"] or "Neutral",
            summary=(head["summary"] or "").strip(),
            sector_ratings=ratings,
        ))
    return out


def _build_sector_rollups(df: pd.DataFrame) -> list[SectorRollup]:
    """Per-sector mean sentiment + top reasoning excerpts."""
    if df.empty or "ticker" not in df.columns:
        return []
    valid = df.dropna(subset=["ticker"])
    if valid.empty:
        return []
    out: list[SectorRollup] = []
    # Group by ticker. Mean over sentiment_score; top 2 reasonings by absolute
    # sentiment magnitude (strongest takes are most informative).
    for ticker, group in valid.groupby("ticker"):
        mean = float(group["sentiment_score"].mean())
        n = int(len(group))
        sorted_by_mag = group.assign(
            _mag=group["sentiment_score"].abs()
        ).sort_values("_mag", ascending=False)
        excerpts: list[str] = []
        for _, r in sorted_by_mag.head(2).iterrows():
            txt = (r["reasoning"] or "").strip()
            if txt:
                excerpts.append(
                    f"[{r['author']}, score={int(r['sentiment_score']):+d}] {txt}"
                )
        out.append(SectorRollup(
            ticker=str(ticker), mean_sentiment=mean, n_obs=n,
            top_excerpts=excerpts,
        ))
    # Stable order: by ticker.
    return sorted(out, key=lambda s: s.ticker)


def _build_macro_snapshots() -> list[MacroSnapshot]:
    """Snapshot every macro indicator the dashboard tracks."""
    out: list[MacroSnapshot] = []
    try:
        macro = fetch_macro_prices()
    except Exception:
        macro = pd.DataFrame()
    try:
        vix = vix_level(macro) if not macro.empty else {}
    except Exception:
        vix = {}
    try:
        gor = gold_oil_ratio(macro) if not macro.empty else {}
    except Exception:
        gor = {}
    try:
        cgr = copper_gold_ratio(macro) if not macro.empty else {}
    except Exception:
        cgr = {}
    try:
        dxy = dxy_level(macro) if not macro.empty else {}
    except Exception:
        dxy = {}
    try:
        yc = yield_curve_spread()
    except Exception:
        yc = {}
    try:
        fred = fetch_fred_indicators()
    except Exception:
        fred = {}

    out.append(_macro_snapshot(
        "VIX", vix.get("current"), vix.get("z_score_1y"),
        "z", _VIX_BANDS, vix.get("current"),
    ))
    out.append(_macro_snapshot(
        "HY OAS", fred.get("HY_OAS", {}).get("current"),
        fred.get("HY_OAS", {}).get("z_score_1y"),
        "z", _HY_OAS_BANDS, fred.get("HY_OAS", {}).get("current"),
    ))
    out.append(_macro_snapshot(
        "10Y-2Y", yc.get("current"), yc.get("slope_30d"),
        "slope", _T10Y2Y_BANDS, yc.get("current"),
    ))
    out.append(_macro_snapshot(
        "10Y nominal", fred.get("UST10", {}).get("current"),
        fred.get("UST10", {}).get("slope_30d"),
        "slope", _UST10_BANDS, fred.get("UST10", {}).get("current"),
    ))
    out.append(_macro_snapshot(
        "10Y real", fred.get("REAL_10Y", {}).get("current"),
        fred.get("REAL_10Y", {}).get("slope_30d"),
        "slope", _REAL_10Y_BANDS, fred.get("REAL_10Y", {}).get("current"),
    ))
    out.append(_macro_snapshot(
        "5Y5Y breakeven", fred.get("BREAKEVEN_5Y5Y", {}).get("current"),
        fred.get("BREAKEVEN_5Y5Y", {}).get("z_score_1y"),
        "z", _BREAKEVEN_BANDS, fred.get("BREAKEVEN_5Y5Y", {}).get("current"),
    ))
    out.append(_macro_snapshot(
        "DXY", dxy.get("current"), dxy.get("z_score_1y"),
        "z", _DXY_BANDS, dxy.get("current"),
    ))
    out.append(_macro_snapshot(
        "Gold/Oil", gor.get("current"), gor.get("z_score_1y"),
        "z", _GOLD_OIL_BANDS, gor.get("current"),
    ))
    # Copper/Gold bands are on the z-score, not the level.
    out.append(_macro_snapshot(
        "Copper/Gold", cgr.get("current"), cgr.get("z_score_1y"),
        "z", _COPPER_GOLD_Z_BANDS, cgr.get("z_score_1y"),
    ))
    return out


def gather_context(as_of: date | None = None,
                   lookback_days: int = 7) -> WeeklyRecapContext:
    """Assemble newsletter + macro context for the recap prompt.

    Args:
        as_of: anchor date for the window (default: today).
        lookback_days: how many days back to include.

    Returns:
        WeeklyRecapContext.  n_newsletters == 0 means no coverage in the
        window — caller should short-circuit and not invoke OpenAI.
    """
    as_of = as_of or date.today()
    df = _query_newsletters_in_window(as_of, lookback_days)
    excerpts = _build_newsletter_excerpts(df)
    rollups = _build_sector_rollups(df)
    macro = _build_macro_snapshots()
    return WeeklyRecapContext(
        as_of=as_of,
        lookback_days=lookback_days,
        n_newsletters=len(excerpts),
        newsletters=excerpts,
        sector_rollups=rollups,
        macro_snapshots=macro,
    )


# ---------------------------------------------------------------------------
# OpenAI synthesis
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are writing for an investor who reads this once a \
week before placing manual trades on Monday. Plain English, no jargon \
unless you also define it inline.

Ground every sector claim in either the supplied newsletter excerpts or the \
supplied macro readings. Do not invent sources.

Where newsletters and macro disagree, say so and pick a side with a one-line \
reason.

Allocation tilts must reference at least one newsletter or one macro \
indicator from the supplied context.

Cover every sector ticker in SECTOR_ETFS. If a sector has no newsletter \
coverage in the lookback window, say 'no coverage' and lean entirely on the \
macro read.

Bias: helpful, decisive, but honest about uncertainty. Avoid hedging \
language ('it depends', 'time will tell') unless the data genuinely \
conflicts.

The `weekly_summary` field is the executive summary the reader sees \
FIRST in the UI, even though you fill it last in the response. After \
you've worked through macro, sectors, and allocation, write 6-10 \
sentences in plain English that tie together: (a) what the newsletters \
said this week, (b) what the macro tape is telling us right now, (c) \
where you came out on the highest-conviction sector tilts and the \
single best reason for each, and (d) where the data genuinely \
conflicts. Read as a coherent narrative — not a list of bullets — and \
do not contradict anything in the macro / sectors / allocation \
sections."""


def _format_context_as_markdown(ctx: WeeklyRecapContext) -> str:
    """Serialise context as readable markdown — the model handles narrative
    input better than raw JSON.

    Layout: header → per-newsletter blocks → per-sector rollup table → macro
    readings table → SECTOR_ETFS reference.
    """
    lines: list[str] = []
    lines.append(f"# Weekly recap context — week ending {ctx.as_of.isoformat()}")
    lines.append(f"_lookback: {ctx.lookback_days} days · "
                 f"{ctx.n_newsletters} newsletter(s) analysed_")
    lines.append("")

    lines.append("## Newsletter excerpts")
    if not ctx.newsletters:
        lines.append("_No newsletters ingested in this window._")
    for n in ctx.newsletters:
        lines.append(f"### {n.author} — {n.publication_date.isoformat()} "
                     f"({n.overall_macro_bias})")
        if n.summary:
            lines.append(n.summary)
        if n.sector_ratings:
            lines.append("")
            lines.append("| Ticker | Score | Reasoning |")
            lines.append("|---|---:|---|")
            for r in n.sector_ratings:
                reasoning = (r.get("reasoning") or "").replace("|", "/")
                lines.append(f"| {r['ticker']} | {r['sentiment_score']:+d} "
                             f"| {reasoning} |")
        lines.append("")

    lines.append("## Sector roll-up (mean sentiment across the window)")
    if not ctx.sector_rollups:
        lines.append("_No sector ratings in this window._")
    else:
        lines.append("| Ticker | Mean | n | Top excerpts |")
        lines.append("|---|---:|---:|---|")
        for s in ctx.sector_rollups:
            excerpts = " · ".join(
                e.replace("|", "/")[:200] for e in s.top_excerpts
            ) or "—"
            lines.append(f"| {s.ticker} | {s.mean_sentiment:+.2f} | "
                         f"{s.n_obs} | {excerpts} |")
    lines.append("")

    lines.append("## Macro readings (current)")
    lines.append("| Indicator | Level | z / slope | Regime |")
    lines.append("|---|---:|---:|---|")
    for m in ctx.macro_snapshots:
        lvl = f"{m.level:.2f}" if m.level is not None else "—"
        zs = (f"{m.z_or_slope:+.2f} ({m.z_or_slope_kind})"
              if m.z_or_slope is not None else "—")
        lines.append(f"| {m.name} | {lvl} | {zs} | "
                     f"{m.band_emoji} {m.band_label} |")
    lines.append("")

    lines.append("## Reference: tickers you must cover")
    lines.append(", ".join(f"{t} ({name})"
                           for t, name in SECTOR_ETFS.items()))
    return "\n".join(lines)


def _get_openai_client():
    """Resolve the OpenAI client.  Wrapped in a function so tests can
    monkey-patch this name and inject a stub.
    """
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=OPENAI_API_KEY)


def resolve_recap_model() -> str:
    """Return the model name `generate_recap` will use. Exposed so the cache
    layer can key persistence by the same name without duplicating the lookup.
    """
    from config import settings as _settings
    return (getattr(_settings, "WEEKLY_RECAP_MODEL", None)
            or _settings.OPENAI_MODEL)


def generate_recap(context: WeeklyRecapContext) -> WeeklyRecap:
    """Call OpenAI to synthesise the weekly recap.

    Uses the project's OPENAI_MODEL unless WEEKLY_RECAP_MODEL is set in
    .env (env var or settings module attribute), letting the user opt
    into a stronger model just for synthesis without touching the rest of
    the pipeline.
    """
    model = resolve_recap_model()

    client = _get_openai_client()
    user_content = _format_context_as_markdown(context)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=WeeklyRecap,
        temperature=0.2,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(
            "OpenAI refused or failed to parse weekly recap: "
            f"{completion.choices[0].message.refusal!r}"
        )
    return parsed
