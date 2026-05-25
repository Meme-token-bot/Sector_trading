"""Pydantic schemas — also drive OpenAI Structured Outputs.

The schemas are deliberately rigid: enum macro bias, fixed-domain ticker list,
integer -5..+5 sentiment. This forces gpt-4o-mini to commit rather than emit
mealy-mouthed adjectives we can't aggregate.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# 11 SPDR Select Sector ETFs plus UFO (supplementary Space sector).  Keep in
# sync with config.settings.SECTOR_ETFS — if you add a sector there, add it
# here too or the LLM tagging will fail structured-output validation.
SectorTicker = Literal[
    "XLK", "XLY", "XLC", "XLF", "XLI", "XLB",
    "XLE", "XLV", "XLP", "XLU", "XLRE", "UFO",
]


class MacroBias(str, Enum):
    DEFENSIVE = "Defensive"
    EXPANSIONARY = "Expansionary"
    NEUTRAL = "Neutral"


class SectorRating(BaseModel):
    ticker: SectorTicker
    sentiment_score: int = Field(
        ..., ge=-5, le=5,
        description="Author's view on this sector. -5 strongly bearish, "
                    "0 neutral/no clear view, +5 strongly bullish.",
    )
    reasoning: str = Field(
        ..., max_length=400,
        description="One or two sentences capturing the author's actual stated "
                    "reasoning. Do NOT invent — if the author did not address "
                    "the sector, do not include it.",
    )


class NewsletterAnalysis(BaseModel):
    """One newsletter -> one of these. Persisted to SQLite."""
    author: str = Field(..., description="Author or publication name.")
    publication_date: date = Field(..., description="Date the piece was published.")
    overall_macro_bias: MacroBias
    sector_ratings: list[SectorRating] = Field(
        default_factory=list,
        description="Only include sectors the author explicitly discusses. "
                    "Omit sectors that are not addressed — do not fill with zeros.",
    )
    summary: str = Field(
        ..., max_length=600,
        description="2-3 sentence neutral summary of the macro thesis.",
    )


# ---------------------------------------------------------------------------
# Weekly Recap schemas
# ---------------------------------------------------------------------------
# Context is the input we build from DB + macro snapshots; recap is the
# structured response we ask gpt-4o-mini to produce.  The recap shape is the
# response_format passed to client.beta.chat.completions.parse(), so all
# fields must be JSON-schema representable (Pydantic handles this).


class NewsletterConsensus(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    MIXED = "mixed"
    NO_COVERAGE = "no coverage"


class AllocationTilt(str, Enum):
    OVERWEIGHT = "Overweight"
    EQUAL_WEIGHT = "Equal-weight"
    UNDERWEIGHT = "Underweight"
    AVOID = "Avoid"


class RegimeLabel(str, Enum):
    RISK_ON = "Risk-on"
    RISK_OFF = "Risk-off"
    LATE_CYCLE = "Late-cycle"
    REFLATIONARY = "Reflationary"
    DISINFLATIONARY = "Disinflationary"
    MIXED = "Mixed"


class WatchDirection(str, Enum):
    BUILDING = "building"          # support gathering ahead of a price confirm
    ROLLING_OVER = "rolling over"  # currently strong but at risk of topping


# --- input side (not sent to OpenAI as response_format) ---

class NewsletterExcerpt(BaseModel):
    """One newsletter as the recap builder sees it."""
    author: str
    publication_date: date
    overall_macro_bias: str
    summary: str
    sector_ratings: list[dict] = Field(default_factory=list)


class SectorRollup(BaseModel):
    """Per-sector aggregation across the lookback window."""
    ticker: str
    mean_sentiment: float
    n_obs: int
    top_excerpts: list[str] = Field(default_factory=list)


class MacroSnapshot(BaseModel):
    """One macro indicator's current reading + regime band."""
    name: str
    level: float | None = None
    z_or_slope: float | None = None
    z_or_slope_kind: str = "z"          # "z" or "slope"
    band_label: str = "—"
    band_emoji: str = "⚪"


class SignalRow(BaseModel):
    """Per-sector convergence snapshot (trend + RS + sentiment + macro + state).

    Lets the recap reason about *gaps* between the layers — e.g. sentiment and
    macro support a sector but price (RS / SMA200) hasn't confirmed yet.
    """
    ticker: str
    above_sma: bool
    rs_3m_pct: float            # 3-month relative strength vs benchmark, in %
    rs_rank: int
    sentiment: float            # mean newsletter sentiment in the window
    state: str                  # NEW_BUY / HOLD_IF_LONG / CHASE / REDUCE / WATCH / HOLD / SELL
    conviction: int             # 0-5
    macro_tailwinds: int
    macro_headwinds: int


class WeeklyRecapContext(BaseModel):
    """Assembled input for generate_recap()."""
    as_of: date
    lookback_days: int
    n_newsletters: int
    newsletters: list[NewsletterExcerpt] = Field(default_factory=list)
    sector_rollups: list[SectorRollup] = Field(default_factory=list)
    macro_snapshots: list[MacroSnapshot] = Field(default_factory=list)
    signal_rows: list[SignalRow] = Field(default_factory=list)


# --- output side (response_format) ---

class SectorRecap(BaseModel):
    ticker: SectorTicker
    sector_name: str
    plain_language_summary: str = Field(
        ..., max_length=800,
        description="3-5 plain-English sentences. No jargon unless defined inline.",
    )
    newsletter_consensus: NewsletterConsensus
    macro_alignment: str = Field(
        ..., max_length=400,
        description="How current macro readings support or contradict the "
                    "newsletter view for this sector.",
    )
    key_risks: list[str] = Field(
        default_factory=list,
        description="1-3 short bullets — concrete risks for this sector.",
    )


class MacroNarrative(BaseModel):
    regime_label: RegimeLabel
    summary: str = Field(
        ..., max_length=1000,
        description="4-6 plain-English sentences synthesising the macro tape "
                    "and the newsletter consensus.",
    )
    dominant_themes: list[str] = Field(
        default_factory=list,
        description="2-4 short themes drawn from the supplied newsletters.",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Cases where newsletters disagree with each other or with "
                    "the macro tape.  Empty list if broadly aligned.",
    )


class Allocation(BaseModel):
    ticker: SectorTicker
    suggested_tilt: AllocationTilt
    rationale: str = Field(
        ..., max_length=300,
        description="One sentence.  Must reference a newsletter excerpt or a "
                    "macro indicator from the supplied context.",
    )


class SectorWatch(BaseModel):
    """Forward-looking flag for a sector worth watching next week.

    Unlike `Allocation` (what to own now), this anticipates a *change*: a
    convergence gap that is about to open or close.
    """
    ticker: SectorTicker
    sector_name: str
    direction: WatchDirection
    rationale: str = Field(
        ..., max_length=400,
        description="Why this sector is on the watchlist. Must cite the "
                    "convergence GAP from the signal rows — e.g. sentiment and "
                    "macro support it but RS/price haven't confirmed (building), "
                    "or it's strong but extended / macro is turning (rolling over). "
                    "Reference at least one signal-row fact plus a newsletter or "
                    "macro reading.",
    )
    what_to_watch: str = Field(
        ..., max_length=200,
        description="The concrete trigger that would confirm the move, e.g. "
                    "'3-month RS turning positive', 'reclaim of SMA200', "
                    "'HY OAS widening past 4%'.",
    )


class WeeklyRecap(BaseModel):
    """Structured response from gpt-4o-mini for the Weekly Recap tab.

    Field order is deliberate.  Structured-output models fill fields in
    declaration order, so the per-sector and per-allocation details are
    decided BEFORE ``weekly_summary`` is written — meaning the summary can
    legitimately reference what's in the rest of the recap.  The UI then
    renders ``weekly_summary`` at the TOP for the reader, even though the
    model wrote it last.
    """
    generated_for_week_ending: date
    n_newsletters: int
    macro: MacroNarrative
    sectors: list[SectorRecap] = Field(
        default_factory=list,
        description="One entry per ticker in SECTOR_ETFS.  If a sector has no "
                    "coverage, newsletter_consensus='no coverage' and the "
                    "summary should lean on the macro snapshot.",
    )
    allocation: list[Allocation] = Field(
        default_factory=list,
        description="Ranked, highest-conviction tilts first.",
    )
    sectors_to_watch: list[SectorWatch] = Field(
        default_factory=list,
        description="2-5 forward-looking watch flags. Anticipate where a "
                    "convergence gap is about to open or close next week — not "
                    "what to own now (that's `allocation`). Omit (empty list) "
                    "only if nothing is set up. Declared before weekly_summary "
                    "so the summary can reference these.",
    )
    weekly_summary: str = Field(
        ..., max_length=1500,
        description="The executive summary the reader sees first.  Write "
                    "this LAST, after macro/sectors/allocation are decided.  "
                    "6-10 sentences in plain English that tie together: "
                    "(a) what newsletters said this week, (b) what the macro "
                    "tape is telling us right now, (c) where you came out on "
                    "the highest-conviction sector tilts and why, and (d) "
                    "where the data genuinely conflicts.  Should read as a "
                    "coherent narrative — not a list of bullet points — and "
                    "should not contradict any sector or allocation entry.",
    )
    caveats: str = Field(
        ..., max_length=400,
        description="1-2 sentences: informational only, not personalised advice.",
    )
