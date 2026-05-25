"""Central configuration. One place to change a threshold or add a ticker."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --- paths --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "sentiment.db"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")

# --- universe -----------------------------------------------------------
SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLY":  "Consumer Discretionary",
    "XLC":  "Communication Services",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "UFO":  "Space",
}
# Sectors that produce signals but do NOT participate in the equal-weight
# target allocation that the 11 SPDR sectors share. Treated as tactical
# overlays — the user sizes them separately. target_weights() filters
# these out; compute_drift_by_sector() stashes them in df.attrs["supplementary"]
# so the UI can render them without a drift comparison.
SUPPLEMENTARY_SECTORS: frozenset[str] = frozenset({"UFO"})
BENCHMARK = "SPY"
MACRO_TICKERS = {
    "GOLD":   "GC=F",
    "OIL":    "CL=F",
    "DXY":    "DX-Y.NYB",
    "COPPER": "HG=F",
    "VIX":    "^VIX",
}

# FRED series IDs pulled via the direct-CSV pattern (see market_engine).
# Kept here so callers reference logical names, not bare series strings.
FRED_SERIES = {
    "HY_OAS":         "BAMLH0A0HYM2",   # ICE BofA US HY OAS, %, daily
    "UST10":          "DGS10",          # 10Y constant-maturity nominal, %, daily
    "REAL_10Y":       "DFII10",         # 10Y TIPS real yield, %, daily
    "BREAKEVEN_5Y5Y": "T5YIFR",         # 5y5y forward inflation expectation, %, daily
    "T10Y2Y":         "T10Y2Y",         # 10Y - 2Y spread, %, daily
}

# --- signal parameters --------------------------------------------------
@dataclass(frozen=True)
class SignalParams:
    sma_window: int = 200
    momentum_window: int = 63
    sentiment_lookback_days: int = 60
    buy_sentiment_threshold: float = 2.0
    sell_sentiment_threshold: float = -3.0
    weak_rs_rank_cutoff: int = 3
    # Late-entry guard: if (price-SMA200)/SMA200 > this, BUY -> CHASE
    extension_pct_cutoff: float = 0.12
    # If a sector has been BUY for >= this many consecutive weekly snapshots,
    # downgrade BUY -> HOLD_IF_LONG (don't add fresh, hold if owned)
    stale_buy_weeks: int = 4
    # How many weekly snapshots to replay for state classification
    history_weeks: int = 12
    # Conviction scoring: relative_strength_3m must exceed this margin
    # (in decimal form, e.g. 0.03 = 3%) to earn the "strong RS" point.
    strong_rs_margin: float = 0.03
    # Macro overlay (net = tailwinds - headwinds from compute_macro_alignment):
    #   * conviction reacts to ANY clear lean (net >= +1 / <= -1), symmetric.
    #   * state veto/override fires only on a STRONG lean (|net| >= this).
    # Higher = macro must be more one-sided before it downgrades a BUY or
    # elevates a HOLD to WATCH. At 1 the override fires on any clear net lean,
    # in step with the conviction nudge (sector rules are now rich enough that
    # net rarely exceeds ±1 on a normal tape).
    macro_strong_count: int = 1

PARAMS = SignalParams()

# --- LLM ---------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Optional override for the Weekly Recap tab — set in .env if synthesis
# quality on gpt-4o-mini is insufficient. Defaults to OPENAI_MODEL.
WEEKLY_RECAP_MODEL = os.getenv("WEEKLY_RECAP_MODEL", "") or OPENAI_MODEL

# --- Tiger -------------------------------------------------------------
TIGER_ID = os.getenv("TIGER_ID", "")
TIGER_ACCOUNT = os.getenv("TIGER_ACCOUNT", "")
TIGER_PRIVATE_KEY_PATH = os.getenv("TIGER_PRIVATE_KEY_PATH", "")
TIGER_SANDBOX = os.getenv("TIGER_SANDBOX", "false").lower() == "true"

def tiger_configured() -> bool:
    return all([TIGER_ID, TIGER_ACCOUNT, TIGER_PRIVATE_KEY_PATH,
                Path(TIGER_PRIVATE_KEY_PATH).exists()])

# --- Gmail (Gmail REST API + OAuth 2.0) --------------------------------
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_FILTER_ADDRESS = os.getenv("GMAIL_FILTER_ADDRESS", "")
# OAuth client_secret.json downloaded from Google Cloud Console.
GMAIL_CREDENTIALS_FILE = os.getenv(
    "GMAIL_CREDENTIALS_FILE", "credentials/gmail_credentials.json")
# Token (with refresh_token) written by scripts/gmail_oauth_setup.py.
GMAIL_TOKEN_FILE = os.getenv(
    "GMAIL_TOKEN_FILE", "credentials/gmail_token.json")

def gmail_configured() -> bool:
    return bool(GMAIL_ADDRESS and Path(GMAIL_TOKEN_FILE).exists())

# --- Content extraction -------------------------------------------------
@dataclass(frozen=True)
class ExtractionParams:
    max_links_per_newsletter: int = 5
    max_total_chars: int = 40_000
    fetch_timeout_seconds: int = 15

EXTRACTION = ExtractionParams()

# --- Expression theme news ---------------------------------------------
@dataclass(frozen=True)
class ExpressionParams:
    # Blend weight for the automated news leg when combining with newsletter
    # theme sentiment: combined = (1 - w)*newsletter + w*news. Newsletters are
    # deeper/slower, news is broader/fresher; default leans on newsletters.
    theme_news_weight: float = 0.4
    # |combined theme sentiment| at/above which the picker raises a news flag
    # (price-vs-news contradiction / divergence).
    theme_news_flag_threshold: float = 2.0
    # How stale a stored theme_news score may be before it's ignored.
    theme_news_max_age_days: int = 14

EXPRESSION = ExpressionParams()
