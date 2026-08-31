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
    "BREAKEVEN_10Y":  "T10YIE",         # 10Y breakeven inflation rate, %, daily
    "INIT_CLAIMS":    "IC4WSA",         # 4-week avg initial claims, thousands, weekly
    "MORTGAGE_30Y":   "MORTGAGE30US",   # 30Y fixed mortgage rate, %, weekly
    "FIN_CONDITIONS": "NFCI",           # Chicago Fed NFCI (z-score, loose<0), weekly
    "UST2":           "DGS2",           # 2Y constant-maturity nominal, %, daily
    "IG_OAS":         "BAMLC0A4CBBB",   # ICE BofA BBB OAS (investment grade), %, daily
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
    # Partial-CHASE participation. When > 0, CHASE-state sectors enter at this
    # fraction of the per-name equal-weight target, FUNDED FROM THE CASH
    # BUFFER (capped so total weight never exceeds 1.0 — no implicit
    # leverage). Set 0 to fully exclude CHASE — the original behaviour.
    #
    # 0.25 was selected by walk-forward sweep (5/6 fold consensus, mean OOS
    # lift +2.18pp in excess CAGR vs the 0.0 default). See
    # WALK_FORWARD_REPORT.md. Sole parameter change the sweep's robustness
    # rule (positive mean OOS lift > 0.5pp) cleared; every other tunable
    # underperformed default OOS and was kept as-is.
    #
    # Note the v1 sweep reported +5.35pp; that number was inflated because
    # the old sleeve implementation allowed total weight > 1.0 (implicit
    # leverage). The capped v2 number is the honest one.
    chase_weight_fraction: float = 0.25

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

# --- Breakout / consolidation / money-flow signal (independent of PARAMS
# above — this system is deliberately NOT sentiment-gated; see
# src/breakout_signals.py module docstring for why). ---------------------
@dataclass(frozen=True)
class BreakoutParams:
    sma_short: int = 50
    sma_long: int = 150
    slope_lookback: int = 10                  # trading days, for Rising/Flat/Falling
    slope_deadband_pct_per_day: float = 0.0005
    consolidation_min_days: int = 60          # ~3 months, per spec
    breakout_lookback_days: int = 10          # how long a breakout stays "active"
    atr_period: int = 14
    atr_baseline_period: int = 252
    atr_compression_threshold: float = 0.75   # current ATR% <= 75% of its own 1y avg
    bb_period: int = 20
    bb_lookback: int = 252
    bbw_percentile_threshold: float = 0.25    # bottom quartile of its own 1y BBW history
    compression_require: str = "either"       # "atr" | "bbw" | "either" | "both"
    cmf_period: int = 20
    obv_period: int = 14
    obv_vol_norm_period: int = 63

BREAKOUT = BreakoutParams()


@dataclass(frozen=True)
class MoneyFlowThresholds:
    """Configurable classification bands -- documented starting priors, not
    fitted. Recalibrate once signal_snapshots-style forward history exists,
    same caveat this codebase already applies to macro/dispersion bands."""
    cmf_strong: float = 0.15
    cmf_mild: float = 0.05
    obv_slope_strong: float = 0.15
    obv_slope_mild: float = 0.05

MONEY_FLOW = MoneyFlowThresholds()



# --- Industry Rotation Grid (src/industry_rotation.py) ------------------
@dataclass(frozen=True)
class IndustryRotationParams:
    """Starting heuristic for the Industry Rotation Grid's 4-state
    classifier (CLIMBING / BASE / TIRED / DOWNHILL) — NOT fitted or
    validated, recalibrate once forward `industry_rotation_snapshots`
    history exists to check state transitions against realized forward
    returns. Mirrors the same caveat DISPERSION_BANDS already carries in
    src/regime_snapshot.py and MoneyFlowThresholds carries above.
    """
    trend_ma_window: int = 50
    rs_window: int = 63
    rs_slope_window: int = 20
    rs_slope_threshold: float = 0.01

INDUSTRY_ROTATION = IndustryRotationParams()


# --- Fund Screener (src/fund_screener.py) --------------------------------
@dataclass(frozen=True)
class FundScoreParams:
    """Starting heuristic for the Fund Screener's 0-100 composite score
    ('Winston Score' framing) — weights are a documented starting point,
    not fitted or backtested. Recalibrate once forward-return tracking
    exists for screened picks. Same caveat as IndustryRotationParams above.
    """
    return_weight: float = 0.50
    liquidity_weight: float = 0.30
    fee_weight: float = 0.20
    return_lookback_days: int = 252

FUND_SCORE_PARAMS = FundScoreParams()
