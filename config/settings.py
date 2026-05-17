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
}
BENCHMARK = "SPY"
MACRO_TICKERS = {"GOLD": "GC=F", "OIL": "CL=F"}

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

PARAMS = SignalParams()

# --- LLM ---------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Tiger -------------------------------------------------------------
TIGER_ID = os.getenv("TIGER_ID", "")
TIGER_ACCOUNT = os.getenv("TIGER_ACCOUNT", "")
TIGER_PRIVATE_KEY_PATH = os.getenv("TIGER_PRIVATE_KEY_PATH", "")
TIGER_SANDBOX = os.getenv("TIGER_SANDBOX", "false").lower() == "true"

def tiger_configured() -> bool:
    return all([TIGER_ID, TIGER_ACCOUNT, TIGER_PRIVATE_KEY_PATH,
                Path(TIGER_PRIVATE_KEY_PATH).exists()])

# --- Gmail -------------------------------------------------------------
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
GMAIL_FILTER_ADDRESS = os.getenv("GMAIL_FILTER_ADDRESS", "")
GMAIL_IMAP_HOST = os.getenv("GMAIL_IMAP_HOST", "imap.gmail.com")
GMAIL_IMAP_PORT = int(os.getenv("GMAIL_IMAP_PORT", "993"))

def gmail_configured() -> bool:
    return bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD)

# --- Content extraction -------------------------------------------------
@dataclass(frozen=True)
class ExtractionParams:
    max_links_per_newsletter: int = 5
    max_total_chars: int = 40_000
    fetch_timeout_seconds: int = 15

EXTRACTION = ExtractionParams()
