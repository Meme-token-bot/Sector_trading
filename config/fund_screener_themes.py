"""Fund Screener theme taxonomy — src/fund_screener.py, 🔍 Fund Screener tab.

Broader and shallower than config/themes.py's 33 sentiment-tagging themes.
config/themes.py exists to let the LLM newsletter-tagger and the
Expressions tab's per-vehicle self-check reason about a CURATED, narrow
vehicle list per SPDR sector. This module exists for DISCOVERY instead: cast
a wide-but-honest net of candidate funds per theme so the screener can rank
and surface options that have no natural SECTOR_ETFS parent at all (a single
country, a factor tilt, a bond ladder, "Dividend Growth" vs plain
"Dividend") — the kind of theme a person browses by, not a sector.

Where a theme genuinely overlaps a config/themes.py entry, the ticker list
is reused (see the per-entry `note`) rather than re-derived from scratch —
same discipline as config/industries.py.

Rules (same as config/expressions.py and config/industries.py):
  * NO leveraged or inverse products (no daily-reset 2x/3x, no "short" funds).
  * NO individual stocks — funds only, always.
  * A ticker appearing in more than one theme is expected and fine (e.g. LIT
    legitimately belongs to both "Lithium/EV" and, arguably, "Clean Energy")
    — themes are a browsing lens, not a partition.

Every ticker here is either already used elsewhere in this codebase or a
long-established, high-AUM, single-purpose fund. `LEVERAGED_EXCLUDED`
documents the one screenshot-matching category deliberately left empty.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenerTheme:
    key: str
    label: str
    tickers: tuple[str, ...]
    note: str = ""


SCREENER_THEMES: dict[str, ScreenerTheme] = {
    "AI": ScreenerTheme(
        "AI", "Artificial Intelligence",
        ("BOTZ", "ROBO", "AIQ", "QTUM"),
        "Reused from config/themes.py AI_ROBOTICS + QUANTUM."),
    "AI_INFRASTRUCTURE": ScreenerTheme(
        "AI_INFRASTRUCTURE", "AI Infrastructure",
        ("SMH", "SOXX", "DTCR", "SRVR", "GRID"),
        "'Picks and shovels' construction: compute (semis), the real "
        "estate that houses it (data-center REITs), and the power grid "
        "that feeds it (smart grid) — not a single-fund theme by nature."),
    "SEMICONDUCTORS": ScreenerTheme(
        "SEMICONDUCTORS", "Semiconductors", ("SMH", "SOXX", "PSI"),
        "Reused from config/industries.py SEMICONDUCTORS."),
    "DEFENSE": ScreenerTheme(
        "DEFENSE", "Defense", ("ITA", "XAR", "PPA"),
        "Reused from config/industries.py DEFENSE_AEROSPACE. No confidently "
        "-liquid drones-only fund exists (the screenshot's 'Drones & "
        "Warfare' theme is folded in here rather than fabricated)."),
    "ENERGY": ScreenerTheme(
        "ENERGY", "Energy", ("XLE", "XOP", "OIH", "AMLP"),
        "Reused from config/industries.py XLE industries."),
    "NUCLEAR": ScreenerTheme(
        "NUCLEAR", "Nuclear", ("NLR", "URA"),
        "NLR = broader nuclear utilities + fuel-cycle basket; URA = pure "
        "uranium-mining tilt (see URANIUM theme below for miners specifically)."),
    "URANIUM": ScreenerTheme(
        "URANIUM", "Uranium", ("URA", "URNM"),
        "Reused from config/industries.py URANIUM_MINING / config/expressions.py."),
    "GOLD": ScreenerTheme(
        "GOLD", "Gold", ("GLD", "IAU", "SGOL", "GDX", "GDXJ"),
        "Spot bullion (GLD/IAU/SGOL) alongside miners (GDX/GDXJ) — "
        "deliberately mixed so the screener shows both legs of the trade."),
    "SILVER": ScreenerTheme(
        "SILVER", "Silver", ("SLV", "SIL", "SILJ"),
        "Reused from config/industries.py SILVER_MINERS / config/expanded_universe.py."),
    "COPPER": ScreenerTheme(
        "COPPER", "Copper", ("COPX", "PICK"),
        "Reused from config/industries.py COPPER_MINERS / config/expanded_universe.py."),
    "CYBERSECURITY": ScreenerTheme(
        "CYBERSECURITY", "Cybersecurity", ("CIBR", "HACK"),
        "Reused from config/expressions.py CYBER theme."),
    "ROBOTICS_AUTOMATION": ScreenerTheme(
        "ROBOTICS_AUTOMATION", "Robotics & Automation", ("BOTZ", "ROBO"),
        "Reused from config/expressions.py AI_ROBOTICS theme."),
    "SPACE": ScreenerTheme(
        "SPACE", "Space", ("ARKX", "UFO", "ROKT"),
        "Reused from config/expressions.py SPACE theme / config/themes.py."),
    "LITHIUM_EV": ScreenerTheme(
        "LITHIUM_EV", "Lithium / EV", ("LIT", "IDRV", "CARZ"),
        "Reused from config/expressions.py LITHIUM_BATTERY + AUTOS_EV themes."),
    "CANNABIS": ScreenerTheme(
        "CANNABIS", "Cannabis", ("MSOS", "MJ"),
        "New addition. CAUTION: cannabis-sector funds have had an elevated "
        "closure/reorg rate historically — verify current listing status "
        "before relying on this theme operationally."),
    "CLEAN_ENERGY": ScreenerTheme(
        "CLEAN_ENERGY", "Clean Energy", ("ICLN", "TAN"),
        "Reused from config/expressions.py CLEAN_ENERGY theme; TAN adds "
        "solar-specific breadth ICLN alone doesn't fully capture."),
    "QUANTUM_COMPUTING": ScreenerTheme(
        "QUANTUM_COMPUTING", "Quantum Computing", ("QTUM", "WQTM"),
        "Reused from config/expressions.py QUANTUM theme."),
    "INFRASTRUCTURE": ScreenerTheme(
        "INFRASTRUCTURE", "Infrastructure", ("PAVE", "IFRA"),
        "Reused from config/industries.py INFRASTRUCTURE_CONSTRUCTION."),
    "BIOTECH": ScreenerTheme(
        "BIOTECH", "Biotech", ("XBI", "IBB"),
        "Reused from config/expressions.py BIOTECH theme."),
    "HEALTHCARE": ScreenerTheme(
        "HEALTHCARE", "Healthcare", ("XLV", "IYH", "IHI"),
        "Broad sector + medical devices — distinct from the BIOTECH theme's "
        "narrower, higher-beta focus."),
    "CONSTRUCTION": ScreenerTheme(
        "CONSTRUCTION", "Construction", ("ITB", "XHB", "PAVE"),
        "Reused from config/industries.py HOME_BUILDERS + INFRASTRUCTURE_CONSTRUCTION."),
    "INDUSTRIALS": ScreenerTheme(
        "INDUSTRIALS", "Industrials", ("XLI", "IYJ"),
        "Broad sector proxy pair."),
    "DIVIDEND": ScreenerTheme(
        "DIVIDEND", "Dividend", ("VYM", "SCHD", "DVY", "HDV"),
        "New addition — long-established, high-AUM dividend-yield funds."),
    "DIVIDEND_GROWTH": ScreenerTheme(
        "DIVIDEND_GROWTH", "Dividend Growth", ("VIG", "DGRO", "SCHD"),
        "New addition — distinct discipline from DIVIDEND above: growth-of-"
        "payout streak requirements rather than current yield."),
    "REITS": ScreenerTheme(
        "REITS", "REITs", ("VNQ", "IYR", "SCHH"),
        "Reused from config/industries.py REIT_BROAD."),
    "BONDS": ScreenerTheme(
        "BONDS", "Bonds", ("AGG", "BND", "SCHZ"),
        "New addition — core US aggregate-bond funds."),
    "TREASURIES": ScreenerTheme(
        "TREASURIES", "Treasuries", ("TLT", "IEF", "SHY", "GOVT"),
        "New addition — spans long (TLT), intermediate (IEF), short (SHY), "
        "and broad (GOVT) duration."),
    "US_LARGE_CAP": ScreenerTheme(
        "US_LARGE_CAP", "US Large Cap", ("SPY", "VOO", "IVV", "SPLG"),
        "New addition — the benchmark itself plus its lowest-cost peers, "
        "so the screener can show the very bar this whole dashboard is "
        "trying to beat."),
    "US_SMALL_CAP": ScreenerTheme(
        "US_SMALL_CAP", "US Small Cap", ("IWM", "VB", "IJR"),
        "New addition."),
    "EUROPE": ScreenerTheme(
        "EUROPE", "Europe", ("VGK", "EZU"),
        "New addition — broad developed Europe (VGK) and Eurozone-only (EZU)."),
    "UK": ScreenerTheme(
        "UK", "United Kingdom", ("EWU",),
        "New addition. Single-ticker theme — see FundScoreParams' minimum-"
        "candidates caveat in src/fund_screener.py."),
    "JAPAN": ScreenerTheme(
        "JAPAN", "Japan", ("EWJ",),
        "New addition. Single-ticker theme."),
    "CHINA": ScreenerTheme(
        "CHINA", "China", ("FXI", "MCHI"),
        "New addition."),
    "INDIA": ScreenerTheme(
        "INDIA", "India", ("INDA", "EPI"),
        "New addition."),
    "EMERGING_MARKETS": ScreenerTheme(
        "EMERGING_MARKETS", "Emerging Markets", ("VWO", "EEM", "IEMG"),
        "New addition."),
    "QUALITY": ScreenerTheme(
        "QUALITY", "Quality (factor)", ("QUAL",),
        "New addition. Single-ticker theme."),
    "MOMENTUM": ScreenerTheme(
        "MOMENTUM", "Momentum (factor)", ("MTUM",),
        "New addition. Single-ticker theme."),
    "VALUE": ScreenerTheme(
        "VALUE", "Value (factor)", ("VTV", "IVE"),
        "New addition."),
}

# Deliberately EMPTY — see module docstring. Leveraged/inverse products are
# excluded from the Fund Screener the same way config/expressions.py
# excludes them from the curated sector-vehicle list. Kept as a named,
# documented, non-deleted category so a reviewer sees the omission was a
# decision, not an oversight, rather than silently having no trace at all.
LEVERAGED_EXCLUDED: tuple[str, ...] = ()


def all_screener_tickers() -> list[str]:
    """Deduped ticker list, insertion order — for the price-update universe."""
    seen: set[str] = set()
    out: list[str] = []
    for theme in SCREENER_THEMES.values():
        for t in theme.tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out