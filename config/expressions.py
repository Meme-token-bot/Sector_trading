"""Maps each signal ETF (XLK, XLB, ...) to a curated set of expression vehicles.

All expressions are PLAIN equity ETFs. No daily-reset leveraged products
(no NUGT, FAS, SOXL, etc). The "leverage" the user wants comes from the
operating leverage of the underlying businesses — e.g. gold miners' fixed
mining costs amplify their earnings beta to gold price.

Each expression carries a `beta_hint`: a rough 3-month price beta vs the
signal ETF. These are guidance numbers shown in the UI to help the user
size positions; they are NOT used in any calculation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    ticker: str
    label: str
    kind: str          # "plain" | "thematic" | "operating_leverage"
    beta_hint: float   # rough 3M price beta vs the signal ETF
    note: str = ""


EXPRESSIONS: dict[str, list[Expression]] = {
    "XLK": [
        Expression("XLK",  "S&P Tech (plain)",             "plain", 1.00),
        Expression("VGT",  "Vanguard Info Tech (plain)",   "plain", 1.00),
        Expression("SOXX", "iShares Semiconductors",       "operating_leverage", 1.6,
                   "Semi earnings cycle amplifies tech beta."),
        Expression("SMH",  "VanEck Semiconductors",        "operating_leverage", 1.6),
        Expression("IGV",  "iShares Software",             "thematic", 1.1),
    ],
    "XLY": [
        Expression("XLY",  "S&P Discretionary (plain)",    "plain", 1.00),
        Expression("XHB",  "SPDR Homebuilders",            "operating_leverage", 1.4,
                   "Homebuilder margins lever to rate cuts."),
        Expression("XRT",  "SPDR Retail (equal-weight)",   "thematic", 1.2),
    ],
    "XLC": [
        Expression("XLC",  "S&P Communications (plain)",   "plain", 1.00),
        Expression("VOX",  "Vanguard Communications",      "plain", 1.00),
    ],
    "XLF": [
        Expression("XLF",  "S&P Financials (plain)",       "plain", 1.00),
        Expression("KRE",  "SPDR Regional Banks",          "operating_leverage", 1.5,
                   "NIM-driven earnings lever to curve steepening."),
        Expression("KBE",  "SPDR Broad Banks",             "thematic", 1.2),
        Expression("IAI",  "iShares Broker-Dealers",       "thematic", 1.3),
    ],
    "XLI": [
        Expression("XLI",  "S&P Industrials (plain)",      "plain", 1.00),
        Expression("ITA",  "iShares US Aerospace & Defense", "thematic", 1.1),
        Expression("XAR",  "SPDR Aerospace & Defense (EW)", "thematic", 1.2),
        Expression("PAVE", "Global X Infrastructure",      "thematic", 1.1),
    ],
    "XLB": [
        Expression("XLB",  "S&P Materials (plain)",        "plain", 1.00),
        Expression("GDX",  "VanEck Gold Miners",           "operating_leverage", 2.0,
                   "Miner earnings lever to gold price via fixed AISC."),
        Expression("GDXJ", "VanEck Junior Gold Miners",    "operating_leverage", 2.5,
                   "Higher operating leverage than seniors; deeper drawdowns."),
        Expression("SIL",  "Global X Silver Miners",       "operating_leverage", 2.2),
        Expression("SILJ", "Amplify Junior Silver Miners", "operating_leverage", 2.8),
        Expression("COPX", "Global X Copper Miners",       "operating_leverage", 1.8),
        Expression("LIT",  "Global X Lithium & Battery",   "operating_leverage", 1.6),
        Expression("URA",  "Global X Uranium",             "operating_leverage", 1.7),
        Expression("URNM", "Sprott Uranium Miners",        "operating_leverage", 2.0),
    ],
    "XLE": [
        Expression("XLE",  "S&P Energy (plain)",           "plain", 1.00),
        Expression("XOP",  "SPDR Oil & Gas E&P (EW)",      "operating_leverage", 1.8,
                   "E&P earnings beta to WTI ~2x vs XLE's ~1.2x."),
        Expression("OIH",  "VanEck Oil Services",          "operating_leverage", 1.7),
        Expression("FCG",  "First Trust Natural Gas",      "operating_leverage", 1.9),
        Expression("AMLP", "Alerian MLP",                  "thematic", 0.9),
    ],
    "XLV": [
        Expression("XLV",  "S&P Health Care (plain)",      "plain", 1.00),
        Expression("XBI",  "SPDR Biotech (equal-weight)",  "operating_leverage", 1.5,
                   "Equal-weight construction gives mid/small biotech torque."),
        Expression("IBB",  "iShares Biotech (cap-weight)", "thematic", 1.2),
        Expression("IHI",  "iShares Medical Devices",      "thematic", 1.0),
    ],
    "XLP": [
        Expression("XLP",  "S&P Staples (plain)",          "plain", 1.00),
        Expression("VDC",  "Vanguard Staples",             "plain", 1.00),
    ],
    "XLU": [
        Expression("XLU",  "S&P Utilities (plain)",        "plain", 1.00),
        Expression("GRID", "First Trust Smart Grid",       "thematic", 1.2,
                   "Grid buildout / AI power infrastructure."),
        Expression("ICLN", "iShares Global Clean Energy",  "thematic", 1.4),
    ],
    "XLRE": [
        Expression("XLRE", "S&P Real Estate (plain)",      "plain", 1.00),
        Expression("VNQ",  "Vanguard REITs",               "plain", 1.00),
        Expression("DTCR", "Pacer Data Center REITs",      "thematic", 1.2),
    ],
}


def all_expression_tickers() -> list[str]:
    """Flat list of every expression ticker across all sectors."""
    seen = set()
    out = []
    for exprs in EXPRESSIONS.values():
        for e in exprs:
            if e.ticker not in seen:
                seen.add(e.ticker)
                out.append(e.ticker)
    return out


def sector_for_ticker(ticker: str) -> str | None:
    """Reverse lookup: given any expression ticker, which signal sector owns it?"""
    for sector, exprs in EXPRESSIONS.items():
        if any(e.ticker == ticker for e in exprs):
            return sector
    return None
