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
        Expression("XLK",  "S&P Tech (plain)",                  "plain",              1.00),
        Expression("VGT",  "Vanguard Info Tech (plain)",        "plain",              1.00),
        Expression("SOXX", "iShares Semiconductors",            "operating_leverage", 1.6,
                   "Semi earnings cycle amplifies tech beta."),
        Expression("SMH",  "VanEck Semiconductors",             "operating_leverage", 1.6),
        Expression("IGV",  "iShares Software",                  "thematic",           1.1),
        Expression("ICLN", "iShares Global Clean Energy",       "thematic",           1.4),
        Expression("CIBR", "First Trust Cybersecurity",         "thematic",           1.2,
                   "Secular cyber spend; lower beta than semis."),
        Expression("HACK", "ETFMG Prime Cybersecurity",         "thematic",           1.2),
        Expression("WCLD", "WisdomTree Cloud Computing",        "thematic",           1.4,
                   "Pure-play SaaS/cloud; higher vol than IGV."),
        Expression("SKYY", "First Trust Cloud Computing",       "thematic",           1.2),
        Expression("AIQ",  "Global X AI & Technology",          "thematic",           1.3),
        Expression("BOTZ", "Global X Robotics & AI",            "thematic",           1.2),
        Expression("ROBO", "ROBO Global Robotics & Automation", "thematic",           1.2),
        Expression("QTUM", "Defiance Quantum & ML",             "thematic",           1.7,
                   "Includes enabling semi/defense layer; behaves like high-beta semi basket."),
        Expression("WQTM", "WisdomTree Quantum Computing",      "thematic",           2.0,
                   "Pure-play quantum names; thin AUM, options-like volatility."),
    ],
    "XLY": [
        Expression("XLY",  "S&P Discretionary (plain)",                    "plain",              1.00),
        Expression("XHB",  "SPDR Homebuilders",                            "operating_leverage", 1.4,
                   "Homebuilder margins lever to rate cuts."),
        Expression("ITB",  "iShares Home Construction",                    "operating_leverage", 1.5,
                   "Pure-play builders; tighter rate sensitivity than XHB."),
        Expression("XRT",  "SPDR Retail (equal-weight)",                   "thematic",           1.2),
        Expression("IDRV", "iShares Self-Driving EV & Tech",               "thematic",           1.4),
        Expression("CARZ", "First Trust S-Network Future Vehicles & Tech", "thematic",           1.3),
        Expression("PEJ",  "Invesco Leisure & Entertainment",              "thematic",           1.2),
        Expression("JETS", "U.S. Global Jets",                             "operating_leverage", 1.4,
                   "Airlines — high operating leverage to fuel/load factor."),
    ],
    "XLC": [
        Expression("XLC",  "S&P Communications (plain)", "plain", 1.00),
        Expression("VOX",  "Vanguard Communications",    "plain", 1.00),
    ],
    "XLF": [
        Expression("XLF",  "S&P Financials (plain)",                    "plain",              1.00),
        Expression("KRE",  "SPDR Regional Banks",                       "operating_leverage", 1.5,
                   "NIM-driven earnings lever to curve steepening."),
        Expression("KBE",  "SPDR Broad Banks",                          "thematic",           1.2),
        Expression("IAI",  "iShares Broker-Dealers",                    "thematic",           1.3),
        Expression("KIE",  "SPDR Insurance (EW)",                       "thematic",           0.9,
                   "Insurance trades distinct from banks; lower curve sensitivity."),
        Expression("IAK",  "iShares Insurance",                         "thematic",           0.9),
        Expression("KBWP", "Invesco KBW Property & Casualty Insurance", "thematic",           0.8),
        Expression("KCE",  "SPDR Capital Markets",                      "thematic",           1.3),
        Expression("FINX", "Global X FinTech",                          "thematic",           1.3),
        Expression("IPAY", "Amplify Mobile Payments",                   "thematic",           1.2),
    ],
    "XLI": [
        Expression("XLI",  "S&P Industrials (plain)",        "plain",              1.00),
        Expression("ITA",  "iShares US Aerospace & Defense", "thematic",           1.1),
        Expression("XAR",  "SPDR Aerospace & Defense (EW)",  "thematic",           1.2),
        Expression("PAVE", "Global X Infrastructure",        "thematic",           1.1),
        Expression("IYT",  "iShares Transportation",         "operating_leverage", 1.3,
                   "Rails/truckers/airfreight — cyclical leading indicator."),
        Expression("XTN",  "SPDR Transportation (EW)",       "operating_leverage", 1.4),
    ],
    "XLB": [
        Expression("XLB",  "S&P Materials (plain)",              "plain",              1.00),
        Expression("XME",  "SPDR Metals & Mining (EW)",          "operating_leverage", 1.7,
                   "Equal-weight steel/coal/diversified miners; cyclical torque."),
        Expression("GDX",  "VanEck Gold Miners",                 "operating_leverage", 2.0,
                   "Miner earnings lever to gold price via fixed AISC."),
        Expression("GDXJ", "VanEck Junior Gold Miners",          "operating_leverage", 2.5,
                   "Higher operating leverage than seniors; deeper drawdowns."),
        Expression("SIL",  "Global X Silver Miners",             "operating_leverage", 2.2),
        Expression("SILJ", "Amplify Junior Silver Miners",       "operating_leverage", 2.8),
        Expression("COPX", "Global X Copper Miners",             "operating_leverage", 1.8),
        Expression("LIT",  "Global X Lithium & Battery",         "operating_leverage", 1.6),
        Expression("URA",  "Global X Uranium",                   "operating_leverage", 1.7),
        Expression("URNM", "Sprott Uranium Miners",              "operating_leverage", 2.0),
        Expression("REMX", "VanEck Rare Earth/Strategic Metals", "operating_leverage", 1.8,
                   "Critical mineral pure-play; thin float, high vol."),
        Expression("MOO",  "VanEck Agribusiness",                "thematic",           1.0),
    ],
    "XLE": [
        Expression("XLE",  "S&P Energy (plain)",      "plain",              1.00),
        Expression("XOP",  "SPDR Oil & Gas E&P (EW)", "operating_leverage", 1.8,
                   "E&P earnings beta to WTI ~2x vs XLE's ~1.2x."),
        Expression("OIH",  "VanEck Oil Services",     "operating_leverage", 1.7),
        Expression("FCG",  "First Trust Natural Gas", "operating_leverage", 1.9),
        Expression("AMLP", "Alerian MLP",             "thematic",           0.9),
    ],
    "XLV": [
        Expression("XLV",  "S&P Health Care (plain)",         "plain",              1.00),
        Expression("XBI",  "SPDR Biotech (equal-weight)",     "operating_leverage", 1.5,
                   "Equal-weight construction gives mid/small biotech torque."),
        Expression("IBB",  "iShares Biotech (cap-weight)",    "thematic",           1.2),
        Expression("IHI",  "iShares Medical Devices",         "thematic",           1.0),
        Expression("XPH",  "SPDR Pharmaceuticals (EW)",       "thematic",           0.9,
                   "Equal-weight pharma; distinct from XBI biotech beta."),
        Expression("PPH",  "VanEck Pharmaceutical",           "thematic",           0.9),
        Expression("IHF",  "iShares US Healthcare Providers", "thematic",           1.0,
                   "Insurers/HMOs; cyclical to medical-cost trend."),
    ],
    "XLP": [
        Expression("XLP",  "S&P Staples (plain)", "plain", 1.00),
        Expression("VDC",  "Vanguard Staples",    "plain", 1.00),
    ],
    "XLU": [
        Expression("XLU",  "S&P Utilities (plain)",  "plain",    1.00),
        Expression("GRID", "First Trust Smart Grid", "thematic", 1.2,
                   "Grid buildout / AI power infrastructure."),
    ],
    "XLRE": [
        Expression("XLRE", "S&P Real Estate (plain)",      "plain",    1.00),
        Expression("VNQ",  "Vanguard REITs",               "plain",    1.00),
        Expression("DTCR", "Pacer Data Center REITs",      "thematic", 1.2),
        Expression("REM",  "iShares Mortgage Real Estate", "thematic", 1.3,
                   "mREITs; rate-sensitive, distinct from equity REIT beta."),
        Expression("MORT", "VanEck Mortgage REIT Income",  "thematic", 1.3),
    ],
    # Supplementary 12th sector — overlaps XLI/XLK/XLC. Excluded from the
    # equal-weight target allocation via SUPPLEMENTARY_SECTORS in settings.py.
    # Sized separately as a tactical overlay. Do not duplicate ITA/XAR here;
    # they stay under XLI (sector_for_ticker returns first match).
    "UFO": [
        Expression("UFO",  "Procure Space (pure-play)",        "plain",    1.00,
                   "OG space pure-play; Rocket Lab/MDA/Viasat top weights."),
        Expression("ARKX", "ARK Space & Defense Innovation",   "thematic", 1.1,
                   "Active; includes A&D — overlaps XLI's ITA/XAR."),
        Expression("ROKT", "SPDR Kensho Final Frontiers",      "thematic", 1.0,
                   "Space + deep-sea exploration; broader taxonomy."),
        Expression("NASA", "Tema Space Innovators",            "thematic", 1.3,
                   "Active; ~10% SpaceX via SPV. WARMING_UP until ~Feb 2027."),
        Expression("ORBX", "Global X Space Tech",              "thematic", 1.2,
                   "Tight space-exploration focus. Launched Apr 2026 — WARMING_UP."),
        Expression("XOVR", "ERShares Private-Public Crossover","thematic", 1.2,
                   "~10% SpaceX SPV; not pure space but a SpaceX vehicle."),
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
