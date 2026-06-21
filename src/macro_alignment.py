"""Per-sector macro alignment.

Maps the macro indicator panel onto each sector's structural sensitivities and
counts how many readings are currently a tailwind, a headwind, or neutral.

This module is pure: given a dict of macro readings (the same payloads the
Macro tab consumes from `src.market_engine`), it returns a tidy frame and a
detail trace per sector. Streamlit-free, network-free.

Indicator keys in `macro_readings` and where each comes from:

    From FRED (via fetch_fred_indicators):
    "HY_OAS"         high-yield OAS %        current + z_score_1y
    "IG_OAS"         BBB OAS %               current + z_score_1y
    "UST10"          10Y nominal yield %      current + slope_30d
    "UST2"           2Y nominal yield %       current + slope_30d
    "REAL_10Y"       10Y TIPS real yield %    current + slope_30d
    "BREAKEVEN_5Y5Y" 5Y5Y forward breakeven % current + z_score_1y
    "BREAKEVEN_10Y"  10Y breakeven %          current + z_score_1y
    "INIT_CLAIMS"    4-wk avg jobless claims  current + z_score_1y
    "MORTGAGE_30Y"   30Y fixed mortgage %     current + slope_30d
    "MORTGAGE_SPREAD" mortgage minus UST10 %  current only (derived)
    "FIN_CONDITIONS" NFCI (z-score, loose<0)  current + slope_30d

    From yield_curve_spread():
    "T10Y2Y"         10Y - 2Y spread %        current + slope_30d

    From yfinance (via fetch_macro_prices):
    "DXY"            US Dollar Index          current + z_score_1y
    "VIX"            VIX level                current + z_score_1y
    "GOLD_OIL"       Gold / Oil ratio         current + z_score_1y
    "COPPER_GOLD"    Copper / Gold ratio      current + z_score_1y

Each payload is a dict with at least a "current" float (NaN if fetch failed).
Rules referencing missing indicators are silently skipped.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from config.settings import SECTOR_ETFS


MacroPayload = dict[str, Any]


def _value(readings: dict[str, MacroPayload], key: str,
           field: str = "current") -> float | None:
    """Pull `field` from the payload for `key`, or None if missing / NaN."""
    payload = readings.get(key)
    if payload is None:
        return None
    val = payload.get(field)
    if val is None:
        return None
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(fval):
        return None
    return fval


Rule = tuple[str, Callable[[float], bool], str, str] | \
       tuple[str, Callable[[float], bool], str, str, str]


def _gt(threshold: float) -> Callable[[float], bool]:
    return lambda v: v > threshold


def _lt(threshold: float) -> Callable[[float], bool]:
    return lambda v: v < threshold


# ---------------------------------------------------------------------------
# Sector macro rule map
# ---------------------------------------------------------------------------
# Each rule is: (indicator_key, predicate, verdict, label [, field])
# Default field is "current"; use "z_score_1y" or "slope_30d" where noted.
# verdict: "tailwind" | "headwind" | "neutral"
# ---------------------------------------------------------------------------

SECTOR_MACRO_MAP: dict[str, list[Rule]] = {

    # -----------------------------------------------------------------------
    # XLF — Financials
    # NIM-driven earnings need: steep curve, tight credit, strong labor.
    # -----------------------------------------------------------------------
    "XLF": [
        # Yield curve
        ("T10Y2Y",   _gt(0.5),  "tailwind",  "T10Y2Y > +0.5% (steep curve lifts NIM)"),
        ("T10Y2Y",   _lt(0.0),  "headwind",  "T10Y2Y inverted (NIM pressure)"),
        # Absolute rate level
        ("UST10",    _gt(4.0),  "tailwind",  "10Y > 4% (higher rates lift NIM)"),
        ("UST10",    _lt(3.0),  "headwind",  "10Y < 3% (low rates squeeze NIM)"),
        # Short end (Fed signal)
        ("UST2",     _gt(4.5),  "headwind",  "2Y > 4.5% (market pricing prolonged tightening)"),
        ("UST2",     _lt(3.5),  "tailwind",  "2Y < 3.5% (market pricing cuts → NIM re-expansion)"),
        # Credit spreads
        ("HY_OAS",   _lt(4.0),  "tailwind",  "HY OAS < 4% (credit benign → loan demand healthy)"),
        ("HY_OAS",   _gt(5.0),  "headwind",  "HY OAS > 5% (credit stress → loan losses rising)"),
        ("IG_OAS",   _lt(1.5),  "tailwind",  "BBB OAS < 1.5% (IG credit healthy → bank lending margins good)"),
        ("IG_OAS",   _gt(2.0),  "headwind",  "BBB OAS > 2.0% (IG stress, early warning of loan losses)"),
        # Financial conditions
        ("FIN_CONDITIONS", _lt(-0.3), "tailwind", "NFCI < -0.3 (loose conditions → credit demand up)"),
        ("FIN_CONDITIONS", _gt(0.5),  "headwind",  "NFCI > 0.5 (tight conditions → credit contraction)"),
        # Labor market (loan demand + default risk)
        ("INIT_CLAIMS", _lt(-0.5), "tailwind", "Claims z < -0.5 (tight labor → consumer loan demand strong)", "z_score_1y"),
        ("INIT_CLAIMS", _gt(1.0),  "headwind",  "Claims z > +1 (rising unemployment → default risk up)", "z_score_1y"),
    ],

    # -----------------------------------------------------------------------
    # XLE — Energy
    # Earnings lever: weak USD, oil-rich Gold/Oil, inflation expectations.
    # -----------------------------------------------------------------------
    "XLE": [
        # Dollar
        ("DXY",            _lt(100.0), "tailwind", "DXY < 100 (weak USD lifts crude prices)"),
        ("DXY",            _gt(105.0), "headwind",  "DXY > 105 (strong USD weighs on crude)"),
        # Gold/Oil ratio (inverse energy-cycle indicator)
        ("GOLD_OIL",       _lt(15.0),  "tailwind",  "Gold/Oil < 15 (oil rich vs gold → strong energy cycle)"),
        ("GOLD_OIL",       _gt(30.0),  "headwind",  "Gold/Oil > 30 (oil cheap vs gold → demand destruction)"),
        # Growth/reflation
        ("COPPER_GOLD",    _gt(0.5),   "tailwind",  "Copper/Gold z > +0.5 (reflation → energy demand)", "z_score_1y"),
        # Inflation expectations (demand for energy)
        ("BREAKEVEN_5Y5Y", _gt(2.5),   "tailwind",  "5Y5Y breakeven > 2.5% (inflation supports commodities)"),
        ("BREAKEVEN_10Y",  _gt(2.5),   "tailwind",  "10Y breakeven > 2.5% (elevated inflation → energy bid)"),
        ("BREAKEVEN_10Y",  _lt(1.8),   "headwind",  "10Y breakeven < 1.8% (deflationary fears → demand collapse)"),
        # Labor / economic activity
        ("INIT_CLAIMS",    _lt(-0.5),  "tailwind",  "Claims z < -0.5 (tight labor → industrial energy demand)", "z_score_1y"),
        ("INIT_CLAIMS",    _gt(1.5),   "headwind",  "Claims z > +1.5 (recession → demand destruction)", "z_score_1y"),
    ],

    # -----------------------------------------------------------------------
    # XLK — Technology
    # Long-duration growth; rate-sensitive, risk-sentiment sensitive.
    # -----------------------------------------------------------------------
    "XLK": [
        # Real rates (discount rate for growth cashflows)
        ("REAL_10Y", _gt(2.0),   "headwind",  "Real 10Y > 2% (high real rates compress growth multiples)"),
        ("REAL_10Y", _lt(1.0),   "tailwind",  "Real 10Y < 1% (low real rates support growth multiples)"),
        # Inflation expectations (drives real rates)
        ("BREAKEVEN_10Y", _lt(2.5), "tailwind", "10Y breakeven < 2.5% (stable inflation → real yields contained)"),
        ("BREAKEVEN_10Y", _gt(3.0), "headwind",  "10Y breakeven > 3% (unanchored inflation → Fed overtightens)"),
        # Dollar (overseas revenue)
        ("DXY",      _gt(105.0), "headwind",  "DXY > 105 (offshore revenue FX drag for US tech)"),
        # Risk sentiment
        ("VIX",      _gt(25.0),  "headwind",  "VIX > 25 (risk-off de-rates growth names)"),
        # Credit conditions (fund growth, share buybacks)
        ("HY_OAS",   _lt(4.0),   "tailwind",  "HY OAS < 4% (risk-on credit funds growth capex)"),
        ("HY_OAS",   _gt(5.0),   "headwind",  "HY OAS > 5% (risk-off credit → growth spending cut)"),
        ("IG_OAS",   _lt(1.5),   "tailwind",  "BBB OAS < 1.5% (IG accessible → tech capex/buybacks)"),
        ("IG_OAS",   _gt(2.0),   "headwind",  "BBB OAS > 2.0% (IG stress → discretionary tech spending cut)"),
        # Financial conditions (aggregate)
        ("FIN_CONDITIONS", _lt(-0.3), "tailwind", "NFCI < -0.3 (loose → growth multiples expand)"),
        ("FIN_CONDITIONS", _gt(0.5),  "headwind",  "NFCI > 0.5 (tight → de-rate growth names)"),
    ],

    # -----------------------------------------------------------------------
    # XLC — Communication Services
    # Similar to XLK: long-duration ad-spend sensitive.
    # -----------------------------------------------------------------------
    "XLC": [
        ("REAL_10Y", _gt(2.0),   "headwind",  "Real 10Y > 2% (duration discount)"),
        ("REAL_10Y", _lt(1.0),   "tailwind",  "Real 10Y < 1% (duration support)"),
        ("VIX",      _gt(25.0),  "headwind",  "VIX > 25 (risk-off)"),
        ("HY_OAS",   _lt(4.0),   "tailwind",  "HY OAS < 4% (risk-on credit)"),
        ("HY_OAS",   _gt(5.0),   "headwind",  "HY OAS > 5% (risk-off)"),
        ("DXY",      _gt(105.0), "headwind",  "DXY > 105 (offshore ad-revenue FX drag)"),
        # Ad spend sensitive to financial conditions and employment
        ("FIN_CONDITIONS", _lt(-0.3), "tailwind", "NFCI < -0.3 (loose → ad budgets expand)"),
        ("FIN_CONDITIONS", _gt(0.5),  "headwind",  "NFCI > 0.5 (tight → ad budgets cut first)"),
        ("INIT_CLAIMS",    _gt(0.5),  "headwind",  "Claims z > +0.5 (consumer stress → streaming churn up, ad spend down)", "z_score_1y"),
    ],

    # -----------------------------------------------------------------------
    # XLY — Consumer Discretionary
    # Cyclical; housing cycle + employment + consumer credit.
    # -----------------------------------------------------------------------
    "XLY": [
        # Credit spreads / risk
        ("HY_OAS",   _lt(4.0),  "tailwind",  "HY OAS < 4% (consumer credit OK → big-ticket demand)"),
        ("HY_OAS",   _gt(5.0),  "headwind",  "HY OAS > 5% (consumer stress → spend pullback)"),
        # Risk sentiment
        ("VIX",      _gt(25.0), "headwind",  "VIX > 25 (cyclical de-rate)"),
        # Curve (late-cycle signal)
        ("T10Y2Y",   _lt(0.0),  "headwind",  "Curve inverted (late-cycle consumer caution)"),
        # Real rates (financing cost on big-ticket)
        ("REAL_10Y", _gt(2.0),  "headwind",  "Real 10Y > 2% (financing drag on big-ticket purchases)"),
        # Mortgage rate (housing wealth effect + homebuilder earnings)
        ("MORTGAGE_30Y", _lt(6.5), "tailwind", "Mortgage 30Y < 6.5% (affordable → housing demand + wealth effect)"),
        ("MORTGAGE_30Y", _gt(7.5), "headwind",  "Mortgage 30Y > 7.5% (housing locked → wealth effect negative)"),
        # Employment (core driver of consumer spend)
        ("INIT_CLAIMS", _lt(-0.5), "tailwind", "Claims z < -0.5 (tight labor → employed consumers spending)", "z_score_1y"),
        ("INIT_CLAIMS", _gt(1.0),  "headwind",  "Claims z > +1 (job losses → consumer pullback)", "z_score_1y"),
        # Financial conditions (auto loans, credit card rates)
        ("FIN_CONDITIONS", _lt(-0.3), "tailwind", "NFCI < -0.3 (loose credit → big-ticket financing available)"),
        ("FIN_CONDITIONS", _gt(0.5),  "headwind",  "NFCI > 0.5 (tight → auto loans and credit cards expensive)"),
    ],

    # -----------------------------------------------------------------------
    # XLI — Industrials
    # Global cyclicals; capex cycle, labour, reflationary demand.
    # -----------------------------------------------------------------------
    "XLI": [
        ("COPPER_GOLD",    _gt(0.5),   "tailwind",  "Copper/Gold z > +0.5 (pro-growth reflation)", "z_score_1y"),
        ("COPPER_GOLD",    _lt(-0.5),  "headwind",  "Copper/Gold z < -0.5 (deflationary impulse)", "z_score_1y"),
        ("DXY",            _gt(105.0), "headwind",  "DXY > 105 (export drag on US industrials)"),
        ("HY_OAS",         _gt(5.0),   "headwind",  "HY OAS > 5% (capex financing risk)"),
        ("T10Y2Y",         _lt(0.0),   "headwind",  "Curve inverted (late-cycle capex caution)"),
        ("BREAKEVEN_5Y5Y", _gt(2.5),   "tailwind",  "5Y5Y breakeven > 2.5% (reflationary demand for industrials)"),
        # Labor market
        ("INIT_CLAIMS", _lt(-0.5), "tailwind", "Claims z < -0.5 (tight labor → industrial order books strong)", "z_score_1y"),
        ("INIT_CLAIMS", _gt(1.0),  "headwind",  "Claims z > +1 (labor weakness → capex delays and order cuts)", "z_score_1y"),
        # Financial conditions (capex financing)
        ("FIN_CONDITIONS", _lt(-0.3), "tailwind", "NFCI < -0.3 (loose → capex financing cheap)"),
        ("FIN_CONDITIONS", _gt(0.5),  "headwind",  "NFCI > 0.5 (tight → capex freeze)"),
        # Mortgage / construction (a large industrials sub-sector)
        ("MORTGAGE_30Y", _gt(7.5), "headwind", "Mortgage 30Y > 7.5% (construction slowdown → less industrial activity)"),
    ],

    # -----------------------------------------------------------------------
    # XLB — Materials
    # Global cyclicals; reflation + weak USD are the key drivers.
    # -----------------------------------------------------------------------
    "XLB": [
        ("COPPER_GOLD",    _gt(0.5),   "tailwind",  "Copper/Gold z > +0.5 (pro-growth reflation)", "z_score_1y"),
        ("COPPER_GOLD",    _lt(-0.5),  "headwind",  "Copper/Gold z < -0.5 (deflationary)", "z_score_1y"),
        ("DXY",            _lt(100.0), "tailwind",  "DXY < 100 (weak USD → commodity tailwind)"),
        ("DXY",            _gt(105.0), "headwind",  "DXY > 105 (strong USD → commodity headwind)"),
        ("BREAKEVEN_5Y5Y", _gt(2.5),   "tailwind",  "5Y5Y breakeven > 2.5% (inflation lifts commodity prices)"),
        # 10Y breakeven (spot inflation expectations)
        ("BREAKEVEN_10Y",  _gt(2.0),   "tailwind",  "10Y breakeven > 2.0% (inflation → commodity prices supported)"),
        ("BREAKEVEN_10Y",  _lt(1.8),   "headwind",  "10Y breakeven < 1.8% (deflation fears → commodity demand collapses)"),
        # Labor / activity
        ("INIT_CLAIMS", _lt(-0.5), "tailwind", "Claims z < -0.5 (strong industrial activity → materials demand)", "z_score_1y"),
        ("INIT_CLAIMS", _gt(1.0),  "headwind",  "Claims z > +1 (economic slowdown → commodity demand falls)", "z_score_1y"),
    ],

    # -----------------------------------------------------------------------
    # XLV — Health Care
    # Defensive bond-proxy; bid in stress, rotation target in risk-on.
    # -----------------------------------------------------------------------
    "XLV": [
        ("VIX",    _gt(25.0),  "tailwind",  "VIX > 25 (risk-off → defensive healthcare bid)"),
        ("VIX",    _lt(15.0),  "headwind",  "VIX < 15 (risk-on → rotation out of defensives)"),
        ("HY_OAS", _gt(5.0),   "tailwind",  "HY OAS > 5% (rotate to defensives)"),
        ("DXY",    _gt(105.0), "headwind",  "DXY > 105 (big-pharma offshore-revenue FX drag)"),
        # Financial conditions / risk appetite
        ("FIN_CONDITIONS", _gt(0.5),  "tailwind",  "NFCI > 0.5 (tight conditions → defensive bid)"),
        ("FIN_CONDITIONS", _lt(-0.5), "headwind",  "NFCI < -0.5 (loose → risk-on, rotate out of defensives)"),
        # Labor (recession fear indicator)
        ("INIT_CLAIMS", _gt(1.0), "tailwind",  "Claims z > +1 (recession fear → defensive healthcare bid)", "z_score_1y"),
        ("INIT_CLAIMS", _lt(-0.5),"headwind",  "Claims z < -0.5 (boom → risk-on rotation out)", "z_score_1y"),
    ],

    # -----------------------------------------------------------------------
    # XLP — Consumer Staples
    # Defensive bond-proxy; risk-on and strong USD weigh.
    # -----------------------------------------------------------------------
    "XLP": [
        ("VIX",      _gt(25.0),  "tailwind",  "VIX > 25 (defensive bid)"),
        ("VIX",      _lt(15.0),  "headwind",  "VIX < 15 (risk-on → rotation out of staples)"),
        ("REAL_10Y", _lt(1.0),   "tailwind",  "Real 10Y < 1% (bond-proxy supportive)"),
        ("REAL_10Y", _gt(2.0),   "headwind",  "Real 10Y > 2% (bond-proxy hurt by rising real rates)"),
        ("DXY",      _gt(105.0), "headwind",  "DXY > 105 (staples multinational FX drag)"),
        # Financial conditions / recession fear
        ("FIN_CONDITIONS", _gt(0.5),  "tailwind",  "NFCI > 0.5 (tight → defensive staples bid)"),
        ("FIN_CONDITIONS", _lt(-0.5), "headwind",  "NFCI < -0.5 (loose → rotate out of defensives)"),
        # Labor (recession signal)
        ("INIT_CLAIMS", _gt(1.0),  "tailwind",  "Claims z > +1 (recession fear → staples bid)", "z_score_1y"),
        ("INIT_CLAIMS", _lt(-0.5), "headwind",  "Claims z < -0.5 (boom → rotate out of defensives)", "z_score_1y"),
    ],

    # -----------------------------------------------------------------------
    # XLU — Utilities
    # Bond proxy; hurt by high rates, tight financial conditions = tailwind.
    # -----------------------------------------------------------------------
    "XLU": [
        ("REAL_10Y", _lt(1.0),  "tailwind",  "Real 10Y < 1% (bond-proxy supportive)"),
        ("REAL_10Y", _gt(2.0),  "headwind",  "Real 10Y > 2% (bond-proxy hurt)"),
        ("T10Y2Y",   _lt(0.0),  "neutral",   "Curve inverted (rate-cut path ahead; mixed for utilities)"),
        ("VIX",      _gt(25.0), "tailwind",  "VIX > 25 (defensive bid)"),
        ("VIX",      _lt(15.0), "headwind",  "VIX < 15 (risk-on → rotate out of defensives)"),
        ("UST10",    _gt(5.0),  "headwind",  "10Y > 5% (bond-proxy competition from Treasuries)"),
        # Mortgage rate (utilities compete with housing-linked yield instruments)
        ("MORTGAGE_30Y", _gt(7.0), "headwind", "Mortgage 30Y > 7% (high rates increase utility debt servicing cost)"),
        # Financial conditions
        ("FIN_CONDITIONS", _gt(0.5),  "tailwind",  "NFCI > 0.5 (tight → defensive utility bid)"),
        ("FIN_CONDITIONS", _lt(-0.5), "headwind",  "NFCI < -0.5 (loose → rotate out of defensives)"),
        # IG credit (utilities are heavy debt issuers)
        ("IG_OAS",   _gt(2.0),  "headwind",  "BBB OAS > 2.0% (utility debt refinancing more expensive)"),
        # Labor
        ("INIT_CLAIMS", _gt(1.0),  "tailwind",  "Claims z > +1 (recession fear → defensive bid)", "z_score_1y"),
        ("INIT_CLAIMS", _lt(-0.5), "headwind",  "Claims z < -0.5 (boom → rotate out of defensives)", "z_score_1y"),
    ],

    # -----------------------------------------------------------------------
    # XLRE — Real Estate
    # Rate sensitive; mortgage rates + financial conditions + credit spreads.
    # -----------------------------------------------------------------------
    "XLRE": [
        ("REAL_10Y", _lt(1.0),  "tailwind",  "Real 10Y < 1% (cap-rate relief for REITs)"),
        ("REAL_10Y", _gt(2.0),  "headwind",  "Real 10Y > 2% (cap-rate pressure)"),
        ("T10Y2Y",   _lt(0.0),  "neutral",   "Curve inverted (rate-cut path; mixed for REITs)"),
        ("HY_OAS",   _gt(5.0),  "headwind",  "HY OAS > 5% (REIT refinancing stress)"),
        ("UST10",    _gt(5.0),  "headwind",  "10Y > 5% (mortgage / cap-rate pressure)"),
        # Mortgage rate (direct housing demand signal)
        ("MORTGAGE_30Y", _lt(6.5), "tailwind", "Mortgage 30Y < 6.5% (affordable → property demand up)"),
        ("MORTGAGE_30Y", _gt(7.5), "headwind",  "Mortgage 30Y > 7.5% (high mortgages → property prices under pressure)"),
        # Mortgage spread (housing credit vs general market)
        ("MORTGAGE_SPREAD", _gt(2.5), "headwind", "Mortgage spread > 2.5% (housing credit tight beyond general rates)"),
        # Financial conditions
        ("FIN_CONDITIONS", _lt(-0.3), "tailwind",  "NFCI < -0.3 (loose → commercial RE financing cheap)"),
        ("FIN_CONDITIONS", _gt(0.5),  "headwind",  "NFCI > 0.5 (tight → REIT refi stress + cap-rate pressure)"),
        # IG credit (REITs are heavy IG bond issuers)
        ("IG_OAS",   _gt(2.0),  "headwind",  "BBB OAS > 2.0% (REIT debt refinancing significantly more expensive)"),
        # Labor (commercial vacancy rates)
        ("INIT_CLAIMS", _gt(1.0), "headwind", "Claims z > +1 (recession → commercial vacancy rates rise)", "z_score_1y"),
    ],

    # -----------------------------------------------------------------------
    # UFO — Space (supplementary thematic)
    # High-beta speculative long-duration; needs risk-on + loose conditions.
    # -----------------------------------------------------------------------
    "UFO": [
        ("VIX",      _lt(15.0), "tailwind",  "VIX < 15 (risk-on supports thematics)"),
        ("VIX",      _gt(25.0), "headwind",  "VIX > 25 (risk-off crushes speculative positions)"),
        ("HY_OAS",   _lt(4.0),  "tailwind",  "HY OAS < 4% (risk-on credit)"),
        ("HY_OAS",   _gt(5.0),  "headwind",  "HY OAS > 5% (risk-off credit)"),
        ("REAL_10Y", _gt(2.0),  "headwind",  "Real 10Y > 2% (crushes long-duration speculative growth)"),
        # Financial conditions
        ("FIN_CONDITIONS", _lt(-0.3), "tailwind",  "NFCI < -0.3 (loose → risk-on, speculative growth bid)"),
        ("FIN_CONDITIONS", _gt(0.5),  "headwind",  "NFCI > 0.5 (tight → speculative long-duration unwind)"),
        # IG credit (risk appetite proxy)
        ("IG_OAS",   _lt(1.5),  "tailwind",  "BBB OAS < 1.5% (IG risk-on → space ETFs benefit)"),
        ("IG_OAS",   _gt(2.0),  "headwind",  "BBB OAS > 2.0% (risk-off → speculative position unwind)"),
        # Labor (recession fear = reduce speculative exposure)
        ("INIT_CLAIMS", _gt(1.0), "headwind", "Claims z > +1 (recession fear → sell speculative positions)", "z_score_1y"),
    ],
}


def compute_macro_alignment(
    macro_readings: dict[str, MacroPayload],
) -> pd.DataFrame:
    """Per-sector macro alignment.

    Returns DataFrame indexed by sector ticker with columns:
      tailwinds: int
      headwinds: int
      neutral:   int
      ratio:     float    # tailwinds / (tailwinds + headwinds); 0 if both 0
      detail:    list[tuple[str, str]]   # [(label, verdict), ...]
    """
    rows: list[dict[str, Any]] = []
    for sector in SECTOR_ETFS.keys():
        rules = SECTOR_MACRO_MAP.get(sector, [])
        tailwinds = headwinds = neutral = 0
        detail: list[tuple[str, str]] = []
        for rule in rules:
            indicator_key, predicate, verdict, label = rule[:4]
            field = rule[4] if len(rule) > 4 else "current"
            val = _value(macro_readings, indicator_key, field)
            if val is None:
                continue
            if not predicate(val):
                continue
            if verdict == "tailwind":
                tailwinds += 1
            elif verdict == "headwind":
                headwinds += 1
            elif verdict == "neutral":
                neutral += 1
            else:
                continue
            detail.append((label, verdict))

        denom = tailwinds + headwinds
        ratio = (tailwinds / denom) if denom > 0 else 0.0
        rows.append({
            "sector": sector,
            "tailwinds": tailwinds,
            "headwinds": headwinds,
            "neutral": neutral,
            "ratio": float(ratio),
            "detail": detail,
        })

    return pd.DataFrame(rows).set_index("sector")
