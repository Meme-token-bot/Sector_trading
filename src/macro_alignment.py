"""Per-sector macro alignment.

Maps the macro indicator panel onto each sector's structural sensitivities and
counts how many readings are currently a tailwind, a headwind, or neutral.

This module is pure: given a dict of macro readings (the same payloads the
Macro tab consumes from `src.market_engine`), it returns a tidy frame and a
detail trace per sector. Streamlit-free, network-free.

The keys in `macro_readings` are logical indicator names — the canonical set
the rest of the codebase already uses:

    "T10Y2Y"          -> yield_curve_spread()       (current = curve in %)
    "HY_OAS"          -> fred["HY_OAS"]             (current = spread in %)
    "UST10"           -> fred["UST10"]              (current = nominal 10Y, %)
    "REAL_10Y"        -> fred["REAL_10Y"]           (current = real 10Y, %)
    "BREAKEVEN_5Y5Y"  -> fred["BREAKEVEN_5Y5Y"]
    "DXY"             -> dxy_level()                (current = DXY index level)
    "VIX"             -> vix_level()                (current = VIX level)
    "GOLD_OIL"        -> gold_oil_ratio()           (current = gold/oil ratio)
    "COPPER_GOLD"     -> copper_gold_ratio()        (current = copper/gold ratio)

Each payload is a `dict` with at least a `"current"` float (which may be
`NaN` if upstream fetching failed). Optional keys (`z_score_1y`,
`slope_30d`, `series`, `error`) are ignored by this module.

Callers may pass a subset; rules referencing missing indicators are silently
skipped. A sector with zero applicable readings reports ratio=0.0 and all
counts at zero.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from config.settings import SECTOR_ETFS


# A macro payload is a dict like {"current": float, ...}. We treat it
# structurally rather than locking to a NewType so existing callers
# (market_engine helpers) work unchanged.
MacroPayload = dict[str, Any]


def _value(readings: dict[str, MacroPayload], key: str) -> float | None:
    """Pull the `current` reading for `key`, or None if missing / NaN."""
    payload = readings.get(key)
    if payload is None:
        return None
    val = payload.get("current")
    if val is None:
        return None
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(fval):
        return None
    return fval


# A rule is (indicator_key, predicate_on_value, verdict, label_for_detail).
# predicate returns True iff the indicator's `current` reading currently
# applies. `verdict` is one of {"tailwind", "headwind", "neutral"}.
Rule = tuple[str, Callable[[float], bool], str, str]


def _gt(threshold: float) -> Callable[[float], bool]:
    return lambda v: v > threshold


def _lt(threshold: float) -> Callable[[float], bool]:
    return lambda v: v < threshold


# Rule mapping per sector. The mapping covers every ticker in
# config.SECTOR_ETFS. Rules deliberately favor *direction* over magnitude;
# the magnitudes mirror the bands used elsewhere in the dashboard (the
# Macro tab's _VIX_BANDS, _HY_OAS_BANDS, etc).
SECTOR_MACRO_MAP: dict[str, list[Rule]] = {
    # Financials — yield curve steepening + benign credit help NIMs.
    "XLF": [
        ("T10Y2Y", _gt(0.5),  "tailwind",  "T10Y2Y > +0.5 (steep curve helps NIM)"),
        ("T10Y2Y", _lt(0.0),  "headwind",  "T10Y2Y inverted (NIM pressure)"),
        ("HY_OAS", _lt(4.0),  "tailwind",  "HY OAS < 4% (credit benign)"),
        ("HY_OAS", _gt(5.0),  "headwind",  "HY OAS > 5% (credit stress)"),
    ],
    # Energy — weak USD + oil-rich gold/oil ratio = supportive.
    "XLE": [
        ("DXY",      _lt(100.0), "tailwind", "DXY < 100 (weak USD lifts crude)"),
        ("DXY",      _gt(105.0), "headwind", "DXY > 105 (strong USD weighs on crude)"),
        ("GOLD_OIL", _lt(15.0),  "tailwind", "Gold/Oil < 15 (oil rich vs gold)"),
        ("GOLD_OIL", _gt(30.0),  "headwind", "Gold/Oil > 30 (oil cheap vs gold)"),
    ],
    # Technology — long-duration cashflow assets; sensitive to real rates,
    # USD, and risk sentiment.
    "XLK": [
        ("REAL_10Y", _gt(2.0),   "headwind", "Real 10Y > 2% (duration discount)"),
        ("REAL_10Y", _lt(1.0),   "tailwind", "Real 10Y < 1% (duration support)"),
        ("DXY",      _gt(105.0), "headwind", "DXY > 105 (offshore-rev FX drag)"),
        ("VIX",      _gt(25.0),  "headwind", "VIX > 25 (risk-off de-rates growth)"),
    ],
    # Communication Services — mega-cap growth profile similar to XLK.
    "XLC": [
        ("REAL_10Y", _gt(2.0),   "headwind", "Real 10Y > 2% (duration discount)"),
        ("REAL_10Y", _lt(1.0),   "tailwind", "Real 10Y < 1% (duration support)"),
        ("VIX",      _gt(25.0),  "headwind", "VIX > 25 (risk-off)"),
    ],
    # Consumer Discretionary — cyclical, sensitive to credit & vol regime.
    "XLY": [
        ("HY_OAS", _lt(4.0), "tailwind", "HY OAS < 4% (consumer credit OK)"),
        ("HY_OAS", _gt(5.0), "headwind", "HY OAS > 5% (consumer stress)"),
        ("VIX",    _gt(25.0), "headwind", "VIX > 25 (cyclical de-rate)"),
        ("T10Y2Y", _lt(0.0),  "headwind", "Curve inverted (late cycle)"),
    ],
    # Industrials — global cyclicals tied to growth proxy & USD.
    "XLI": [
        ("COPPER_GOLD", _gt(0.20), "tailwind", "Copper/Gold high (pro-growth)"),
        ("DXY",          _gt(105.0), "headwind", "DXY > 105 (export drag)"),
        ("HY_OAS",       _gt(5.0),   "headwind", "HY OAS > 5% (capex risk)"),
    ],
    # Materials — global cyclicals; same drivers as industrials.
    "XLB": [
        ("COPPER_GOLD", _gt(0.20), "tailwind", "Copper/Gold high (pro-growth)"),
        ("DXY",          _lt(100.0), "tailwind", "DXY < 100 (commodity tailwind)"),
        ("DXY",          _gt(105.0), "headwind", "DXY > 105 (commodity headwind)"),
    ],
    # Health Care — defensive; rises with stress and risk-off.
    "XLV": [
        ("VIX",    _gt(25.0), "tailwind", "VIX > 25 (defensive bid)"),
        ("HY_OAS", _gt(5.0),  "tailwind", "HY OAS > 5% (rotate to defensives)"),
    ],
    # Consumer Staples — defensive; benefit from rate-cut expectations & risk-off.
    "XLP": [
        ("VIX",      _gt(25.0), "tailwind", "VIX > 25 (defensive bid)"),
        ("REAL_10Y", _lt(1.0),  "tailwind", "Real 10Y < 1% (bond-proxy supportive)"),
        ("REAL_10Y", _gt(2.0),  "headwind", "Real 10Y > 2% (bond-proxy hurts)"),
    ],
    # Utilities — bond proxy.
    "XLU": [
        ("REAL_10Y", _lt(1.0),  "tailwind", "Real 10Y < 1% (bond-proxy supportive)"),
        ("REAL_10Y", _gt(2.0),  "headwind", "Real 10Y > 2% (bond-proxy hurts)"),
        ("T10Y2Y",   _lt(0.0),  "neutral",  "Curve inverted (rate-cut path)"),
        ("VIX",      _gt(25.0), "tailwind", "VIX > 25 (defensive bid)"),
    ],
    # Real Estate — rate sensitive, bond proxy.
    "XLRE": [
        ("REAL_10Y", _lt(1.0),  "tailwind", "Real 10Y < 1% (cap-rate relief)"),
        ("REAL_10Y", _gt(2.0),  "headwind", "Real 10Y > 2% (cap-rate pressure)"),
        ("T10Y2Y",   _lt(0.0),  "neutral",  "Curve inverted (rate-cut path)"),
        ("HY_OAS",   _gt(5.0),  "headwind", "HY OAS > 5% (refi stress)"),
    ],
    # Space (supplementary thematic) — high-beta speculative; needs risk-on.
    "UFO": [
        ("VIX",    _lt(15.0), "tailwind", "VIX < 15 (risk-on supports thematics)"),
        ("VIX",    _gt(25.0), "headwind", "VIX > 25 (risk-off crushes thematics)"),
        ("HY_OAS", _lt(4.0),  "tailwind", "HY OAS < 4% (risk-on credit)"),
        ("HY_OAS", _gt(5.0),  "headwind", "HY OAS > 5% (risk-off credit)"),
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
      detail:    list[tuple[str, str]]   # [(indicator, verdict), ...]

    Sectors with no applicable rules (or where every relevant reading is
    missing) report all counts at zero and ratio 0.0.
    """
    rows: list[dict[str, Any]] = []
    for sector in SECTOR_ETFS.keys():
        rules = SECTOR_MACRO_MAP.get(sector, [])
        tailwinds = headwinds = neutral = 0
        detail: list[tuple[str, str]] = []
        for indicator_key, predicate, verdict, label in rules:
            val = _value(macro_readings, indicator_key)
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
                # Unknown verdict — skip silently; we don't want to crash
                # the dashboard over a rule typo.
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
