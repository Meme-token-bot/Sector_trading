"""Regime + breadth snapshot for the Dashboard's top-of-tab decision strip.

Pure functions, no IO, no Streamlit — same convention as `regime_analysis.py`
and `macro_alignment.py`. Answers the single most-skipped question in the old
layout: "should I even lean into rotation this week, or should risk be cut
across the board regardless of what any individual sector's state says?"

Two independent reads, combined into one snapshot:
  * regime  — SPY drawdown-from-high band, reusing
              `regime_analysis.classify_regimes` (BULL / CORRECTION / BEAR),
              plus how many consecutive days it's held.
  * breadth — cross-sectional participation: how many CORE sectors (excluding
              `SUPPLEMENTARY_SECTORS`, matching the RS-rank convention in
              `src/signals.py::build_signals`) are above their own SMA200, and
              how dispersed relative strength is across the universe. Low
              dispersion is exactly the regime a cross-sectional rotation
              strategy structurally struggles in; this is the cheapest
              available proxy for that until a real correlation matrix exists
              (see TRADING_EDGE_AUDIT.md §7 — this module deliberately does
              NOT try to be that; it's the Phase-0 stopgap, not the Phase-2
              risk block).
"""
from __future__ import annotations

import pandas as pd

from config.settings import SUPPLEMENTARY_SECTORS
from src.regime_analysis import classify_regimes


def _regime_run_length(regimes: pd.Series) -> int:
    """Consecutive days the CURRENT regime label has held, walking backward
    from the last observation. 0 if `regimes` is empty."""
    if regimes.empty:
        return 0
    current = regimes.iloc[-1]
    rev = regimes.iloc[::-1]
    n = 0
    for v in rev:
        if v == current:
            n += 1
        else:
            break
    return n


def compute_regime_and_breadth(
    spy_close: pd.Series,
    metrics: pd.DataFrame,
) -> dict:
    """One-call snapshot for the Dashboard's top strip.

    Parameters
    ----------
    spy_close : SPY daily close series (DatetimeIndex), any length — longer
                history gives a more meaningful `regime_days` count. On the
                live dashboard, pass the same SPY column already loaded for
                `compute_sector_metrics` (`_cached_prices()[BENCHMARK]`) —
                no new price fetch needed.
    metrics   : the frame returned by `compute_sector_metrics(prices)` — must
                carry `above_sma` and `relative_strength_3m` columns indexed
                by ticker.

    Returns
    -------
    {
      "regime": "BULL" | "CORRECTION" | "BEAR" | "—",
      "regime_days": int,
      "n_above_sma": int,
      "n_core": int,
      "pct_above_sma": float,          # 0..1, 0.0 if n_core == 0
      "rs_dispersion_pct": float,      # stdev of relative_strength_3m across
                                        # core sectors, in percentage points
                                        # (e.g. 4.2 == 4.2pp), NaN if unknown
      "rs_mean_pct": float,            # same units
    }

    Restricted to core (non-supplementary) sectors throughout, matching the
    RS-rank convention already established in `src/signals.py::build_signals`
    — UFO shouldn't dilute a read of "how is the core 11-sector universe
    behaving" any more than it should dilute the bottom-3 SELL-rank pool.
    """
    spy_close = spy_close.dropna() if spy_close is not None else pd.Series(dtype=float)
    if spy_close.empty:
        regime, regime_days = "—", 0
    else:
        regimes = classify_regimes(spy_close)
        regime = str(regimes.iloc[-1]) if not regimes.empty else "—"
        regime_days = _regime_run_length(regimes)

    if metrics is None or metrics.empty:
        return {
            "regime": regime, "regime_days": regime_days,
            "n_above_sma": 0, "n_core": 0, "pct_above_sma": 0.0,
            "rs_dispersion_pct": float("nan"), "rs_mean_pct": float("nan"),
        }

    core_idx = [t for t in metrics.index if t not in SUPPLEMENTARY_SECTORS]
    core = metrics.loc[core_idx]
    n_core = int(len(core))
    n_above = int(core["above_sma"].sum()) if "above_sma" in core.columns and n_core else 0
    pct_above = (n_above / n_core) if n_core else 0.0

    if "relative_strength_3m" in core.columns and n_core:
        rs = core["relative_strength_3m"].astype(float)
        rs_dispersion = float(rs.std(ddof=0) * 100.0)
        rs_mean = float(rs.mean() * 100.0)
    else:
        rs_dispersion = float("nan")
        rs_mean = float("nan")

    return {
        "regime": regime,
        "regime_days": regime_days,
        "n_above_sma": n_above,
        "n_core": n_core,
        "pct_above_sma": pct_above,
        "rs_dispersion_pct": rs_dispersion,
        "rs_mean_pct": rs_mean,
    }


# ---------------------------------------------------------------------------
# Dispersion regime read — is the current dispersion level rotation-friendly?
# ---------------------------------------------------------------------------
# A cross-sectional rotation strategy needs sectors to actually disperse; in
# an "everything moves together" tape (dispersion near zero) equal-weighting
# whatever fires BUY buys the same bet several times over. These bands are a
# starting heuristic, NOT a fitted threshold — recalibrate once enough
# `signal_snapshots` history exists to check realized dispersion against
# forward hit-rate (see TRADING_EDGE_AUDIT.md §6, the same "start the
# measurement clock now" logic already applied to sentiment).
DISPERSION_BANDS: list[tuple[float, str, str]] = [
    (2.0, "🔴", "Low — sectors moving together; rotation has little to work with"),
    (5.0, "🟡", "Moderate"),
    (float("inf"), "🟢", "High — rotation-friendly dispersion"),
]


def dispersion_band(rs_dispersion_pct: float) -> tuple[str, str]:
    """(emoji, label) for a dispersion reading. NaN-safe."""
    if rs_dispersion_pct is None or rs_dispersion_pct != rs_dispersion_pct:  # NaN check
        return "⚪", "—"
    for threshold, emoji, label in DISPERSION_BANDS:
        if rs_dispersion_pct < threshold:
            return emoji, label
    return "⚪", "—"
