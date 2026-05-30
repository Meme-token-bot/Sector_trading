"""Regime-conditional and drawdown-attribution analysis for the backtest.

Why this exists
---------------
The headline "+11.48% CAGR vs SPY +17.99%" averages across regimes. A
rotation strategy is structurally defensive — it gives up upside in steady
bull markets to (hopefully) preserve capital in drawdowns. Averaging
across both regimes hides the question that actually matters for the user
("did the rotation work when the market broke?").

This module answers two questions explicitly:

1. **Regime-conditional**: for each regime (BULL / CORRECTION / BEAR),
   what was the strategy's compound return vs SPY's?
2. **Drawdown attribution**: for each significant SPY drawdown in the
   window, what was the strategy's drawdown over the same window — i.e.,
   did it lose less than SPY did?

Definitions
-----------
- **BULL**: SPY within 5% of its 252-day rolling high.
- **CORRECTION**: SPY between -5% and -15% from its 252-day rolling high.
- **BEAR**: SPY below -15% from its 252-day rolling high.

Drawdowns are identified peak-to-trough on the SPY close series, filtered
to those exceeding `min_dd_pct` (default 5%) and lasting at least
`min_days` (default 5) trading days. A trailing un-recovered drawdown is
reported with `recovery_date=None`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import BacktestResult


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

REGIME_BULL = "BULL"
REGIME_CORRECTION = "CORRECTION"
REGIME_BEAR = "BEAR"

# (lower_bound, upper_bound, label) — exclusive upper, inclusive lower except
# the very bottom band which is half-open below. Encoded as a table so the
# thresholds are inspectable in one place.
_REGIME_BANDS: list[tuple[float, str]] = [
    (-0.05, REGIME_BULL),         # dd > -5%
    (-0.15, REGIME_CORRECTION),   # -15% < dd <= -5%
    (-1.00, REGIME_BEAR),         # dd <= -15%
]


def classify_regimes(spy_close: pd.Series,
                     lookback: int = 252) -> pd.Series:
    """Per-day regime classification from SPY close.

    Regime is decided by drawdown from a rolling 252-day high (clipped to
    expanding-min behaviour for the warmup). Date-indexed string Series
    aligned to `spy_close`. Days with insufficient history (no rolling
    max available yet) are labelled BULL — the safest assumption pre-data.
    """
    s = spy_close.dropna()
    rolling_max = s.rolling(lookback, min_periods=1).max()
    dd_from_high = s / rolling_max - 1.0
    regimes = pd.Series(REGIME_BULL, index=s.index, name="regime")
    for d, dd in dd_from_high.items():
        for threshold, label in _REGIME_BANDS:
            if dd > threshold:
                regimes.loc[d] = label
                break
        else:
            regimes.loc[d] = REGIME_BEAR
    return regimes


def regime_episodes(regimes: pd.Series) -> pd.DataFrame:
    """Group consecutive same-regime days into episodes.

    Returns DataFrame columns: regime, start, end, n_days. Useful for the
    drawdown-timeline view and for sanity-checking the classification.
    """
    if regimes.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "n_days"])
    runs = (regimes != regimes.shift()).cumsum()
    eps: list[dict] = []
    for _, sub in regimes.groupby(runs):
        eps.append({
            "regime": sub.iloc[0],
            "start": sub.index[0],
            "end": sub.index[-1],
            "n_days": int(len(sub)),
        })
    return pd.DataFrame(eps)


# ---------------------------------------------------------------------------
# Per-regime conditional stats
# ---------------------------------------------------------------------------

def regime_conditional_stats(equity: pd.Series,
                              benchmark_equity: pd.Series,
                              regimes: pd.Series) -> pd.DataFrame:
    """Compound the strategy and benchmark daily returns within each regime.

    Returns DataFrame indexed by regime label with columns:
      n_days, years, strategy_cum, spy_cum, excess_cum,
      strategy_cagr, spy_cagr, excess_cagr,
      strategy_ann_vol, spy_ann_vol, strategy_mdd_in_regime,
      spy_mdd_in_regime, capture_up, capture_down.

    `capture_up` = strategy_ret / spy_ret on days where spy_ret > 0,
    `capture_down` = strategy_ret / spy_ret on days where spy_ret < 0 —
    the classic rotation-strategy diagnostics. <100% down-capture and
    >100% up-capture is the dream; <100% on both is "low-beta drag".
    """
    df = pd.concat([
        equity.rename("equity"),
        benchmark_equity.rename("spy"),
        regimes.rename("regime"),
    ], axis=1).dropna()
    df["strat_ret"] = df["equity"].pct_change()
    df["spy_ret"] = df["spy"].pct_change()
    df = df.dropna()

    rows: list[dict] = []
    for r, sub in df.groupby("regime"):
        if len(sub) < 2:
            continue
        strat_cum = float((1 + sub["strat_ret"]).prod() - 1)
        spy_cum = float((1 + sub["spy_ret"]).prod() - 1)
        years = len(sub) / 252.0
        cagr = lambda x: ((1 + x) ** (1 / years) - 1) if years > 0 else 0.0  # noqa: E731

        # Per-regime MDD (computed on equity rebased at regime start so
        # cross-regime carryover doesn't pollute).
        rebased_s = (1 + sub["strat_ret"]).cumprod()
        rebased_b = (1 + sub["spy_ret"]).cumprod()
        s_mdd = float((rebased_s / rebased_s.cummax() - 1).min())
        b_mdd = float((rebased_b / rebased_b.cummax() - 1).min())

        # Up / down capture (only meaningful when SPY had non-zero days).
        up = sub[sub["spy_ret"] > 0]
        dn = sub[sub["spy_ret"] < 0]
        cap_up = (float(up["strat_ret"].mean() / up["spy_ret"].mean())
                  if len(up) and up["spy_ret"].mean() != 0 else float("nan"))
        cap_dn = (float(dn["strat_ret"].mean() / dn["spy_ret"].mean())
                  if len(dn) and dn["spy_ret"].mean() != 0 else float("nan"))

        rows.append({
            "regime": r,
            "n_days": int(len(sub)),
            "years": round(years, 2),
            "strategy_cum": strat_cum,
            "spy_cum": spy_cum,
            "excess_cum": strat_cum - spy_cum,
            "strategy_cagr": cagr(strat_cum),
            "spy_cagr": cagr(spy_cum),
            "excess_cagr": cagr(strat_cum) - cagr(spy_cum),
            "strategy_ann_vol": float(sub["strat_ret"].std(ddof=0) * np.sqrt(252)),
            "spy_ann_vol": float(sub["spy_ret"].std(ddof=0) * np.sqrt(252)),
            "strategy_mdd_in_regime": s_mdd,
            "spy_mdd_in_regime": b_mdd,
            "capture_up": cap_up,
            "capture_down": cap_dn,
        })
    out = pd.DataFrame(rows).set_index("regime")
    # Stable ordering BULL → CORRECTION → BEAR.
    order = [r for r in (REGIME_BULL, REGIME_CORRECTION, REGIME_BEAR)
             if r in out.index]
    return out.loc[order]


# ---------------------------------------------------------------------------
# Drawdown identification
# ---------------------------------------------------------------------------

@dataclass
class DrawdownEpisode:
    peak_date: pd.Timestamp
    peak_value: float
    trough_date: pd.Timestamp
    trough_value: float
    recovery_date: pd.Timestamp | None     # first new high after trough
    drawdown_pct: float                    # signed, negative
    days_to_trough: int
    days_to_recover: int | None


def identify_drawdowns(spy_close: pd.Series,
                       min_dd_pct: float = 0.05,
                       min_days: int = 5) -> list[DrawdownEpisode]:
    """Peak-to-trough drawdown episodes on a price series.

    Walks the series tracking the running peak. When a new high is set,
    any in-flight drawdown is closed (with the previous bar as recovery
    date). A trailing un-recovered drawdown is reported with
    `recovery_date=None`.

    Filters out drawdowns shallower than `min_dd_pct` (default 5%) or
    shorter than `min_days` peak-to-trough.
    """
    s = spy_close.dropna()
    if s.empty:
        return []
    episodes: list[DrawdownEpisode] = []
    peak_value = float(s.iloc[0])
    peak_date = s.index[0]
    trough_value = peak_value
    trough_date = peak_date
    in_dd = False

    for d, raw in s.items():
        price = float(raw)
        if price >= peak_value:
            if in_dd:
                dd_pct = (trough_value / peak_value) - 1.0
                dd_days = (trough_date - peak_date).days
                if abs(dd_pct) >= min_dd_pct and dd_days >= min_days:
                    episodes.append(DrawdownEpisode(
                        peak_date=peak_date, peak_value=peak_value,
                        trough_date=trough_date, trough_value=trough_value,
                        recovery_date=d, drawdown_pct=dd_pct,
                        days_to_trough=dd_days,
                        days_to_recover=(d - trough_date).days,
                    ))
                in_dd = False
            peak_value = price
            peak_date = d
            trough_value = price
            trough_date = d
        else:
            in_dd = True
            if price < trough_value:
                trough_value = price
                trough_date = d

    if in_dd:
        dd_pct = (trough_value / peak_value) - 1.0
        dd_days = (trough_date - peak_date).days
        if abs(dd_pct) >= min_dd_pct and dd_days >= min_days:
            episodes.append(DrawdownEpisode(
                peak_date=peak_date, peak_value=peak_value,
                trough_date=trough_date, trough_value=trough_value,
                recovery_date=None, drawdown_pct=dd_pct,
                days_to_trough=dd_days, days_to_recover=None,
            ))
    return episodes


# ---------------------------------------------------------------------------
# Drawdown attribution
# ---------------------------------------------------------------------------

def _value_on_or_before(series: pd.Series, d: pd.Timestamp) -> float | None:
    """Pick the last value on or before `d`. Returns None if no such bar."""
    idx = series.index.searchsorted(d, side="right")
    if idx == 0:
        return None
    return float(series.iloc[idx - 1])


def _states_on_date(states_history: pd.DataFrame,
                    d: pd.Timestamp) -> dict[str, str]:
    """Per-ticker state as of the most recent rebalance ≤ d."""
    sub = states_history[states_history["date"] <= d]
    if sub.empty:
        return {}
    latest_rb = sub["date"].max()
    snap = sub[sub["date"] == latest_rb]
    return dict(zip(snap["ticker"], snap["state"]))


def _portfolio_value_during(result: BacktestResult,
                             d: pd.Timestamp) -> tuple[float, float, str]:
    """Total equity + cash share + dominant-state label on `d`."""
    eq = _value_on_or_before(result.equity, d) or 0.0
    # Cash share is hard to extract post-hoc; approximate by 1 - sum of held
    # weights. We don't carry per-day weights, only per-rebalance targets —
    # use them as a proxy.
    if result.weights_history.empty:
        return eq, float("nan"), "—"
    wh = result.weights_history
    wh_dates = pd.to_datetime(wh["date"])
    idx = wh_dates.searchsorted(d, side="right")
    if idx == 0:
        return eq, float("nan"), "—"
    row = wh.iloc[idx - 1]
    cash = float(row.get("cash_buffer", float("nan")))
    return eq, cash, str(row.get("fill_date", "—"))


def drawdown_attribution(result: BacktestResult,
                          benchmark_equity: pd.Series,
                          spy_close: pd.Series,
                          min_dd_pct: float = 0.05,
                          min_days: int = 5) -> list[dict]:
    """For each SPY drawdown episode, compare strategy behaviour.

    Per episode returns:
      - peak/trough dates and values
      - SPY drawdown over the window
      - Strategy drawdown over the SAME window (peak → SPY trough)
      - Excess (positive = strategy lost less)
      - States held at peak vs trough (model rotation evidence)
      - Best / worst sectors over the period (price-only — what the model
        SHOULD have rotated into)
    """
    eps = identify_drawdowns(spy_close, min_dd_pct=min_dd_pct,
                              min_days=min_days)
    if not eps:
        return []

    # Only attribute drawdowns whose PEAK falls inside the backtest window —
    # otherwise the strategy had no equity yet and the comparison is bogus.
    eval_start = result.equity.index[0]
    eps = [e for e in eps if e.peak_date >= eval_start]
    if not eps:
        return []

    rows: list[dict] = []
    for ep in eps:
        s_peak = _value_on_or_before(result.equity, ep.peak_date)
        s_trough = _value_on_or_before(result.equity, ep.trough_date)
        b_peak = _value_on_or_before(benchmark_equity, ep.peak_date)
        b_trough = _value_on_or_before(benchmark_equity, ep.trough_date)
        if not (s_peak and s_trough and b_peak and b_trough):
            continue
        strat_dd = s_trough / s_peak - 1.0
        spy_dd = b_trough / b_peak - 1.0

        states_at_peak = _states_on_date(result.states_history, ep.peak_date)
        states_at_trough = _states_on_date(result.states_history, ep.trough_date)
        held_at_peak = sorted([
            t for t, s in states_at_peak.items()
            if s in ("NEW_BUY", "HOLD_IF_LONG")
        ])
        held_at_trough = sorted([
            t for t, s in states_at_trough.items()
            if s in ("NEW_BUY", "HOLD_IF_LONG")
        ])
        rotated_in = sorted(set(held_at_trough) - set(held_at_peak))
        rotated_out = sorted(set(held_at_peak) - set(held_at_trough))

        rows.append({
            "peak_date": ep.peak_date.date(),
            "trough_date": ep.trough_date.date(),
            "recovery_date": (ep.recovery_date.date()
                              if ep.recovery_date is not None else None),
            "days_to_trough": ep.days_to_trough,
            "days_to_recover": ep.days_to_recover,
            "spy_drawdown": spy_dd,
            "strategy_drawdown": strat_dd,
            "excess_drawdown": strat_dd - spy_dd,
            "held_at_peak": held_at_peak,
            "held_at_trough": held_at_trough,
            "rotated_in_during_dd": rotated_in,
            "rotated_out_during_dd": rotated_out,
        })
    return rows


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def run_regime_analysis(result: BacktestResult,
                        spy_close: pd.Series,
                        min_dd_pct: float = 0.05,
                        min_days: int = 5) -> dict:
    """One-call: regimes + per-regime stats + drawdown attribution.

    Returns:
      {
        "regimes": Series,
        "regime_stats": DataFrame,
        "drawdowns": list[dict],
      }
    """
    regimes = classify_regimes(spy_close.reindex(result.equity.index, method="ffill"))
    stats = regime_conditional_stats(result.equity, result.benchmark_equity,
                                      regimes)
    dd = drawdown_attribution(result, result.benchmark_equity, spy_close,
                               min_dd_pct=min_dd_pct, min_days=min_days)
    return {"regimes": regimes, "regime_stats": stats, "drawdowns": dd}
