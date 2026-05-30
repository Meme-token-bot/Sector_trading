"""'Tilt, don't rotate' backtest — SPY core + a momentum overweight that RIDES.

Motivation
----------
The state-machine rotation strategy trails SPY by ~5.2%/yr over 2019-2026, and
the backtest attributes most of the gap to STRUCTURAL DRIFT: it trims winners
every week to fund laggards, and it rotates OUT of the one sector that
mattered (tech) by flagging it "too extended". This backtest tests the
opposite discipline:

  * Hold a permanent SPY core (`core_weight`).
  * Put the rest in the top-`top_n` sectors by trailing momentum, weighted by
    momentum strength — and then LEAVE IT until the next rebalance.
  * Rebalance INFREQUENTLY (monthly / quarterly / annual) so winners are
    allowed to run instead of being trimmed.

This is deliberately a *concentration bet* (it will overweight whatever is
leading), not a defensive strategy. We measure it against 100% SPY in total
dollars — the bar the user wants to beat — net of costs.

No look-ahead: at each rebalance date `t`, momentum uses only closes <= t, and
the new weights apply to returns AFTER t.

Reproduce:
    PYTHONPATH=. python3 scripts/tilt_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import BENCHMARK, SECTOR_ETFS, SUPPLEMENTARY_SECTORS  # noqa: E402
from src.price_store import load_ohlcv_multi  # noqa: E402

TRADING_DAYS = 252
START = pd.Timestamp("2019-01-18")   # matches the rotation backtest's first tradeable week
INITIAL = 10_000.0
COST_BPS = 5.0                       # per-side, on turnover


def load_closes() -> pd.DataFrame:
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    tickers = universe + [BENCHMARK]
    wide = load_ohlcv_multi(tickers, "1d")
    close = wide.xs("close", axis=1, level=1).sort_index()
    close.index = pd.to_datetime(close.index)
    return close.loc[START:].dropna(how="all"), universe


def momentum(close: pd.DataFrame, t: pd.Timestamp, lookback: int) -> pd.Series:
    """Trailing `lookback`-day total return as of date t (causal)."""
    hist = close.loc[:t]
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    window = hist.iloc[-(lookback + 1):]
    return window.iloc[-1] / window.iloc[0] - 1.0


def rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """Last trading day of each period."""
    s = pd.Series(idx, index=idx)
    rule = {"M": "ME", "Q": "QE", "A": "YE"}[freq]
    return list(pd.DatetimeIndex(s.groupby(s.dt.to_period({"M": "M", "Q": "Q", "A": "Y"}[freq])).last().values))


def target_weights(close, universe, t, core_weight, top_n, lookback) -> pd.Series:
    """SPY core + momentum-weighted tilt into the top_n sectors as of t."""
    w = pd.Series(0.0, index=close.columns)
    w[BENCHMARK] = core_weight
    mom = momentum(close[universe], t, lookback).dropna()
    if mom.empty:
        w[BENCHMARK] = 1.0
        return w
    top = mom.sort_values(ascending=False).head(top_n)
    # Only tilt into POSITIVE-momentum sectors; if none, park tilt in SPY.
    top = top[top > 0]
    tilt = 1.0 - core_weight
    if top.empty:
        w[BENCHMARK] = w[BENCHMARK] + tilt
        return w
    # Momentum-proportional weights within the tilt sleeve.
    w_tilt = top / top.sum() * tilt
    for tkr, wt in w_tilt.items():
        w[tkr] += wt
    return w


def run(close, universe, core_weight, top_n, lookback, freq) -> dict:
    idx = close.index
    rebs = [d for d in rebalance_dates(idx, freq) if d >= idx[0] and d <= idx[-1]]
    daily_ret = close.pct_change().fillna(0.0)

    equity = INITIAL
    curve = pd.Series(index=idx, dtype=float)
    cur_w = pd.Series(0.0, index=close.columns)
    cur_w[BENCHMARK] = 1.0
    next_reb_i = 0
    turnover_total = 0.0

    prev_date = idx[0]
    for d in idx:
        # apply the day's return to current weights
        r = float((cur_w * daily_ret.loc[d]).sum())
        equity *= (1.0 + r)
        curve[d] = equity
        # rebalance at end of this day if it's a rebalance date
        if next_reb_i < len(rebs) and d == rebs[next_reb_i]:
            new_w = target_weights(close, universe, d, core_weight, top_n, lookback)
            turnover = float((new_w - cur_w).abs().sum()) / 2.0
            cost = turnover * (COST_BPS / 10_000.0)
            equity *= (1.0 - cost)
            curve[d] = equity
            turnover_total += turnover
            cur_w = new_w
            next_reb_i += 1
        prev_date = d

    curve = curve.dropna()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1.0
    mdd = float((curve / curve.cummax() - 1.0).min())
    ann_turnover = turnover_total / years
    return {"final": float(curve.iloc[-1]), "cagr": float(cagr), "mdd": mdd,
            "ann_turnover": float(ann_turnover), "curve": curve}


def spy_benchmark(close) -> dict:
    spy = close[BENCHMARK].dropna()
    curve = spy / spy.iloc[0] * INITIAL
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1.0
    mdd = float((curve / curve.cummax() - 1.0).min())
    return {"final": float(curve.iloc[-1]), "cagr": float(cagr), "mdd": mdd,
            "ann_turnover": 0.0}


def main() -> int:
    close, universe = load_closes()
    print(f"Window: {close.index[0].date()} → {close.index[-1].date()}  "
          f"(${INITIAL:,.0f} start, {COST_BPS:.0f}bps/side)\n")

    spy = spy_benchmark(close)

    configs = [
        # (core_weight, top_n, lookback_days, freq)
        (0.60, 3, 126, "Q"),
        (0.60, 2, 126, "Q"),
        (0.50, 3, 126, "Q"),
        (0.50, 2, 252, "Q"),
        (0.70, 3, 126, "M"),
        (0.50, 3, 126, "A"),
        (0.00, 3, 126, "Q"),   # pure tilt, no SPY core (max concentration)
        (0.00, 1, 126, "Q"),   # single best sector, quarterly
    ]

    rows = []
    for cw, tn, lb, fr in configs:
        res = run(close, universe, cw, tn, lb, fr)
        rows.append((f"core={cw:.0%} top{tn} {lb}d {fr}", res))

    name_w = 26
    print(f"{'Strategy':<{name_w}}{'Final $':>11}{'CAGR':>9}{'MaxDD':>9}{'Turn/yr':>9}{'vs SPY $':>11}")
    print("-" * (name_w + 49))
    print(f"{'100% SPY':<{name_w}}{spy['final']:>11,.0f}{spy['cagr']*100:>8.1f}%"
          f"{spy['mdd']*100:>8.1f}%{spy['ann_turnover']:>8.1f}x{0:>11,.0f}")
    print(f"{'rotation (state-machine)':<{name_w}}{22535:>11,.0f}{11.7:>8.1f}%"
          f"{-26.1:>8.1f}%{16.5:>8.1f}x{22535-spy['final']:>11,.0f}")
    print("-" * (name_w + 49))
    for name, res in rows:
        print(f"{name:<{name_w}}{res['final']:>11,.0f}{res['cagr']*100:>8.1f}%"
              f"{res['mdd']*100:>8.1f}%{res['ann_turnover']:>8.1f}x"
              f"{res['final']-spy['final']:>11,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
