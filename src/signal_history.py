"""Replay weekly BUY/HOLD/SELL signals over recent history.

Used by the state-aware classifier in src/signals.py to detect:
- Sectors that just turned BUY (NEW_BUY — fresh entry OK)
- Sectors that have been BUY for too long (HOLD_IF_LONG — don't chase)
- Sectors that recently degraded from BUY (REDUCE)

Pure function over (prices, sentiment_db). Stateless. Same inputs → same output.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config.settings import PARAMS, SECTOR_ETFS
from src.db import aggregate_sentiment
from src.market_engine import compute_sector_metrics
from src.signals import build_signals


def build_signal_history(prices: pd.DataFrame,
                         end: date | None = None,
                         n_weeks: int | None = None) -> pd.DataFrame:
    """Weekly signal snapshots, oldest -> newest.

    Returns DataFrame with index=week date, columns=sector tickers,
    values=signal label ('BUY' / 'HOLD' / 'SELL'). Missing data => NaN.
    """
    end = end or date.today()
    n = n_weeks or PARAMS.history_weeks

    snapshots: dict[pd.Timestamp, pd.Series] = {}
    for i in range(n - 1, -1, -1):
        d = end - timedelta(weeks=i)
        m = compute_sector_metrics(prices, as_of=pd.Timestamp(d))
        if m.empty:
            continue
        s = aggregate_sentiment(as_of=d)
        sig = build_signals(m, s)
        snapshots[pd.Timestamp(d)] = sig["signal"]

    if not snapshots:
        return pd.DataFrame(columns=list(SECTOR_ETFS.keys()))

    df = pd.DataFrame(snapshots).T
    df = df.reindex(columns=list(SECTOR_ETFS.keys()))
    df.index.name = "week"
    return df.sort_index()


def consecutive_buy_weeks(history: pd.DataFrame) -> pd.Series:
    """For each sector, count how many consecutive most-recent weekly
    snapshots had signal == 'BUY'. Returns Series indexed by ticker.
    """
    if history.empty:
        return pd.Series(dtype=int)

    # Walk backwards from the latest snapshot; stop on first non-BUY per ticker.
    rev = history.iloc[::-1]
    counts: dict[str, int] = {}
    for tkr in rev.columns:
        n = 0
        for v in rev[tkr]:
            if v == "BUY":
                n += 1
            else:
                break
        counts[tkr] = n
    return pd.Series(counts, name="consecutive_buy_weeks")


def _pick_current_state(current: pd.DataFrame, ticker: str) -> str | None:
    """Resolve a sector's "current" label, preferring the refined state."""
    for col in ("state", "signal"):
        if col in current.columns and ticker in current.index:
            val = current.loc[ticker, col]
            if isinstance(val, str) and val:
                return val
    return None


def _change_reason(current: pd.DataFrame, ticker: str,
                   prior: str, new: str) -> str:
    """Heuristically pick the dominant driver for a state flip.

    Reads columns from `current` (the refined signals frame) when available.
    Falls back to a generic "{prior} -> {new}" string when nothing useful
    is present.
    """
    if ticker not in current.index:
        return f"{prior} -> {new}"

    row = current.loc[ticker]

    def _num(col: str) -> float | None:
        if col not in current.columns:
            return None
        try:
            v = float(row[col])
        except (TypeError, ValueError):
            return None
        if np.isnan(v):
            return None
        return v

    rs3 = _num("relative_strength_3m")
    above_sma = bool(row["above_sma"]) if "above_sma" in current.columns else None
    ext = _num("extension_pct")
    sent = _num("sentiment_score")

    # Bias the explanation by the kind of flip.
    leaving_buy = prior in ("BUY", "NEW_BUY", "HOLD_IF_LONG", "CHASE")
    entering_buy = new in ("BUY", "NEW_BUY", "HOLD_IF_LONG", "CHASE")

    if new == "SELL":
        if above_sma is False:
            return "crossed below SMA200"
        if sent is not None and sent <= -3:
            return f"sentiment turned negative ({sent:+.1f})"
        if rs3 is not None and rs3 < 0:
            return f"RS turned negative ({rs3*100:+.1f}%)"
        return f"{prior} -> SELL"

    if new == "CHASE":
        if ext is not None:
            return f"became extended past cutoff ({ext*100:+.1f}% above SMA200)"
        return "became extended past cutoff"

    if leaving_buy and not entering_buy:
        # Lost BUY status — figure out what gave way first.
        if above_sma is False:
            return "crossed below SMA200"
        if rs3 is not None and rs3 <= 0:
            return f"RS turned negative ({rs3*100:+.1f}%)"
        if sent is not None:
            return f"sentiment fell to {sent:+.1f}"
        return f"{prior} -> {new}"

    if entering_buy and not leaving_buy:
        # Gained BUY status — pick the strongest contributor.
        if rs3 is not None and rs3 > 0:
            return f"RS turned positive ({rs3*100:+.1f}%)"
        if above_sma is True:
            return "crossed above SMA200"
        if sent is not None and sent >= 2:
            return f"sentiment rose to {sent:+.1f}"
        return f"{prior} -> {new}"

    # State refinement flip inside the BUY family (e.g. NEW_BUY -> HOLD_IF_LONG).
    if "state_reason" in current.columns:
        sr = row["state_reason"]
        if isinstance(sr, str) and sr:
            return sr
    return f"{prior} -> {new}"


def detect_state_changes(history: pd.DataFrame,
                         current: pd.DataFrame) -> pd.DataFrame:
    """One row per sector whose state changed vs the most-recent prior snapshot.

    Columns: sector, prior_state, new_state, reason.
    Returns an empty frame when no prior snapshot exists or no sector flipped.

    `history` is the weekly snapshot frame from `build_signal_history`
    (index = week date, columns = sector tickers, values = raw signal labels).
    `current` is the *refined* signals frame returned by
    `src.signals.refine_signals` — its `state` column (or `signal` if state
    is absent) supplies `new_state`.
    """
    cols = ["sector", "prior_state", "new_state", "reason"]
    if history is None or history.empty:
        return pd.DataFrame(columns=cols)

    last = history.iloc[-1]
    rows: list[dict[str, Any]] = []
    for ticker in current.index:
        new = _pick_current_state(current, ticker)
        if new is None:
            continue
        prior_raw = last.get(ticker) if ticker in last.index else None
        if prior_raw is None or (isinstance(prior_raw, float) and np.isnan(prior_raw)):
            continue
        prior = str(prior_raw)
        if prior == new:
            continue
        rows.append({
            "sector": ticker,
            "prior_state": prior,
            "new_state": new,
            "reason": _change_reason(current, ticker, prior, new),
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def signal_performance_vs_benchmark(
    history: pd.DataFrame,
    prices: dict[str, pd.Series],
    benchmark_ticker: str = "SPY",
    weeks: int = 12,
) -> dict:
    """Forward 1-week excess return for sectors flagged NEW_BUY/BUY in history.

    For each weekly snapshot in the trailing `weeks` window, identify
    sectors whose raw signal == 'BUY' (the closest analogue of NEW_BUY in
    the persisted history frame, which stores raw labels) and compute the
    sector's forward 1-week return vs the benchmark's forward 1-week
    return. Returns aggregates and a per-state breakdown.

    `prices` is a mapping ticker -> close Series indexed by date. Forward
    returns are computed by stepping ~5 trading days (one calendar week)
    after each snapshot date.

    Returns:
      {
        n_signals: int,
        mean_excess_return: float,
        hit_rate: float,
        by_state: {state: {n, mean_excess_return, hit_rate}},
      }
    Returns {n_signals: 0, mean_excess_return: 0.0, hit_rate: 0.0, by_state: {}}
    if `history` is shorter than 4 weeks.
    """
    empty = {
        "n_signals": 0,
        "mean_excess_return": 0.0,
        "hit_rate": 0.0,
        "by_state": {},
    }
    if history is None or history.empty or len(history) < 4:
        return empty
    if benchmark_ticker not in prices:
        return empty

    bench = prices[benchmark_ticker].dropna()
    if bench.empty:
        return empty

    window = history.tail(weeks)
    records: list[tuple[str, float]] = []  # (state_label, excess_return)

    for snap_date, row in window.iterrows():
        snap_ts = pd.Timestamp(snap_date)
        # Forward 1-week point (~5 trading days). Find the first trading
        # day >= snap_ts (entry) and the first trading day >= snap_ts + 7d
        # (exit). Skip if either window-end falls outside available bars.
        b_idx_entry = bench.index.searchsorted(snap_ts, side="left")
        b_idx_exit = bench.index.searchsorted(snap_ts + pd.Timedelta(days=7), side="left")
        if b_idx_entry >= len(bench) or b_idx_exit >= len(bench):
            continue
        if b_idx_exit <= b_idx_entry:
            continue
        b_entry = float(bench.iloc[b_idx_entry])
        b_exit = float(bench.iloc[b_idx_exit])
        if b_entry == 0:
            continue
        bench_fwd = b_exit / b_entry - 1.0

        for ticker, label in row.items():
            if not isinstance(label, str):
                continue
            if label not in ("BUY", "NEW_BUY"):
                continue
            s = prices.get(ticker)
            if s is None:
                continue
            s = s.dropna()
            if s.empty:
                continue
            s_idx_entry = s.index.searchsorted(snap_ts, side="left")
            s_idx_exit = s.index.searchsorted(snap_ts + pd.Timedelta(days=7), side="left")
            if s_idx_entry >= len(s) or s_idx_exit >= len(s):
                continue
            if s_idx_exit <= s_idx_entry:
                continue
            entry = float(s.iloc[s_idx_entry])
            exit_ = float(s.iloc[s_idx_exit])
            if entry == 0:
                continue
            fwd = exit_ / entry - 1.0
            excess = fwd - bench_fwd
            records.append((label, excess))

    if not records:
        return empty

    arr = np.array([r[1] for r in records], dtype=float)
    n = int(arr.size)
    mean_ex = float(arr.mean())
    hit_rate = float((arr > 0).mean())

    by_state: dict[str, dict[str, float]] = {}
    states = sorted({r[0] for r in records})
    for st in states:
        sub = np.array([x for (s, x) in records if s == st], dtype=float)
        by_state[st] = {
            "n": int(sub.size),
            "mean_excess_return": float(sub.mean()),
            "hit_rate": float((sub > 0).mean()),
        }

    return {
        "n_signals": n,
        "mean_excess_return": mean_ex,
        "hit_rate": hit_rate,
        "by_state": by_state,
    }
