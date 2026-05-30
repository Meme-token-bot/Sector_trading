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
    horizon: str = "next_state_exit",
    source: str = "auto",
) -> dict:
    """Forward excess return of refined NEW_BUY signals vs the benchmark.

    Reads persisted refined states from ``signal_snapshots`` first; falls
    back to scanning the in-memory `history` frame (which carries raw labels
    only) when the persisted table is empty — useful before any snapshots
    have been written.

    `horizon` controls how long each signal is held:
      * "next_state_exit" (default) — hold from the snapshot date until the
        first subsequent snapshot where the ticker's state leaves the BUY
        class (SELL/REDUCE/HOLD/CHASE). This matches the live "hold until
        state change" rule, so the headline number reflects the strategy's
        actual P&L profile rather than an arbitrary 1-week clip.
      * "1w" — legacy fixed 1-week horizon. Faster to compute and what the
        UI used to claim it was showing.

    Returns:
      {
        n_signals: int,
        mean_excess_return: float,
        hit_rate: float,
        median_hold_days: float | None,
        by_state: {state: {n, mean_excess_return, hit_rate}},
        horizon: str,
        source: "snapshots" | "history",
      }
    """
    empty = {
        "n_signals": 0,
        "mean_excess_return": 0.0,
        "hit_rate": 0.0,
        "median_hold_days": None,
        "by_state": {},
        "horizon": horizon,
        "source": "none",
    }
    if benchmark_ticker not in prices:
        return empty
    bench = prices[benchmark_ticker].dropna()
    if bench.empty:
        return empty

    # ---- Source selection -------------------------------------------------
    # 'auto'     : prefer persisted snapshots (the strict NEW_BUY record);
    #              fall back to raw history when the table is empty.
    # 'snapshots': read snapshots only (return empty if missing).
    # 'history'  : ignore snapshots entirely and use the in-memory `history`
    #              frame (raw BUY/HOLD/SELL labels). Useful in tests that
    #              fabricate a history but don't write snapshots.
    if source == "history":
        snaps = pd.DataFrame()
    else:
        from src.db import load_signal_snapshots
        snaps = load_signal_snapshots()
        if snaps.empty and source == "snapshots":
            return {**empty, "source": "snapshots"}
    if not snaps.empty:
        # Build a date-indexed refined-state frame from snapshots, then
        # apply the same trailing-`weeks` clip used by the UI.
        wide = snaps.pivot(index="as_of", columns="ticker", values="state").sort_index()
        wide = wide.tail(weeks)
        source = "snapshots"
        is_new_buy = lambda v: isinstance(v, str) and v == "NEW_BUY"  # noqa: E731
    else:
        if history is None or history.empty or len(history) < 4:
            return empty
        wide = history.tail(weeks)
        source = "history"
        # raw history carries only BUY/HOLD/SELL — accept either label as the
        # NEW_BUY proxy until snapshots exist.
        is_new_buy = lambda v: isinstance(v, str) and v in ("BUY", "NEW_BUY")  # noqa: E731

    records: list[dict] = []  # {state, excess, hold_days}

    snap_dates = list(wide.index)
    for i, snap_date in enumerate(snap_dates):
        row = wide.loc[snap_date]
        snap_ts = pd.Timestamp(snap_date)

        for ticker, label in row.items():
            if not is_new_buy(label):
                continue
            s = prices.get(ticker)
            if s is None:
                continue
            s = s.dropna()
            if s.empty:
                continue

            # Entry bar: first trading day >= snap_date.
            s_entry_i = s.index.searchsorted(snap_ts, side="left")
            b_entry_i = bench.index.searchsorted(snap_ts, side="left")
            if s_entry_i >= len(s) or b_entry_i >= len(bench):
                continue

            # Exit bar: depends on horizon.
            if horizon == "1w":
                exit_ts = snap_ts + pd.Timedelta(days=7)
            else:  # next_state_exit
                exit_ts = None
                for j in range(i + 1, len(snap_dates)):
                    nxt_label = wide.loc[snap_dates[j], ticker]
                    if isinstance(nxt_label, str) and nxt_label not in (
                            "NEW_BUY", "HOLD_IF_LONG", "BUY"):
                        exit_ts = pd.Timestamp(snap_dates[j])
                        break
                if exit_ts is None:
                    # Still in BUY class as of the latest snapshot — mark to
                    # the last available bar so live positions still count.
                    exit_ts = s.index[-1]

            s_exit_i = s.index.searchsorted(exit_ts, side="left")
            b_exit_i = bench.index.searchsorted(exit_ts, side="left")
            if s_exit_i >= len(s):
                s_exit_i = len(s) - 1
            if b_exit_i >= len(bench):
                b_exit_i = len(bench) - 1
            if s_exit_i <= s_entry_i or b_exit_i <= b_entry_i:
                continue

            s_e, s_x = float(s.iloc[s_entry_i]), float(s.iloc[s_exit_i])
            b_e, b_x = float(bench.iloc[b_entry_i]), float(bench.iloc[b_exit_i])
            if s_e == 0 or b_e == 0:
                continue
            excess = (s_x / s_e - 1.0) - (b_x / b_e - 1.0)
            hold_days = (s.index[s_exit_i] - s.index[s_entry_i]).days
            records.append({"state": str(label), "excess": excess,
                             "hold_days": hold_days})

    if not records:
        return {**empty, "source": source}

    arr = np.array([r["excess"] for r in records], dtype=float)
    n = int(arr.size)
    by_state: dict[str, dict[str, float]] = {}
    for st in sorted({r["state"] for r in records}):
        sub = np.array([r["excess"] for r in records if r["state"] == st],
                       dtype=float)
        by_state[st] = {
            "n": int(sub.size),
            "mean_excess_return": float(sub.mean()),
            "hit_rate": float((sub > 0).mean()),
        }
    hold_days = np.array([r["hold_days"] for r in records], dtype=float)
    return {
        "n_signals": n,
        "mean_excess_return": float(arr.mean()),
        "hit_rate": float((arr > 0).mean()),
        "median_hold_days": float(np.median(hold_days)),
        "by_state": by_state,
        "horizon": horizon,
        "source": source,
    }
