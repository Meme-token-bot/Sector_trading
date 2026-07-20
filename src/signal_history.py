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
    if history.empty:
        return pd.Series(dtype=int)

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
    for col in ("state", "signal"):
        if col in current.columns and ticker in current.index:
            val = current.loc[ticker, col]
            if isinstance(val, str) and val:
                return val
    return None


def _change_reason(current: pd.DataFrame, ticker: str,
                   prior: str, new: str) -> str:
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
        if above_sma is False:
            return "crossed below SMA200"
        if rs3 is not None and rs3 <= 0:
            return f"RS turned negative ({rs3*100:+.1f}%)"
        if sent is not None:
            return f"sentiment fell to {sent:+.1f}"
        return f"{prior} -> {new}"

    if entering_buy and not leaving_buy:
        if rs3 is not None and rs3 > 0:
            return f"RS turned positive ({rs3*100:+.1f}%)"
        if above_sma is True:
            return "crossed above SMA200"
        if sent is not None and sent >= 2:
            return f"sentiment rose to {sent:+.1f}"
        return f"{prior} -> {new}"

    if "state_reason" in current.columns:
        sr = row["state_reason"]
        if isinstance(sr, str) and sr:
            return sr
    return f"{prior} -> {new}"


def detect_state_changes(history: pd.DataFrame,
                         current: pd.DataFrame) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Shared trade-resolution core
# ---------------------------------------------------------------------------
# Extracted from the body of signal_performance_vs_benchmark (which used to
# inline this logic once, for a single "as of right now" evaluation) so that
# `rolling_signal_performance` and `performance_by_conviction`
# (TRADING_EDGE_AUDIT.md items S3/S4) can reuse EXACTLY the same trade
# resolution rules instead of a second, drift-prone reimplementation.
#
# LOOK-AHEAD DISCIPLINE (read this before touching the function):
#   `evaluation_date` is the caller's "now". Two places can leak future
#   information if this isn't respected:
#     1. Finding an exit via "next state transition" — the caller MUST have
#        already excluded snapshot dates > evaluation_date from `wide`
#        before calling this function, or the exit search will walk into
#        the future.
#     2. Marking a still-open position — must use the last price bar ON OR
#        BEFORE evaluation_date, never a later one. The subtlety: when
#        `evaluation_date is None` (the single "as of right now" call site),
#        "last bar in the series" and "last bar on or before evaluation_date"
#        are the same thing, since `prices` is fetched fresh — so that call
#        path is intentionally left byte-identical to the original
#        implementation. When `evaluation_date` is an explicit historical
#        date (the rolling call site), `prices` extends WELL past it (it's
#        the full current price history), so this distinction is the whole
#        difference between "a real edge-decay measurement" and "a chart
#        that quietly cheats."
# ---------------------------------------------------------------------------

def _resolve_buy_class_records(
    wide: pd.DataFrame,
    prices: dict[str, pd.Series],
    bench: pd.Series,
    horizon: str,
    is_new_buy,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    evaluation_date: pd.Timestamp | None = None,
    conviction_lookup: dict[tuple, Any] | None = None,
) -> list[dict]:
    snap_dates = list(wide.index)
    records: list[dict] = []

    for i, snap_date in enumerate(snap_dates):
        if window_start is not None and pd.Timestamp(snap_date) < window_start:
            continue
        if window_end is not None and pd.Timestamp(snap_date) > window_end:
            continue
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

            s_entry_i = s.index.searchsorted(snap_ts, side="left")
            b_entry_i = bench.index.searchsorted(snap_ts, side="left")
            if s_entry_i >= len(s) or b_entry_i >= len(bench):
                continue

            mark_to_eval = False
            if horizon == "1w":
                exit_ts = snap_ts + pd.Timedelta(days=7)
                if evaluation_date is not None and exit_ts > pd.Timestamp(evaluation_date):
                    exit_ts = pd.Timestamp(evaluation_date)
                    mark_to_eval = True
            else:  # next_state_exit
                exit_ts = None
                for j in range(i + 1, len(snap_dates)):
                    nxt_label = wide.loc[snap_dates[j], ticker]
                    if isinstance(nxt_label, str) and nxt_label not in (
                            "NEW_BUY", "HOLD_IF_LONG", "BUY"):
                        exit_ts = pd.Timestamp(snap_dates[j])
                        break
                if exit_ts is None:
                    if evaluation_date is not None:
                        exit_ts = pd.Timestamp(evaluation_date)
                        mark_to_eval = True
                    else:
                        exit_ts = s.index[-1]

            if mark_to_eval:
                # Last bar ON OR BEFORE evaluation_date — never later.
                s_exit_i = s.index.searchsorted(exit_ts, side="right") - 1
                b_exit_i = bench.index.searchsorted(exit_ts, side="right") - 1
            else:
                s_exit_i = s.index.searchsorted(exit_ts, side="left")
                b_exit_i = bench.index.searchsorted(exit_ts, side="left")

            if s_exit_i >= len(s):
                s_exit_i = len(s) - 1
            if b_exit_i >= len(bench):
                b_exit_i = len(bench) - 1
            if s_exit_i <= s_entry_i or b_exit_i <= b_entry_i or s_exit_i < 0 or b_exit_i < 0:
                continue

            s_e, s_x = float(s.iloc[s_entry_i]), float(s.iloc[s_exit_i])
            b_e, b_x = float(bench.iloc[b_entry_i]), float(bench.iloc[b_exit_i])
            if s_e == 0 or b_e == 0:
                continue
            excess = (s_x / s_e - 1.0) - (b_x / b_e - 1.0)
            hold_days = (s.index[s_exit_i] - s.index[s_entry_i]).days
            rec = {"as_of": snap_date, "ticker": ticker, "state": str(label),
                  "excess": excess, "hold_days": hold_days}
            if conviction_lookup is not None:
                rec["conviction"] = conviction_lookup.get((snap_date, ticker))
            records.append(rec)
    return records


def _wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95%-by-default Wilson score interval. Duplicated (deliberately, it's
    8 lines) from app.py's copy rather than sharing a utils module — matches
    this codebase's existing convention of small, independent pure modules
    (macro_alignment.py doesn't import from regime_analysis.py either)."""
    if n <= 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1.0 + (z ** 2) / n
    center = p + (z ** 2) / (2 * n)
    margin = z * ((p * (1 - p) / n + (z ** 2) / (4 * n ** 2)) ** 0.5)
    lo, hi = (center - margin) / denom, (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)


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

    if source == "history":
        snaps = pd.DataFrame()
    else:
        from src.db import load_signal_snapshots
        snaps = load_signal_snapshots()
        if snaps.empty and source == "snapshots":
            return {**empty, "source": "snapshots"}
    if not snaps.empty:
        wide = snaps.pivot(index="as_of", columns="ticker", values="state").sort_index()
        wide = wide.tail(weeks)
        source = "snapshots"
        is_new_buy = lambda v: isinstance(v, str) and v == "NEW_BUY"  # noqa: E731
    else:
        if history is None or history.empty or len(history) < 4:
            return empty
        wide = history.tail(weeks)
        source = "history"
        is_new_buy = lambda v: isinstance(v, str) and v in ("BUY", "NEW_BUY")  # noqa: E731

    records = _resolve_buy_class_records(wide, prices, bench, horizon, is_new_buy)

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


# ---------------------------------------------------------------------------
# S3 — rolling expectancy (edge-decay tracking)
# ---------------------------------------------------------------------------

def rolling_signal_performance(
    prices: dict[str, pd.Series],
    benchmark_ticker: str = "SPY",
    window_weeks: int = 12,
    step_weeks: int = 1,
    horizon: str = "next_state_exit",
    min_signals: int = 1,
) -> pd.DataFrame:
    """Trailing `window_weeks` hit-rate / mean-excess, recomputed at each of
    a series of historical evaluation points — the time series
    `signal_performance_vs_benchmark` doesn't provide (it only ever answers
    "as of right now"). This is what actually answers "is the edge currently
    working or currently decaying" (TRADING_EDGE_AUDIT.md item S3).

    Reads exclusively from the persisted `signal_snapshots` table — no
    raw-history fallback. Edge-decay tracking is only meaningful once real
    refined-state snapshots exist; replaying the mechanical-core-only
    history would measure a different, simpler system than the one that's
    actually live (same reasoning `BACKTEST_REPORT.md` already applies to
    the historical backtest).

    Returns a DataFrame indexed by evaluation date (`as_of`) with columns:
      n_signals, hit_rate, mean_excess_return, ci_lo, ci_hi
    (ci_lo/ci_hi: 95% Wilson interval on hit_rate, so a 3-signal point and a
    40-signal point don't read as equally confident on a chart). Empty
    DataFrame if fewer than 2 distinct snapshot dates exist yet.
    """
    empty_cols = ["n_signals", "hit_rate", "mean_excess_return", "ci_lo", "ci_hi"]
    from src.db import load_signal_snapshots
    snaps = load_signal_snapshots()
    if snaps.empty or benchmark_ticker not in prices:
        return pd.DataFrame(columns=empty_cols)
    bench = prices[benchmark_ticker].dropna()
    if bench.empty:
        return pd.DataFrame(columns=empty_cols)

    full_wide = snaps.pivot(index="as_of", columns="ticker", values="state").sort_index()
    is_new_buy = lambda v: isinstance(v, str) and v == "NEW_BUY"  # noqa: E731

    all_dates = list(full_wide.index)
    if len(all_dates) < 2:
        return pd.DataFrame(columns=empty_cols)

    eval_dates = all_dates[::max(1, step_weeks)]
    if eval_dates[-1] != all_dates[-1]:
        eval_dates.append(all_dates[-1])

    rows: list[dict] = []
    for E in eval_dates:
        E_ts = pd.Timestamp(E)
        window_start = E_ts - pd.Timedelta(weeks=window_weeks)
        # Causal slice: this evaluation point may not see snapshot dates
        # after itself, for state-transition purposes OR for price marking.
        wide_causal = full_wide[full_wide.index <= E_ts]
        records = _resolve_buy_class_records(
            wide_causal, prices, bench, horizon, is_new_buy,
            window_start=window_start, window_end=E_ts, evaluation_date=E_ts,
        )
        if len(records) < min_signals:
            continue
        arr = np.array([r["excess"] for r in records], dtype=float)
        hits = int((arr > 0).sum())
        n = int(arr.size)
        lo, hi = _wilson_ci(hits, n)
        rows.append({
            "as_of": E_ts, "n_signals": n,
            "hit_rate": float(hits / n),
            "mean_excess_return": float(arr.mean()),
            "ci_lo": lo, "ci_hi": hi,
        })

    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows).set_index("as_of")


# ---------------------------------------------------------------------------
# S4 — conviction calibration
# ---------------------------------------------------------------------------

def performance_by_conviction(
    prices: dict[str, pd.Series],
    benchmark_ticker: str = "SPY",
    weeks: int = 52,
    horizon: str = "next_state_exit",
) -> pd.DataFrame:
    """Historical hit-rate / mean-excess return, bucketed by the conviction
    score each NEW_BUY signal carried AT ENTRY (TRADING_EDGE_AUDIT.md S4).

    This is the check the 0-5 conviction score has never had: is a 5-dot
    signal actually better than a 1-dot signal, or is the score just a
    plausible-looking prior nobody's verified against realized outcomes?

    Reads from `signal_snapshots` only — conviction doesn't exist in the raw
    mechanical-core replay (`history` frames carry only BUY/HOLD/SELL
    labels), so there is no fallback source here, unlike
    `signal_performance_vs_benchmark`.

    Returns a DataFrame indexed by conviction score (0-5, only scores that
    actually occurred in the sample) with columns: n_signals, hit_rate,
    mean_excess_return, ci_lo, ci_hi. Empty if no snapshots exist yet.
    """
    empty_cols = ["n_signals", "hit_rate", "mean_excess_return", "ci_lo", "ci_hi"]
    from src.db import load_signal_snapshots
    snaps = load_signal_snapshots()
    if snaps.empty or benchmark_ticker not in prices:
        return pd.DataFrame(columns=empty_cols).rename_axis("conviction")
    bench = prices[benchmark_ticker].dropna()
    if bench.empty:
        return pd.DataFrame(columns=empty_cols).rename_axis("conviction")

    wide = snaps.pivot(index="as_of", columns="ticker", values="state").sort_index()
    wide = wide.tail(weeks)
    conv_lookup = {
        (row["as_of"], row["ticker"]): row["conviction"]
        for _, row in snaps.iterrows()
    }
    is_new_buy = lambda v: isinstance(v, str) and v == "NEW_BUY"  # noqa: E731

    records = _resolve_buy_class_records(
        wide, prices, bench, horizon, is_new_buy, conviction_lookup=conv_lookup,
    )
    records = [r for r in records if r.get("conviction") is not None
              and not pd.isna(r.get("conviction"))]
    if not records:
        return pd.DataFrame(columns=empty_cols).rename_axis("conviction")

    df = pd.DataFrame(records)
    rows: list[dict] = []
    for conv, sub in df.groupby("conviction"):
        arr = sub["excess"].to_numpy(dtype=float)
        hits = int((arr > 0).sum())
        n = int(arr.size)
        lo, hi = _wilson_ci(hits, n)
        rows.append({"conviction": int(conv), "n_signals": n,
                    "hit_rate": float(hits / n),
                    "mean_excess_return": float(arr.mean()),
                    "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).set_index("conviction").sort_index()
