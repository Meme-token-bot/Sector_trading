"""Tests for the new S3 (rolling expectancy) and S4 (conviction calibration)
functions in src.signal_history. `load_signal_snapshots` is patched at its
source (`src.db.load_signal_snapshots`) since signal_history.py imports it
locally at call time — same pattern the project already uses elsewhere
(e.g. tests/test_weekly_recap.py patching `_build_macro_snapshots`).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.signal_history import (
    performance_by_conviction,
    rolling_signal_performance,
    signal_performance_vs_benchmark,
)


def _bdays(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _snap_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a signal_snapshots-shaped DataFrame (as load_signal_snapshots
    would return it — as_of as Timestamp, one row per (as_of, ticker))."""
    df = pd.DataFrame(rows)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


# ---------------------------------------------------------------------------
# rolling_signal_performance — the no-lookahead proof
# ---------------------------------------------------------------------------

def test_rolling_performance_does_not_leak_future_prices():
    """The one property this function cannot get wrong: a rolling point
    evaluated 'as of' a past date must not be able to see price action that
    happens after that date, even though `prices` (the full current series)
    obviously contains it.

    Construction: XLK is flat 100 through day 100, then jumps 5x on day 101
    and stays there. A NEW_BUY entry at day 90 never transitions away (still
    "open" at every subsequent snapshot). Evaluated ROLLING as of day 97
    (before the spike), the position must be marked to day-97's price, not
    to the actual latest price in the series (which includes the spike) —
    so the measured excess return must be ~flat, not a 5x windfall.
    """
    idx = _bdays("2026-01-01", 200)
    xlk = pd.Series(100.0, index=idx)
    xlk.iloc[101:] = 500.0  # 5x jump on day 101, stays there
    spy = pd.Series(100.0, index=idx)
    prices = {"XLK": xlk, "SPY": spy}

    entry_day = idx[90]
    later_day = idx[97]  # still before the day-101 spike

    snaps = _snap_frame([
        {"as_of": entry_day, "ticker": "XLK", "state": "NEW_BUY", "conviction": 3},
        {"as_of": later_day, "ticker": "XLK", "state": "NEW_BUY", "conviction": 3},
    ])

    with patch("src.db.load_signal_snapshots", return_value=snaps):
        rolling = rolling_signal_performance(prices, benchmark_ticker="SPY",
                                             window_weeks=52, step_weeks=1)

    assert not rolling.empty
    # The evaluation point at `later_day` is what matters here.
    row = rolling.loc[pd.Timestamp(later_day)]
    # Flat-to-flat over this window (both entry and mark-to-eval prices are
    # 100, pre-spike) — excess must be small, NOT anywhere near the ~400%
    # windfall a lookahead bug would produce.
    assert abs(row["mean_excess_return"]) < 0.02

    # Contrast: the "as of right now" function (evaluation_date=None,
    # correctly marks to the ACTUAL last bar, which is post-spike) SHOULD
    # see the windfall. This proves the two code paths are doing something
    # genuinely different, not both accidentally landing on ~0.
    now_result = signal_performance_vs_benchmark(
        pd.DataFrame(), prices, benchmark_ticker="SPY", weeks=52,
    )
    # (uses snapshots via the patched loader too, source='auto')
    with patch("src.db.load_signal_snapshots", return_value=snaps):
        now_result = signal_performance_vs_benchmark(
            pd.DataFrame(), prices, benchmark_ticker="SPY", weeks=52,
        )
    assert now_result["mean_excess_return"] > 1.0  # the 5x windfall shows up


def test_rolling_performance_window_excludes_old_entries():
    """A NEW_BUY entry far outside the trailing window shouldn't contribute
    to a later evaluation point's stats."""
    idx = _bdays("2026-01-01", 400)
    xlk = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    spy = pd.Series(100.0, index=idx)
    prices = {"XLK": xlk, "SPY": spy}

    old_entry = idx[10]
    recent_entry = idx[300]
    eval_point = idx[310]

    snaps = _snap_frame([
        {"as_of": old_entry, "ticker": "XLK", "state": "NEW_BUY", "conviction": 2},
        {"as_of": recent_entry, "ticker": "XLK", "state": "NEW_BUY", "conviction": 4},
        {"as_of": eval_point, "ticker": "XLK", "state": "NEW_BUY", "conviction": 4},
    ])
    with patch("src.db.load_signal_snapshots", return_value=snaps):
        rolling = rolling_signal_performance(prices, benchmark_ticker="SPY",
                                             window_weeks=4, step_weeks=1)
    row = rolling.loc[pd.Timestamp(eval_point)]
    # Only the recent_entry should count (old_entry is >4 weeks before
    # eval_point) — plus the degenerate same-day entry at eval_point itself
    # (which contributes nothing, per the general resolution rules).
    assert row["n_signals"] <= 2


def test_rolling_performance_empty_with_fewer_than_two_snapshot_dates():
    idx = _bdays("2026-01-01", 50)
    prices = {"XLK": pd.Series(100.0, index=idx), "SPY": pd.Series(100.0, index=idx)}
    snaps = _snap_frame([{"as_of": idx[5], "ticker": "XLK", "state": "NEW_BUY", "conviction": 3}])
    with patch("src.db.load_signal_snapshots", return_value=snaps):
        rolling = rolling_signal_performance(prices)
    assert rolling.empty


def test_rolling_performance_ci_contains_point_estimate():
    idx = _bdays("2026-01-01", 300)
    xlk = pd.Series(np.linspace(100, 140, len(idx)), index=idx)
    spy = pd.Series(100.0, index=idx)
    prices = {"XLK": xlk, "SPY": spy}
    dates = [idx[20 * i] for i in range(6)]
    snaps = _snap_frame([
        {"as_of": d, "ticker": "XLK", "state": "NEW_BUY", "conviction": 3}
        for d in dates
    ])
    with patch("src.db.load_signal_snapshots", return_value=snaps):
        rolling = rolling_signal_performance(prices, window_weeks=52)
    assert not rolling.empty
    for _, row in rolling.iterrows():
        assert row["ci_lo"] <= row["hit_rate"] <= row["ci_hi"]


# ---------------------------------------------------------------------------
# performance_by_conviction
# ---------------------------------------------------------------------------

def test_conviction_buckets_are_separated_correctly():
    """Two tickers, deliberately engineered so conviction=5 entries are all
    winners and conviction=1 entries are all losers — the calibration table
    must actually separate them, not blend everything into one bucket."""
    idx = _bdays("2026-01-01", 120)
    winner = pd.Series(np.linspace(100, 130, len(idx)), index=idx)  # up a lot
    loser = pd.Series(np.linspace(100, 80, len(idx)), index=idx)    # down a lot
    spy = pd.Series(100.0, index=idx)
    prices = {"WIN": winner, "LOSE": loser, "SPY": spy}

    d1, d2 = idx[10], idx[15]
    snaps = _snap_frame([
        {"as_of": d1, "ticker": "WIN",  "state": "NEW_BUY", "conviction": 5},
        {"as_of": d2, "ticker": "LOSE", "state": "NEW_BUY", "conviction": 1},
        # A later snapshot so the entries actually get "resolved" against
        # forward price action rather than left open at evaluation="now".
        {"as_of": idx[100], "ticker": "WIN", "state": "SELL", "conviction": 0},
        {"as_of": idx[100], "ticker": "LOSE", "state": "SELL", "conviction": 0},
    ])
    with patch("src.db.load_signal_snapshots", return_value=snaps):
        table = performance_by_conviction(prices, weeks=52)

    assert 5 in table.index
    assert 1 in table.index
    assert table.loc[5, "mean_excess_return"] > 0
    assert table.loc[1, "mean_excess_return"] < 0
    assert table.loc[5, "hit_rate"] == pytest.approx(1.0)
    assert table.loc[1, "hit_rate"] == pytest.approx(0.0)


def test_conviction_table_empty_when_no_snapshots():
    with patch("src.db.load_signal_snapshots", return_value=pd.DataFrame()):
        table = performance_by_conviction({"SPY": pd.Series([100.0])})
    assert table.empty


def test_conviction_table_skips_records_with_missing_conviction():
    idx = _bdays("2026-01-01", 60)
    prices = {"XLK": pd.Series(100.0, index=idx), "SPY": pd.Series(100.0, index=idx)}
    snaps = _snap_frame([
        {"as_of": idx[5], "ticker": "XLK", "state": "NEW_BUY", "conviction": None},
    ])
    with patch("src.db.load_signal_snapshots", return_value=snaps):
        table = performance_by_conviction(prices)
    assert table.empty


def test_conviction_table_ci_contains_point_estimate():
    idx = _bdays("2026-01-01", 150)
    xlk = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    spy = pd.Series(100.0, index=idx)
    prices = {"XLK": xlk, "SPY": spy}
    snaps = _snap_frame([
        {"as_of": idx[5], "ticker": "XLK", "state": "NEW_BUY", "conviction": 4},
        {"as_of": idx[10], "ticker": "XLK", "state": "NEW_BUY", "conviction": 4},
        {"as_of": idx[100], "ticker": "XLK", "state": "SELL", "conviction": 0},
    ])
    with patch("src.db.load_signal_snapshots", return_value=snaps):
        table = performance_by_conviction(prices, weeks=52)
    for _, row in table.iterrows():
        assert row["ci_lo"] <= row["hit_rate"] <= row["ci_hi"]
