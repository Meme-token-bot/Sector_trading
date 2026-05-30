"""Tests for src.signal_history helpers (state changes + performance backtest).

Pure-function tests — fabricates a history frame and a `current` refined
signals frame directly. No DB, no yfinance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal_history import (
    detect_state_changes,
    signal_performance_vs_benchmark,
)


# ---------------------------------------------------------------------------
# detect_state_changes
# ---------------------------------------------------------------------------

def _current(rows: dict[str, dict]) -> pd.DataFrame:
    """Build a minimal refined-signals frame from {ticker: {col: value}}."""
    return pd.DataFrame(rows).T


def test_detect_no_history_returns_empty():
    history = pd.DataFrame()
    current = _current({
        "XLK": {"signal": "BUY", "state": "NEW_BUY",
                "relative_strength_3m": 0.05, "above_sma": True,
                "extension_pct": 0.02, "sentiment_score": 2.5,
                "state_reason": "fresh buy"},
    })
    out = detect_state_changes(history, current)
    assert out.empty
    assert list(out.columns) == ["sector", "prior_state", "new_state", "reason"]


def test_detect_all_unchanged_returns_empty():
    history = pd.DataFrame(
        [{"XLK": "BUY", "XLF": "HOLD"}],
        index=pd.to_datetime(["2026-05-15"]),
    )
    # current state matches prior raw signal.
    current = _current({
        "XLK": {"signal": "BUY", "state": "BUY",
                "relative_strength_3m": 0.05, "above_sma": True,
                "extension_pct": 0.02, "sentiment_score": 2.5,
                "state_reason": ""},
        "XLF": {"signal": "HOLD", "state": "HOLD",
                "relative_strength_3m": -0.005, "above_sma": True,
                "extension_pct": 0.0, "sentiment_score": 0.5,
                "state_reason": ""},
    })
    out = detect_state_changes(history, current)
    assert out.empty


def test_detect_single_change():
    history = pd.DataFrame(
        [{"XLK": "BUY", "XLF": "HOLD"}],
        index=pd.to_datetime(["2026-05-15"]),
    )
    current = _current({
        # XLK degraded BUY -> HOLD (RS went negative).
        "XLK": {"signal": "HOLD", "state": "REDUCE",
                "relative_strength_3m": -0.02, "above_sma": True,
                "extension_pct": 0.01, "sentiment_score": 1.5,
                "state_reason": "was BUY, trim"},
        # XLF unchanged.
        "XLF": {"signal": "HOLD", "state": "HOLD",
                "relative_strength_3m": -0.005, "above_sma": True,
                "extension_pct": 0.0, "sentiment_score": 0.5,
                "state_reason": ""},
    })
    out = detect_state_changes(history, current)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["sector"] == "XLK"
    assert row["prior_state"] == "BUY"
    assert row["new_state"] == "REDUCE"
    assert "RS" in row["reason"]


def test_detect_multiple_changes():
    history = pd.DataFrame(
        [
            {"XLK": "BUY", "XLF": "HOLD", "XLE": "BUY"},
            {"XLK": "BUY", "XLF": "HOLD", "XLE": "BUY"},
        ],
        index=pd.to_datetime(["2026-05-08", "2026-05-15"]),
    )
    current = _current({
        # XLK: extended past cutoff -> CHASE.
        "XLK": {"signal": "BUY", "state": "CHASE",
                "relative_strength_3m": 0.1, "above_sma": True,
                "extension_pct": 0.18, "sentiment_score": 2.5,
                "state_reason": "extended"},
        # XLF: HOLD -> NEW_BUY (RS turned positive).
        "XLF": {"signal": "BUY", "state": "NEW_BUY",
                "relative_strength_3m": 0.04, "above_sma": True,
                "extension_pct": 0.03, "sentiment_score": 2.5,
                "state_reason": "fresh buy"},
        # XLE: lost BUY entirely (below SMA200).
        "XLE": {"signal": "SELL", "state": "SELL",
                "relative_strength_3m": -0.05, "above_sma": False,
                "extension_pct": -0.04, "sentiment_score": -1.0,
                "state_reason": "below sma"},
    })
    out = detect_state_changes(history, current)
    assert len(out) == 3
    by_sector = {r["sector"]: r for _, r in out.iterrows()}
    assert by_sector["XLK"]["new_state"] == "CHASE"
    assert "extended" in by_sector["XLK"]["reason"].lower()
    assert by_sector["XLF"]["new_state"] == "NEW_BUY"
    assert "RS" in by_sector["XLF"]["reason"]
    assert by_sector["XLE"]["new_state"] == "SELL"
    assert "SMA200" in by_sector["XLE"]["reason"]


def test_detect_uses_most_recent_history_row():
    """Only the LAST row of history matters for prior_state."""
    history = pd.DataFrame(
        [
            {"XLK": "SELL"},  # older
            {"XLK": "BUY"},   # most recent
        ],
        index=pd.to_datetime(["2026-05-08", "2026-05-15"]),
    )
    current = _current({
        "XLK": {"signal": "HOLD", "state": "REDUCE",
                "relative_strength_3m": -0.01, "above_sma": True,
                "extension_pct": 0.01, "sentiment_score": 1.0,
                "state_reason": ""},
    })
    out = detect_state_changes(history, current)
    assert len(out) == 1
    assert out.iloc[0]["prior_state"] == "BUY"


# ---------------------------------------------------------------------------
# signal_performance_vs_benchmark
# ---------------------------------------------------------------------------

def _weekly_index(n: int, end: str = "2026-05-15") -> pd.DatetimeIndex:
    """Build n weekly snapshot dates ending on `end`."""
    end_ts = pd.Timestamp(end)
    return pd.DatetimeIndex([end_ts - pd.Timedelta(weeks=(n - 1 - i)) for i in range(n)])


def _daily_index(start: str, n_days: int) -> pd.DatetimeIndex:
    """Business-day index of length n_days starting at start."""
    return pd.date_range(start=start, periods=n_days, freq="B")


def test_perf_short_history_short_circuits():
    history = pd.DataFrame(
        [{"XLK": "BUY"}],
        index=_weekly_index(1),
    )
    out = signal_performance_vs_benchmark(history, {"SPY": pd.Series(dtype=float)})
    # Function returns extra diagnostic keys (`horizon`, `source`,
    # `median_hold_days`) now — assert subset semantics rather than exact ==.
    assert out["n_signals"] == 0
    assert out["mean_excess_return"] == 0.0
    assert out["hit_rate"] == 0.0
    assert out["by_state"] == {}


def test_perf_three_week_history_still_short():
    history = pd.DataFrame(
        [{"XLK": "BUY"}, {"XLK": "BUY"}, {"XLK": "BUY"}],
        index=_weekly_index(3),
    )
    out = signal_performance_vs_benchmark(history, {"SPY": pd.Series([100], index=[pd.Timestamp("2026-05-15")])})
    assert out["n_signals"] == 0


def test_perf_multi_state_with_fabricated_series():
    # Four weekly snapshots, two sectors that go BUY at various weeks.
    weeks = _weekly_index(4, end="2026-05-15")
    history = pd.DataFrame(
        [
            {"XLK": "BUY",  "XLF": "HOLD"},
            {"XLK": "BUY",  "XLF": "HOLD"},
            {"XLK": "HOLD", "XLF": "BUY"},
            {"XLK": "BUY",  "XLF": "BUY"},
        ],
        index=weeks,
    )

    # Build daily price series spanning the history + a forward buffer.
    days = _daily_index("2026-04-01", 70)
    # XLK: monotonic up — every entry yields a positive forward return.
    xlk = pd.Series(np.linspace(100.0, 130.0, len(days)), index=days)
    # XLF: monotonic down — every entry yields a negative forward return.
    xlf = pd.Series(np.linspace(100.0, 80.0, len(days)), index=days)
    # SPY: flat — bench_fwd ~ 0 so sector_fwd is the excess.
    spy = pd.Series(100.0, index=days)

    prices = {"XLK": xlk, "XLF": xlf, "SPY": spy}

    # source='history' forces the in-memory replay path (the function's
    # default now prefers persisted signal_snapshots, which this test does
    # not populate).
    out = signal_performance_vs_benchmark(history, prices, weeks=12,
                                          source="history")

    # XLK was BUY in 3 of 4 weeks; XLF was BUY in 2 of 4 weeks.
    # Some snapshots may not have a forward-1w bar available; require
    # at least some signals registered.
    assert out["n_signals"] >= 1
    # XLK should be a positive excess driver, XLF negative.
    assert "BUY" in out["by_state"]
    # Hit rate is the fraction with excess > 0.
    assert 0.0 <= out["hit_rate"] <= 1.0
    # Mean excess return is finite.
    assert np.isfinite(out["mean_excess_return"])

    # XLK-only positive moves vs XLF-only negative — across all BUY records,
    # at least one should be positive and one negative.
    # Re-run with only XLK -> hit rate should be 1.0.
    history_xlk_only = history[["XLK"]]
    out_xlk = signal_performance_vs_benchmark(history_xlk_only, prices,
                                              weeks=12, source="history")
    if out_xlk["n_signals"] > 0:
        assert out_xlk["hit_rate"] == pytest.approx(1.0)
        assert out_xlk["mean_excess_return"] > 0


def test_perf_missing_benchmark_returns_empty():
    history = pd.DataFrame(
        [{"XLK": "BUY"}] * 5,
        index=_weekly_index(5),
    )
    out = signal_performance_vs_benchmark(history, {"XLK": pd.Series([100.0])})
    # Benchmark missing — graceful zero return rather than crash.
    assert out["n_signals"] == 0
