"""End-to-end tests for src.breakout_signals -- the signal hierarchy,
including the RISK_EXIT-overrides-BUY ordering guarantee.
"""

import numpy as np
import pandas as pd
from src.breakout_signals import compute_breakout_signal, BreakoutParams


def _flat_ohlcv(n, level=100.0, vol=10_000.0):
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame({"open": level, "high": level + 0.5, "low": level - 0.5,
                         "close": level, "volume": vol}, index=idx)


def _series(closes, highs=None, lows=None, vol=10_000.0):
    n = len(closes)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float) if highs is not None else closes + 0.4
    lows = np.asarray(lows, dtype=float) if lows is not None else closes - 0.4
    return pd.DataFrame({"open": closes, "high": highs, "low": lows,
                         "close": closes, "volume": vol}, index=idx)


def test_not_enough_data():
    sig = compute_breakout_signal("XYZ", _flat_ohlcv(50))
    assert sig.signal == "NOT_ENOUGH_DATA"
    assert sig.conviction == 0


def test_no_signal_on_a_directionless_choppy_series():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 1.5, 250))
    sig = compute_breakout_signal("XYZ", _series(closes))
    assert sig.signal in ("NO_SIGNAL", "WATCH", "RISK_EXIT")  # never a bare BUY on noise alone


def test_risk_exit_when_price_below_sma150():
    # Long decline -- price ends up well below its own SMA150.
    closes = np.linspace(150, 90, 250)
    sig = compute_breakout_signal("XYZ", _series(closes))
    assert sig.signal == "RISK_EXIT", sig.signal
    assert any("below SMA150" in r for r in sig.risk_flags), sig.risk_flags
    assert sig.conviction == 0


def test_risk_exit_when_sma150_declining_even_if_price_above_it():
    # Choppy-but-declining SMA150 with price whipping slightly above it.
    rng = np.random.default_rng(2)
    trend = np.linspace(140, 100, 200)
    noise = rng.normal(0, 0.5, 200)
    closes = trend + noise
    sig = compute_breakout_signal("XYZ", _series(closes))
    assert sig.sma_150_slope_state == "Falling"
    assert sig.signal == "RISK_EXIT"


def test_high_conviction_buy_a_few_days_after_a_clean_breakout_with_follow_through():
    rng = np.random.default_rng(3)
    warm = np.cumsum(rng.normal(0.15, 1.0, 260))         # rising warmup
    warm = warm - warm[-1] + 100                          # anchor last warmup close at 100
    cons = 100 + rng.normal(0, 0.15, 65)                   # tight sideways
    # Breakout bar plus 5 days of follow-through so SMA50 has time to
    # actually bend upward -- a single breakout candle alone cannot move a
    # 50-day average enough, over any reasonable slope-lookback window, to
    # read "Rising" on the very day it prints. That lag is real and correct
    # behaviour, not a bug; the test must reflect it.
    follow_through = np.array([112.0, 113.5, 115.0, 116.0, 117.5, 119.0])
    closes = np.concatenate([warm, cons, follow_through])
    sig = compute_breakout_signal(
        "XYZ", _series(closes),
        params=BreakoutParams(atr_compression_threshold=0.85, bbw_percentile_threshold=0.35),
    )
    assert sig.consolidation.state == "DETECTED", sig.consolidation
    assert sig.consolidation.breakout is True
    assert sig.sma_50_slope_state == "Rising", sig.sma_50_slope_state
    assert sig.signal == "HIGH_CONVICTION_BUY", (sig.signal, sig.signal_reasons, sig.risk_flags)
    assert sig.conviction >= 3


def test_breakout_flag_does_not_vanish_the_day_after_it_fires():
    """The specific bug this regression test exists to catch: without a
    short follow-through lookback window, `breakout` would only be True on
    the single bar where the range first broke and disappear immediately
    after -- useless for a dashboard reviewed weekly."""
    rng = np.random.default_rng(3)
    warm = np.cumsum(rng.normal(0.15, 1.0, 260))
    warm = warm - warm[-1] + 100
    cons = 100 + rng.normal(0, 0.15, 65)
    follow_through = np.array([112.0, 112.3])   # breakout bar + ONE quiet day after
    closes = np.concatenate([warm, cons, follow_through])
    sig = compute_breakout_signal(
        "XYZ", _series(closes),
        params=BreakoutParams(atr_compression_threshold=0.85, bbw_percentile_threshold=0.35),
    )
    assert sig.consolidation.breakout is True, (
        "breakout flag disappeared one day after firing -- lookback window regressed")
    assert sig.consolidation.days_since_breakout == 1


def test_watch_when_consolidating_but_not_yet_broken_out():
    rng = np.random.default_rng(4)
    warm = np.cumsum(rng.normal(0, 1.2, 260))
    warm = warm - warm[-1] + 100
    cons = 100 + rng.normal(0, 0.15, 65)   # still inside the range, no breakout bar appended
    closes = np.concatenate([warm, cons])
    sig = compute_breakout_signal(
        "XYZ", _series(closes),
        params=BreakoutParams(atr_compression_threshold=0.85, bbw_percentile_threshold=0.35),
    )
    assert sig.signal == "WATCH", (sig.signal, sig.consolidation)


def test_risk_exit_overrides_what_would_otherwise_be_a_buy():
    """The critical ordering guarantee: a breakout inside a LONG-TERM
    downtrend (price still below a falling SMA150) must NEVER be reported
    as BUY / HIGH_CONVICTION_BUY just because a short-term range broke."""
    rng = np.random.default_rng(5)
    decline = np.linspace(160, 95, 260)                 # sma150 falling, price low
    cons = 95 + rng.normal(0, 0.1, 65)                    # tight range at the bottom
    breakout_bar = np.array([98.0])                       # breaks the local range...
    closes = np.concatenate([decline, cons, breakout_bar])
    sig = compute_breakout_signal(
        "XYZ", _series(closes),
        params=BreakoutParams(atr_compression_threshold=0.9, bbw_percentile_threshold=0.4),
    )
    assert sig.sma_150_slope_state == "Falling"
    assert sig.signal == "RISK_EXIT", (
        f"got {sig.signal} -- a falling SMA150 must override any breakout BUY signal")

