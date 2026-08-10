"""Tests for src.consolidation -- the highest-risk module (look-ahead
bias, breakout timing). Synthetic OHLCV series, no DB, no network.
"""

import numpy as np
import pandas as pd
from src.consolidation import (
    detect_consolidation_and_breakout, atr_compression_ratio,
    bollinger_width_percentile, sma_slope_state,
)
from src.indicators import sma


def _build_series(n_warmup=300, n_consolidation=70, breakout=True):
    """warmup (real, NON-PERIODIC volatility -- a random walk with no
    natural quiet trough, so the regime boundary is unambiguous) -> tight
    consolidation -> optional breakout day with a deliberately extreme
    high/low WICK, to make any 'evaluation bar leaked into its own
    boundary' bug obvious."""
    rng = np.random.default_rng(42)

    steps = rng.normal(0, 1.2, n_warmup)
    warm_close = 100 + np.cumsum(steps) - np.cumsum(steps)[-1] + 0  # re-based below
    warm_close = 100 + np.cumsum(steps)
    warm_close = warm_close - warm_close[-1] + 100  # anchor the LAST warmup close at 100
    warm_high = warm_close + 1.5 + rng.normal(0, 0.1, n_warmup).clip(min=0)
    warm_low = warm_close - 1.5 - rng.normal(0, 0.1, n_warmup).clip(min=0)

    t2 = np.arange(n_consolidation)
    base = warm_close[-1]
    cons_close = base + 0.5 * np.sin(2 * np.pi * t2 / 10)
    cons_high = cons_close + 0.3
    cons_low = cons_close - 0.3

    closes = np.concatenate([warm_close, cons_close])
    highs = np.concatenate([warm_high, cons_high])
    lows = np.concatenate([warm_low, cons_low])

    if breakout:
        # Deliberately extreme wick on the breakout bar itself.
        closes = np.append(closes, base + 15.0)     # clear close-above-range breakout
        highs = np.append(highs, base + 30.0)        # huge wick -- must NOT leak into the range
        lows = np.append(lows, base - 5.0)            # huge wick -- must NOT leak into the range

    n = len(closes)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    vol = np.full(n, 10_000.0)
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows,
                       "close": closes, "volume": vol}, index=idx)
    return df, n_warmup, n_consolidation


def test_not_enough_data_short_series():
    df, *_ = _build_series(n_warmup=300, n_consolidation=70, breakout=True)
    short = df.iloc[:100]
    res = detect_consolidation_and_breakout(short, min_days=60)
    assert res.state == "NOT_ENOUGH_DATA"
    assert res.consolidation_days == 0
    assert res.breakout is False


def test_consolidation_detected_before_breakout_no_false_early_trigger():
    df, n_warmup, n_cons = _build_series(n_warmup=300, n_consolidation=70, breakout=True)
    # Truncate to the LAST day still inside consolidation (one day before breakout).
    day_before = df.iloc[:n_warmup + n_cons]     # excludes the breakout row entirely
    res = detect_consolidation_and_breakout(day_before, min_days=60)
    assert res.state == "DETECTED", res
    assert res.consolidation_days >= 60, res.consolidation_days
    assert res.breakout is False, "must not fire early -- no breakout bar exists in this slice"


def test_breakout_fires_on_the_correct_day():
    df, n_warmup, n_cons = _build_series(n_warmup=300, n_consolidation=70, breakout=True)
    at_breakout = df.iloc[:n_warmup + n_cons + 1]   # includes the breakout row as the LAST row
    res = detect_consolidation_and_breakout(at_breakout, min_days=60)
    assert res.state == "DETECTED", res
    assert res.breakout is True, res


def test_breakout_bar_does_not_leak_into_its_own_boundary():
    """THE critical look-ahead-adjacent guard: the breakout day's own
    extreme high/low wick (set to base+30 / base-5 in the fixture) must
    NEVER appear in consolidation_high/consolidation_low -- those must be
    derived purely from the 70 days STRICTLY BEFORE the breakout bar."""
    df, n_warmup, n_cons = _build_series(n_warmup=300, n_consolidation=70, breakout=True)
    at_breakout = df.iloc[:n_warmup + n_cons + 1]
    res = detect_consolidation_and_breakout(at_breakout, min_days=60)
    base = df["close"].iloc[n_warmup - 1]
    wick_high, wick_low = base + 30.0, base - 5.0
    # True consolidation range is roughly [base-0.3, base+0.8]. Allow a
    # generous margin for legitimate boundary fuzziness (a real regime
    # transition is never perfectly crisp), but this must stay an order of
    # magnitude away from the wick to prove no leak occurred.
    assert res.consolidation_high < base + 8.0, (
        f"consolidation_high={res.consolidation_high} is suspiciously close to "
        f"the breakout bar's own wick high of {wick_high} -- possible leak")
    assert res.consolidation_low > base - 8.0, (
        f"consolidation_low={res.consolidation_low} is suspiciously close to "
        f"the breakout bar's own wick low of {wick_low} -- possible leak")
    # The sharpest possible check: the wick itself must be strictly outside
    # the reported range by a wide margin.
    assert res.consolidation_high < wick_high - 15.0
    assert res.consolidation_low > wick_low + 3.0


def test_consolidation_days_grows_monotonically_as_more_bars_are_fed():
    df, n_warmup, n_cons = _build_series(n_warmup=300, n_consolidation=70, breakout=False)
    counts = []
    for extra in range(55, 71, 5):
        sl = df.iloc[:n_warmup + extra]
        res = detect_consolidation_and_breakout(sl, min_days=60)
        counts.append(res.consolidation_days)
    assert counts == sorted(counts), counts


def test_slicing_more_future_history_never_changes_an_earlier_evaluation():
    """Regression guard mirroring the existing repo's
    test_backtest_no_lookahead_signal_only_uses_prior_bars pattern: the
    result AS OF a given day must be identical whether or not more rows
    exist after it in the un-sliced source -- i.e. callers who properly
    slice `ohlcv` up to `as_of` get a stable answer."""
    df, n_warmup, n_cons = _build_series(n_warmup=300, n_consolidation=70, breakout=True)
    eval_end = n_warmup + n_cons + 1
    short_slice = df.iloc[:eval_end]

    # Simulate "more data becomes available later" by appending 80 more
    # days of a violent, unrelated move AFTER the evaluation point, then
    # re-slicing back to the same evaluation point.
    extra_idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=80, freq="B")
    spike = pd.DataFrame({
        "open": 500.0, "high": 520.0, "low": 480.0, "close": 500.0,
        "volume": 10_000.0,
    }, index=extra_idx)
    extended = pd.concat([df, spike])
    resliced = extended.iloc[:eval_end]

    res_short = detect_consolidation_and_breakout(short_slice, min_days=60)
    res_resliced = detect_consolidation_and_breakout(resliced, min_days=60)
    assert res_short == res_resliced


def test_sma_slope_state_classifies_rising_flat_falling():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    rising = sma(pd.Series(np.linspace(100, 130, 40), index=idx), 10)
    falling = sma(pd.Series(np.linspace(130, 100, 40), index=idx), 10)
    flat = sma(pd.Series(np.full(40, 100.0), index=idx), 10)
    _, s1 = sma_slope_state(rising, lookback=10)
    _, s2 = sma_slope_state(falling, lookback=10)
    _, s3 = sma_slope_state(flat, lookback=10)
    assert s1 == "Rising", s1
    assert s2 == "Falling", s2
    assert s3 == "Flat", s3


def test_sma_slope_state_not_enough_data():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    s = sma(pd.Series(np.linspace(100, 105, 5), index=idx), 3)
    _, state = sma_slope_state(s, lookback=10)
    assert state == "NOT_ENOUGH_DATA"

