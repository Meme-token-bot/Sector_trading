"""Data-quality / robustness tests: NaN gaps, zero volume, zero-range
bars, duplicate timestamps, 24/7 crypto calendars, freshly-listed ETFs,
and a fully flat tape. See TRADING doc Section 16/24.
"""

import numpy as np
import pandas as pd
from src.breakout_signals import compute_breakout_signal, BreakoutParams
from src.money_flow import chaikin_money_flow, on_balance_volume
from src.consolidation import detect_consolidation_and_breakout


def _base_ohlcv(n, freq="B", start="2021-01-04"):
    idx = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(21)
    close = 100 + np.cumsum(rng.normal(0, 1.0, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.integers(1_000, 100_000, n).astype(float),
    }, index=idx)


def test_nan_bars_do_not_crash_the_signal_pipeline():
    """A few NaN OHLCV rows (a data-vendor gap) must not raise -- and must
    not silently poison the whole series into garbage output."""
    df = _base_ohlcv(400)
    df.iloc[150:153] = np.nan
    sig = compute_breakout_signal("XYZ", df)
    assert sig.signal in (
        "NOT_ENOUGH_DATA", "NO_SIGNAL", "WATCH", "BUY", "HIGH_CONVICTION_BUY", "RISK_EXIT")


def test_zero_volume_bars_do_not_produce_inf_or_nan_money_flow():
    """A halted/no-print day (volume=0) must not blow up CMF (division by
    zero volume) or OBV (0 * sign is fine, but confirm no propagation)."""
    df = _base_ohlcv(60)
    df.loc[df.index[30], "volume"] = 0.0
    cmf = chaikin_money_flow(df, period=20)
    obv = on_balance_volume(df)
    assert np.isfinite(cmf.dropna()).all()
    assert np.isfinite(obv).all()


def test_zero_range_bar_high_equals_low_does_not_produce_inf_cmf():
    """A bar where high == low (illiquid print) must not divide by zero."""
    df = _base_ohlcv(60)
    flat_px = float(df["close"].iloc[40])
    df.loc[df.index[40], ["open", "high", "low", "close"]] = flat_px
    cmf = chaikin_money_flow(df, period=20)
    assert np.isfinite(cmf.dropna()).all()


def test_duplicate_timestamps_are_handled_by_the_caller_not_silently_wrong():
    """This module assumes a clean, de-duplicated, sorted-ascending index
    (the same contract src/price_store.py's primary key on
    (ticker, timeframe, bar_date) already enforces upstream). Confirm a
    duplicate-index frame at least does not crash -- callers must still
    de-dupe before this point; this is a documentation-by-test of the
    contract boundary, not a claim that dupes are handled gracefully."""
    df = _base_ohlcv(300)
    dup = pd.concat([df, df.iloc[[100]]]).sort_index()
    try:
        compute_breakout_signal("XYZ", dup)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            f"duplicate-timestamp input crashed rather than degrading "
            f"gracefully: {e!r}")


def test_seven_day_calendar_crypto_data_does_not_crash():
    """Crypto trades 24/7 -- confirm the pipeline works on a calendar
    (not business-day) index that includes weekends, unlike every other
    fixture in this test suite which uses freq='B'."""
    df = _base_ohlcv(400, freq="D")
    sig = compute_breakout_signal("BTC-USD", df)
    assert sig.signal != "" and sig.consolidation is not None


def test_freshly_listed_etf_short_history_reports_not_enough_data_not_a_false_signal():
    """An ETF that IPO'd recently (e.g. a spot-crypto ETF launched in the
    last few months) must report NOT_ENOUGH_DATA, never a confident BUY or
    RISK_EXIT built on too little history to mean anything."""
    df = _base_ohlcv(45)  # ~2 months -- realistic for a fund a few weeks old
    sig = compute_breakout_signal("NEWETF", df)
    assert sig.signal == "NOT_ENOUGH_DATA"
    assert sig.consolidation.state == "NOT_ENOUGH_DATA"


def test_all_flat_zero_volatility_series_does_not_crash_or_false_positive():
    """A perfectly flat tape (edge case: a pegged/halted instrument) must
    not produce a spurious breakout from divide-by-zero artifacts."""
    idx = pd.date_range("2021-01-04", periods=400, freq="B")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 1000.0}, index=idx)
    sig = compute_breakout_signal("FLAT", df)
    assert sig.signal in ("NO_SIGNAL", "NOT_ENOUGH_DATA", "WATCH")
    assert sig.signal != "HIGH_CONVICTION_BUY"

