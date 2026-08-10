"""Tests for src.money_flow -- CMF, OBV, and the accumulation/
distribution classifier. Synthetic OHLCV series, no DB, no network.
"""

import numpy as np
import pandas as pd
from src.money_flow import (
    chaikin_money_flow, on_balance_volume, obv_trend_slope, classify_money_flow,
    MoneyFlowThresholds,
)


def _ohlcv(n, close_at_high=True, vol=1000.0, trend="up"):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    if trend == "up":
        close = np.linspace(100, 100 + n * 0.5, n)
    else:
        close = np.full(n, 100.0)
    high = close + 1.0
    low = close - 1.0
    c = high if close_at_high else low
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": c, "volume": vol}, index=idx)


def test_cmf_all_closes_at_high_gives_plus_one():
    df = _ohlcv(40, close_at_high=True)
    cmf = chaikin_money_flow(df, period=20)
    assert abs(cmf.iloc[-1] - 1.0) < 1e-9, cmf.iloc[-1]


def test_cmf_all_closes_at_low_gives_minus_one():
    df = _ohlcv(40, close_at_high=False)
    cmf = chaikin_money_flow(df, period=20)
    assert abs(cmf.iloc[-1] - (-1.0)) < 1e-9, cmf.iloc[-1]


def test_obv_monotonic_up_accumulates_full_volume():
    df = _ohlcv(30, trend="up")
    obv = on_balance_volume(df)
    # every day close rises -> obv should be cumulative sum of volume (1000/day)
    assert abs(obv.iloc[-1] - 1000.0 * 29) < 1e-6, obv.iloc[-1]


def test_obv_slope_normalized_is_near_one_for_steady_uptrend():
    df = _ohlcv(120, trend="up", vol=1000.0)
    slope = obv_trend_slope(df, period=14, vol_norm_period=63)
    # raw slope should be ~1000/day (every day adds full volume), avg vol 1000
    # -> normalized ~1.0
    assert 0.9 < slope.iloc[-1] < 1.1, slope.iloc[-1]


def test_obv_slope_is_scale_invariant_across_volume_levels():
    """The whole point of normalization: a thin stock and a mega-cap ETF in
    the identical PATTERN should produce the same normalized slope."""
    thin = _ohlcv(120, trend="up", vol=500.0)
    mega = _ohlcv(120, trend="up", vol=5_000_000.0)
    s_thin = obv_trend_slope(thin, period=14, vol_norm_period=63).iloc[-1]
    s_mega = obv_trend_slope(mega, period=14, vol_norm_period=63).iloc[-1]
    assert abs(s_thin - s_mega) < 1e-6, (s_thin, s_mega)


def test_classify_money_flow_strong_accumulation():
    t = MoneyFlowThresholds()
    assert classify_money_flow(0.20, 0.20, t) == "Strong Accumulation"


def test_classify_money_flow_mixed_signals_are_muted_not_strong():
    """One leg strongly positive, other strongly negative -> must NOT read
    as 'Strong' anything -- the legs should net out toward Neutral."""
    t = MoneyFlowThresholds()
    result = classify_money_flow(0.20, -0.20, t)
    assert result == "Neutral", result


def test_classify_money_flow_missing_data_is_neutral_not_a_guess():
    assert classify_money_flow(None, 0.2) == "Neutral"
    assert classify_money_flow(float("nan"), 0.2) == "Neutral"


def test_classify_money_flow_strong_distribution():
    t = MoneyFlowThresholds()
    assert classify_money_flow(-0.20, -0.20, t) == "Strong Distribution"

