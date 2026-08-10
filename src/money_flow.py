from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import MoneyFlowThresholds, MONEY_FLOW


def chaikin_money_flow(ohlcv: pd.DataFrame, period: int = 20) -> pd.Series:
    high, low, close, vol = (ohlcv["high"], ohlcv["low"], ohlcv["close"],
                              ohlcv["volume"].astype(float))
    rng = (high - low)
    mfm = ((close - low) - (high - close)) / rng.replace(0.0, np.nan)
    mfm = mfm.fillna(0.0)
    mfv = mfm * vol
    cmf = (mfv.rolling(period, min_periods=period).sum() /
           vol.rolling(period, min_periods=period).sum().replace(0.0, np.nan))
    return cmf


def on_balance_volume(ohlcv: pd.DataFrame) -> pd.Series:
    close, vol = ohlcv["close"], ohlcv["volume"].astype(float)
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * vol).cumsum()


def obv_trend_slope(ohlcv: pd.DataFrame, period: int = 14,
                     vol_norm_period: int = 63) -> pd.Series:
    obv = on_balance_volume(ohlcv)
    raw_slope = (obv - obv.shift(period)) / period
    avg_vol = ohlcv["volume"].astype(float).rolling(
        vol_norm_period, min_periods=max(5, vol_norm_period // 4)).mean()
    return raw_slope / avg_vol.replace(0.0, np.nan)


def classify_money_flow(cmf, obv_slope_norm,
                         thresholds: MoneyFlowThresholds = MONEY_FLOW
                         ) -> str:
    if cmf is None or obv_slope_norm is None or pd.isna(cmf) or pd.isna(obv_slope_norm):
        return "Neutral"

    def _vote(x, strong, mild):
        if x >= strong:
            return 2
        if x >= mild:
            return 1
        if x <= -strong:
            return -2
        if x <= -mild:
            return -1
        return 0

    score = (_vote(cmf, thresholds.cmf_strong, thresholds.cmf_mild) +
             _vote(obv_slope_norm, thresholds.obv_slope_strong, thresholds.obv_slope_mild))
    if score >= 3:
        return "Strong Accumulation"
    if score >= 1:
        return "Accumulation"
    if score <= -3:
        return "Strong Distribution"
    if score <= -1:
        return "Distribution"
    return "Neutral"
