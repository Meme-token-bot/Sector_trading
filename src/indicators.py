"""Hand-rolled technical indicators.

Pure pandas/numpy. No streamlit imports, no DB access, no side effects.
Inputs are a Close `pd.Series` (DatetimeIndex). Outputs are aligned to that
index so callers can paste indicator columns straight onto an OHLCV frame.

Kept intentionally small — RSI, MACD, Bollinger, SMA. Other indicators
(Stochastic, ATR, etc.) are out of scope for the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average. Result is NaN until `period` observations seen."""
    if period <= 0:
        raise ValueError("period must be positive")
    return close.rolling(window=period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Uses an exponential (Wilder) smoothing of gains/losses
    with alpha = 1/period, which is the canonical formulation.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # Where there were zero losses, RSI = 100 by convention.
    out = out.where(avg_loss != 0, 100.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """Moving-average convergence/divergence.

    Returns a DataFrame with columns: macd, signal, histogram.
    EMAs use `adjust=False` (the standard form used by trading platforms).
    """
    if not (0 < fast < slow):
        raise ValueError("require 0 < fast < slow")
    if signal <= 0:
        raise ValueError("signal must be positive")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": hist,
    }, index=close.index)


def bollinger(close: pd.Series, period: int = 20,
              num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Population standard deviation (ddof=0), matching the
    original Bollinger formulation. Returns columns: middle, upper, lower.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    return pd.DataFrame({
        "middle": mid,
        "upper": mid + num_std * std,
        "lower": mid - num_std * std,
    }, index=close.index)
