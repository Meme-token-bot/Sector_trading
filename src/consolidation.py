from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.indicators import atr as _atr, bollinger as _bollinger


def atr_compression_ratio(ohlcv: pd.DataFrame, atr_period: int = 14,
                           baseline_period: int = 252) -> pd.Series:
    atr_pct = _atr(ohlcv, atr_period) / ohlcv["close"]
    baseline = atr_pct.rolling(baseline_period,
                                min_periods=max(20, baseline_period // 4)).mean()
    return atr_pct / baseline.replace(0.0, np.nan)


def bollinger_width_percentile(ohlcv: pd.DataFrame, bb_period: int = 20,
                                lookback: int = 252) -> pd.Series:
    bb = _bollinger(ohlcv["close"], period=bb_period)
    width = (bb["upper"] - bb["lower"]) / bb["middle"].replace(0.0, np.nan)
    return width.rolling(lookback, min_periods=max(30, lookback // 4)).rank(pct=True)


@dataclass(frozen=True)
class ConsolidationResult:
    state: str                        # NOT_ENOUGH_DATA | NONE | DETECTED
    consolidation_days: int
    consolidation_high: float | None
    consolidation_low: float | None
    breakout: bool
    atr_compression_ratio: float | None
    bbw_percentile: float | None
    days_since_breakout: int | None = None


def _compressed_series(ohlcv, atr_compression_threshold, bbw_percentile_threshold,
                        atr_period, atr_baseline_period, bb_period, bb_lookback, require):
    """Compute the ATR/BBW compression boolean series ONCE over the full
    frame. Rolling/EWM windows are causal by construction -- a value at
    index i depends only on data at indices <= i -- so this single pass is
    reusable for every truncation point a caller wants to evaluate, instead
    of recomputing rolling stats from scratch per evaluation point (this
    used to be an ~11x redundant cost in the lookback-window scan below,
    and was the dominant cost in a weekly backtest loop)."""
    acr = atr_compression_ratio(ohlcv, atr_period, atr_baseline_period)
    bbw_pct = bollinger_width_percentile(ohlcv, bb_period, bb_lookback)
    atr_ok = acr <= atr_compression_threshold
    bbw_ok = bbw_pct <= bbw_percentile_threshold
    if require == "atr":
        compressed = atr_ok
    elif require == "bbw":
        compressed = bbw_ok
    elif require == "both":
        compressed = atr_ok & bbw_ok
    else:
        compressed = atr_ok | bbw_ok
    return compressed.fillna(False).to_numpy(), acr, bbw_pct


def _run_length_ending_at(compressed: np.ndarray, end_idx: int) -> int:
    """Consecutive True count in `compressed` walking backward from
    `end_idx` (inclusive). Plain numpy indexing -- no pandas overhead."""
    n = 0
    i = end_idx
    while i >= 0 and compressed[i]:
        n += 1
        i -= 1
    return n


def _scan_from_compressed(ohlcv, compressed, acr, bbw_pct, min_days,
                           eval_idx: int) -> ConsolidationResult:
    """Evaluate 'is bar `eval_idx` a breakout' using an ALREADY-COMPUTED
    compressed/acr/bbw_pct series. The boundary is built strictly from
    bars before `eval_idx` -- this is the look-ahead-safety invariant."""
    prior_end = eval_idx - 1
    last_acr = float(acr.iloc[eval_idx]) if pd.notna(acr.iloc[eval_idx]) else None
    last_bbw = float(bbw_pct.iloc[eval_idx]) if pd.notna(bbw_pct.iloc[eval_idx]) else None
    if prior_end < 0:
        return ConsolidationResult("NONE", 0, None, None, False, last_acr, last_bbw)

    n = _run_length_ending_at(compressed, prior_end)
    if n == 0:
        return ConsolidationResult("NONE", 0, None, None, False, last_acr, last_bbw)

    window_start = prior_end - n + 1
    high, low, close = (ohlcv["high"].to_numpy(), ohlcv["low"].to_numpy(),
                         ohlcv["close"].to_numpy())
    cons_high = float(np.max(high[window_start:prior_end + 1]))
    cons_low = float(np.min(low[window_start:prior_end + 1]))
    eval_close = float(close[eval_idx])
    detected = n >= min_days
    breakout = bool(detected and eval_close > cons_high)
    return ConsolidationResult(
        state="DETECTED" if detected else "NONE",
        consolidation_days=n, consolidation_high=cons_high, consolidation_low=cons_low,
        breakout=breakout, atr_compression_ratio=last_acr, bbw_percentile=last_bbw,
    )


def detect_consolidation_and_breakout(
    ohlcv: pd.DataFrame,
    min_days: int = 60,
    atr_compression_threshold: float = 0.75,
    bbw_percentile_threshold: float = 0.25,
    atr_period: int = 14,
    atr_baseline_period: int = 252,
    bb_period: int = 20,
    bb_lookback: int = 252,
    require: str = "either",
    breakout_lookback_days: int = 10,
) -> ConsolidationResult:
    """`ohlcv` must be sorted ascending with the evaluation bar as the LAST
    row; every boundary is computed from rows strictly before the bar that
    tests against it (see `_scan_from_compressed`).

    A breakout is reported ACTIVE if the most recent qualifying
    (>= min_days) compression run ended within the trailing
    `breakout_lookback_days` bars AND today's close is still above that
    run's FROZEN high (price hasn't fallen back into the range since).
    Without this window, `breakout` would only ever be True on the single
    calendar day the range first broke -- useless for a dashboard reviewed
    weekly, since the flag would already be gone by the next visit even
    though the trade setup is still live. The consolidation boundary itself
    is never recomputed after the fact; only whether price still clears it
    is re-checked with fresh data.
    """
    n = len(ohlcv)
    needed = (atr_period + max(20, atr_baseline_period // 4) +
              max(30, bb_lookback // 4) + min_days + 5)
    if n < max(2, needed):
        return ConsolidationResult("NOT_ENOUGH_DATA", 0, None, None, False, None, None)

    compressed, acr, bbw_pct = _compressed_series(
        ohlcv, atr_compression_threshold, bbw_percentile_threshold,
        atr_period, atr_baseline_period, bb_period, bb_lookback, require)

    today_close = float(ohlcv["close"].iloc[-1])
    today_read = _scan_from_compressed(ohlcv, compressed, acr, bbw_pct, min_days, n - 1)

    best, best_k = None, None
    for k in range(0, min(breakout_lookback_days, n - 1) + 1):
        probe = (today_read if k == 0 else
                 _scan_from_compressed(ohlcv, compressed, acr, bbw_pct, min_days, n - 1 - k))
        if probe.state == "DETECTED" and probe.breakout:
            best, best_k = probe, k
            break   # k=0,1,2.. is most-recent-first; first hit wins

    if best is not None and today_close > best.consolidation_high:
        return ConsolidationResult(
            state="DETECTED", consolidation_days=best.consolidation_days,
            consolidation_high=best.consolidation_high,
            consolidation_low=best.consolidation_low, breakout=True,
            atr_compression_ratio=today_read.atr_compression_ratio,
            bbw_percentile=today_read.bbw_percentile,
            days_since_breakout=best_k,
        )

    # No currently-active breakout -- fall back to today's own read (still
    # useful for a WATCH state if a compression regime is live right now).
    return today_read


def sma_slope_state(sma_series: pd.Series, lookback: int = 10,
                     deadband_pct_per_day: float = 0.0005):
    s = sma_series.dropna()
    if len(s) < lookback + 1:
        return None, "NOT_ENOUGH_DATA"
    start, end = float(s.iloc[-1 - lookback]), float(s.iloc[-1])
    if start == 0:
        return None, "NOT_ENOUGH_DATA"
    slope = (end / start - 1.0) / lookback
    if slope > deadband_pct_per_day:
        return slope, "Rising"
    if slope < -deadband_pct_per_day:
        return slope, "Falling"
    return slope, "Flat"
