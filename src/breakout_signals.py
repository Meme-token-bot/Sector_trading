"""Unified breakout / consolidation / money-flow signal -- a SEPARATE
layer from src/signals.py's sentiment-gated sector convergence model.

Why separate: src/signals.py::build_signals requires newsletter sentiment
>= PARAMS.buy_sentiment_threshold to ever emit BUY. There is no newsletter
coverage for the expanded universe (dual equity ETF proxies, metals,
crypto), and routing this signal through that gate would make every one
of those tickers permanently un-buyable regardless of price action. This
module does not touch build_signals, refine_signals, target_weights, or
signal_snapshots, and does not require sentiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import BREAKOUT, MONEY_FLOW, BreakoutParams, MoneyFlowThresholds
from src.consolidation import (
    ConsolidationResult, detect_consolidation_and_breakout, sma_slope_state,
)
from src.indicators import sma
from src.money_flow import chaikin_money_flow, classify_money_flow, obv_trend_slope

SIGNAL_HIERARCHY = ("NOT_ENOUGH_DATA", "NO_SIGNAL", "WATCH", "BUY",
                     "HIGH_CONVICTION_BUY", "RISK_EXIT")


@dataclass(frozen=True)
class BreakoutSignal:
    ticker: str
    signal: str
    signal_reasons: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)
    conviction: int = 0
    price: float | None = None
    sma_50: float | None = None
    sma_150: float | None = None
    sma_50_slope_state: str = "NOT_ENOUGH_DATA"
    sma_150_slope_state: str = "NOT_ENOUGH_DATA"
    consolidation: ConsolidationResult = None
    cmf: float | None = None
    obv_slope: float | None = None
    money_flow_state: str = "Neutral"


def compute_breakout_signal(
    ticker: str, ohlcv: pd.DataFrame,
    params: BreakoutParams = BREAKOUT,
    money_flow_thresholds: MoneyFlowThresholds = MONEY_FLOW,
) -> BreakoutSignal:
    reasons, risk = [], []
    min_bars = params.sma_long + params.slope_lookback + 5
    if ohlcv is None or ohlcv.empty or len(ohlcv) < min_bars:
        return BreakoutSignal(
            ticker=ticker, signal="NOT_ENOUGH_DATA",
            signal_reasons=[f"only {0 if ohlcv is None else len(ohlcv)} bars stored, "
                             f"need >= {min_bars} for a SMA{params.sma_long} regime read"],
            consolidation=ConsolidationResult("NOT_ENOUGH_DATA", 0, None, None, False, None, None),
        )

    close = ohlcv["close"]
    price = float(close.iloc[-1])
    sma50_s, sma150_s = sma(close, params.sma_short), sma(close, params.sma_long)
    sma50 = float(sma50_s.iloc[-1]) if pd.notna(sma50_s.iloc[-1]) else None
    sma150 = float(sma150_s.iloc[-1]) if pd.notna(sma150_s.iloc[-1]) else None
    _, sma50_state = sma_slope_state(sma50_s, params.slope_lookback, params.slope_deadband_pct_per_day)
    _, sma150_state = sma_slope_state(sma150_s, params.slope_lookback, params.slope_deadband_pct_per_day)

    cons = detect_consolidation_and_breakout(
        ohlcv, min_days=params.consolidation_min_days,
        atr_compression_threshold=params.atr_compression_threshold,
        bbw_percentile_threshold=params.bbw_percentile_threshold,
        atr_period=params.atr_period, atr_baseline_period=params.atr_baseline_period,
        bb_period=params.bb_period, bb_lookback=params.bb_lookback,
        require=params.compression_require, breakout_lookback_days=params.breakout_lookback_days,
    )

    cmf_s = chaikin_money_flow(ohlcv, params.cmf_period)
    obv_s = obv_trend_slope(ohlcv, params.obv_period, params.obv_vol_norm_period)
    cmf = float(cmf_s.iloc[-1]) if pd.notna(cmf_s.iloc[-1]) else None
    obv_slope = float(obv_s.iloc[-1]) if pd.notna(obv_s.iloc[-1]) else None
    mf_state = classify_money_flow(cmf, obv_slope, money_flow_thresholds)

    risk_exit = False
    if sma150 is not None and price < sma150:
        risk_exit = True
        risk.append(f"Price ${price:,.2f} is below SMA{params.sma_long} (${sma150:,.2f})")
    if sma150_state == "Falling":
        risk_exit = True
        risk.append(f"SMA{params.sma_long} is declining")

    if cons.state == "NOT_ENOUGH_DATA":
        signal = "NOT_ENOUGH_DATA"
        reasons.append(f"Insufficient history for a {params.consolidation_min_days}-day consolidation range")
    elif cons.breakout and sma50_state == "Rising":
        reasons.append(f"{cons.consolidation_days}-day consolidation "
                        f"(${cons.consolidation_low:,.2f}-${cons.consolidation_high:,.2f})"
                        + (f", broke out {cons.days_since_breakout}d ago" if cons.days_since_breakout else ""))
        reasons.append(f"Price above consolidation high (${cons.consolidation_high:,.2f})")
        reasons.append(f"SMA{params.sma_short} rising")
        if mf_state in ("Accumulation", "Strong Accumulation"):
            reasons.append(f"Money flow confirms: {mf_state}")
        elif mf_state in ("Distribution", "Strong Distribution"):
            risk.append(f"Money flow diverges from price: {mf_state}")
        if sma150 is not None and price > sma150:
            signal = "HIGH_CONVICTION_BUY"
            reasons.append(f"Price above SMA{params.sma_long} (trend-regime confirmed)")
        else:
            signal = "BUY"
    elif cons.state == "DETECTED" and not cons.breakout:
        signal = "WATCH"
        reasons.append(f"{cons.consolidation_days}-day consolidation in progress "
                        f"(${cons.consolidation_low:,.2f}-${cons.consolidation_high:,.2f}) -- no breakout yet")
    else:
        signal = "NO_SIGNAL"
        reasons.append("No qualifying consolidation/breakout setup right now")

    if risk_exit:
        signal = "RISK_EXIT"
        reasons = [f"(prior setup) {r}" for r in reasons]

    conviction = sum([
        cons.state == "DETECTED", cons.breakout, sma50_state == "Rising",
        sma150 is not None and price > sma150,
        mf_state in ("Accumulation", "Strong Accumulation"),
    ])
    if signal == "RISK_EXIT":
        conviction = 0

    return BreakoutSignal(
        ticker=ticker, signal=signal, signal_reasons=reasons, risk_flags=risk,
        conviction=int(conviction), price=price, sma_50=sma50, sma_150=sma150,
        sma_50_slope_state=sma50_state, sma_150_slope_state=sma150_state,
        consolidation=cons, cmf=cmf, obv_slope=obv_slope, money_flow_state=mf_state,
    )


def relative_strength_rank(returns_by_ticker: dict[str, float]) -> dict[str, float]:
    """Percentile rank (0..1) of each ticker's return within the supplied
    set -- e.g. trailing 60-day % return across the whole expanded
    universe. Use rank, not raw return, when comparing across asset
    classes with very different volatility regimes."""
    if not returns_by_ticker:
        return {}
    s = pd.Series(returns_by_ticker)
    return s.rank(pct=True).to_dict()
