from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import BreakoutParams
from src.breakout_signals import compute_breakout_signal

BUY_CLASS = {"BUY", "HIGH_CONVICTION_BUY"}


def weekly_rebalance_dates(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each ISO week -- same definition src/backtest.py
    already uses elsewhere in this codebase."""
    s = pd.Series(idx, index=idx)
    iso = s.groupby([idx.isocalendar().year, idx.isocalendar().week]).last()
    return list(pd.DatetimeIndex(iso.values))


@dataclass
class BreakoutBacktestConfig:
    cost_bps: float = 5.0
    cash_buffer: float = 0.10
    initial_capital: float = 100_000.0
    max_positions: int = 8
    params: BreakoutParams = field(default_factory=BreakoutParams)


@dataclass
class BreakoutBacktestResult:
    equity: pd.Series
    benchmark_equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def _rank_candidates(signals: dict[str, object], max_positions: int) -> list[str]:
    """BUY-class tickers, highest conviction first; ties broken by money
    flow state. Caps the book at max_positions so a wide simultaneous
    breakout across a noisy universe doesn't over-diversify into mush."""
    buy_class = [(t, s) for t, s in signals.items() if s.signal in BUY_CLASS]
    order = {"Strong Accumulation": 0, "Accumulation": 1, "Neutral": 2,
             "Distribution": 3, "Strong Distribution": 4}
    buy_class.sort(key=lambda ts: (-ts[1].conviction, order.get(ts[1].money_flow_state, 2)))
    return [t for t, _ in buy_class[:max_positions]]


def run_breakout_backtest(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    cfg: BreakoutBacktestConfig = BreakoutBacktestConfig(),
) -> BreakoutBacktestResult:
    """Weekly-cadence backtest of the independent breakout signal.

    No look-ahead: at each rebalance date `rb`, every ticker's signal is
    computed on `df.loc[:rb]` -- a causal slice, exactly the discipline
    src/backtest.py already applies to the sentiment-gated model.

    Equity is marked WEEKLY (at each rebalance date), not daily -- adequate
    for answering "does this signal have edge", and avoids re-implementing
    src/backtest.py's full daily mark-to-market machinery for a first-pass
    validation harness. A production-grade version should upgrade to daily
    marking before being trusted for live sizing.
    """
    all_dates = sorted(set().union(*(df.index for df in ohlcv_by_ticker.values())))
    idx = pd.DatetimeIndex(all_dates)
    rebal_dates = weekly_rebalance_dates(idx)

    shares: dict[str, float] = {}
    cash = cfg.initial_capital
    equity_records: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    cost_rate = cfg.cost_bps / 10_000.0

    def _last_close_on_or_before(df: pd.DataFrame, d: pd.Timestamp) -> float | None:
        sub = df.loc[:d]
        if sub.empty:
            return None
        return float(sub["close"].iloc[-1])

    for rb in rebal_dates:
        signals = {}
        for tkr, df in ohlcv_by_ticker.items():
            # Bound the slice to a generous trailing window rather than the
            # full history-to-date. compute_breakout_signal only ever reads
            # trailing bars (ATR baseline 252d + BBW lookback 252d + 60d
            # minimum + slop), so feeding it the ENTIRE growing history on
            # every rebalance recomputes the same rolling stats over an
            # ever-larger, mostly-irrelevant window -- this is the actual
            # cost driver in a weekly loop, not the signal math itself.
            window_start = rb - pd.Timedelta(days=700)
            sliced = df.loc[window_start:rb]
            if sliced.empty:
                continue
            signals[tkr] = compute_breakout_signal(tkr, sliced, params=cfg.params)

        target_tickers = _rank_candidates(signals, cfg.max_positions)
        per_name_weight = (1.0 - cfg.cash_buffer) / len(target_tickers) if target_tickers else 0.0

        # Mark-to-market NAV as of this rebalance date, using last available close.
        nav = cash
        for tkr, qty in shares.items():
            px = _last_close_on_or_before(ohlcv_by_ticker[tkr], rb)
            if px:
                nav += qty * px

        held = set(shares.keys())
        target_set = set(target_tickers)

        # Exit anything no longer BUY-class (includes explicit RISK_EXIT).
        for tkr in sorted(held - target_set):
            px = _last_close_on_or_before(ohlcv_by_ticker[tkr], rb)
            if px is None or shares[tkr] <= 0:
                continue
            notional = shares[tkr] * px
            cost = abs(notional) * cost_rate
            cash += notional - cost
            trades.append({"date": rb, "ticker": tkr, "side": "SELL",
                           "notional": notional, "cost": cost,
                           "reason": signals.get(tkr).signal if tkr in signals else "no_data"})
            shares[tkr] = 0.0

        # Enter new BUY-class names.
        for tkr in sorted(target_set - held):
            px = _last_close_on_or_before(ohlcv_by_ticker[tkr], rb)
            if px is None:
                continue
            want = nav * per_name_weight
            if want < 1.0:
                continue
            qty = want / px
            cost = want * cost_rate
            cash -= want + cost
            shares[tkr] = qty
            trades.append({"date": rb, "ticker": tkr, "side": "BUY",
                           "notional": want, "cost": cost,
                           "reason": signals[tkr].signal})

        shares = {t: q for t, q in shares.items() if q > 1e-9}
        post_nav = cash + sum(
            shares[t] * (_last_close_on_or_before(ohlcv_by_ticker[t], rb) or 0.0)
            for t in shares
        )
        equity_records.append((rb, post_nav))

    equity = pd.Series(dict(equity_records)).sort_index()
    equity.name = "breakout_strategy"

    bench = benchmark_close.reindex(equity.index, method="ffill")
    bench = (bench / bench.iloc[0]) * cfg.initial_capital
    bench.name = "benchmark"

    trades_df = pd.DataFrame(trades)
    years = (equity.index[-1] - equity.index[0]).days / 365.25 or 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0
    bench_cagr = (bench.iloc[-1] / bench.iloc[0]) ** (1 / years) - 1.0
    dd = (equity / equity.cummax() - 1.0).min()
    total_costs = float(trades_df["cost"].sum()) if not trades_df.empty else 0.0

    stats = {
        "window_start": str(equity.index[0].date()), "window_end": str(equity.index[-1].date()),
        "cagr": float(cagr), "benchmark_cagr": float(bench_cagr),
        "excess_cagr": float(cagr - bench_cagr), "max_drawdown": float(dd),
        "n_trades": int(len(trades_df)), "total_costs": total_costs,
        "final_equity": float(equity.iloc[-1]),
    }
    return BreakoutBacktestResult(equity=equity, benchmark_equity=bench,
                                   trades=trades_df, stats=stats)
