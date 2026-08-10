"""End-to-end tests for src.breakout_backtest, including a full-system
regression test mirroring this repo's existing
test_backtest_no_lookahead_signal_only_uses_prior_bars pattern.
"""

import numpy as np
import pandas as pd
from src.breakout_backtest import run_breakout_backtest, BreakoutBacktestConfig
from src.breakout_signals import BreakoutParams


def _ohlcv_from_close(close: np.ndarray, start="2019-01-02", vol=1_000_000.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(close), freq="B")
    close = pd.Series(close, index=idx)
    high = close + np.abs(np.diff(close, prepend=close.iloc[0])) * 0.6 + 0.3
    low = close - np.abs(np.diff(close, prepend=close.iloc[0])) * 0.6 - 0.3
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def _clean_breakout_ticker(rng, n_total=900):
    """Long rising warmup -> tight consolidation -> clean breakout ->
    sustained follow-through -- should be entered and ridden."""
    warm = np.cumsum(rng.normal(0.10, 1.0, n_total - 260))
    warm = warm - warm[-1] + 100
    cons = 100 + rng.normal(0, 0.15, 180)
    trend = 100 + np.linspace(0, 40, 80) + rng.normal(0, 0.5, 80)
    return _ohlcv_from_close(np.concatenate([warm, cons, trend]))


def _choppy_ticker(rng, n_total=900):
    return _ohlcv_from_close(100 + np.cumsum(rng.normal(0, 1.3, n_total)))


def _decline_ticker(rng, n_total=900):
    """Grinds down for most of the window -- if ever bought, must be exited
    promptly via RISK_EXIT rather than ridden to the bottom."""
    return _ohlcv_from_close(np.linspace(140, 60, n_total) + rng.normal(0, 1.0, n_total))


def _build_universe(seed=11):
    rng = np.random.default_rng(seed)
    return {
        "WINNER1": _clean_breakout_ticker(np.random.default_rng(seed + 1)),
        "WINNER2": _clean_breakout_ticker(np.random.default_rng(seed + 2)),
        "CHOP1": _choppy_ticker(np.random.default_rng(seed + 3)),
        "CHOP2": _choppy_ticker(np.random.default_rng(seed + 4)),
        "LOSER1": _decline_ticker(np.random.default_rng(seed + 5)),
        "LOSER2": _decline_ticker(np.random.default_rng(seed + 6)),
    }


def test_backtest_runs_end_to_end_and_produces_a_dated_equity_curve():
    universe = _build_universe()
    bench = pd.Series(100.0, index=universe["WINNER1"].index)  # flat benchmark
    res = run_breakout_backtest(universe, bench, BreakoutBacktestConfig(
        params=BreakoutParams(atr_compression_threshold=0.85, bbw_percentile_threshold=0.35)))
    assert not res.equity.empty
    assert res.equity.index.is_monotonic_increasing
    assert res.equity.notna().all()


def test_strategy_avoids_riding_decliners_to_the_bottom():
    """The capital-preservation property that actually matters: a
    buy-and-hold of the SAME universe (including the two decliners) should
    lose materially more than the strategy, which is supposed to exit on
    RISK_EXIT rather than holding through a SMA150 breakdown."""
    universe = _build_universe()
    bench = pd.Series(100.0, index=universe["WINNER1"].index)
    res = run_breakout_backtest(universe, bench, BreakoutBacktestConfig(
        params=BreakoutParams(atr_compression_threshold=0.85, bbw_percentile_threshold=0.35)))

    # Naive equal-weight buy-and-hold of the whole universe from day 1,
    # for comparison -- this is what "no risk management at all" looks like.
    closes = pd.DataFrame({t: df["close"] for t, df in universe.items()}).dropna()
    bh_rets = closes.pct_change().mean(axis=1).fillna(0.0)
    bh_curve = (1 + bh_rets).cumprod() * 100_000.0

    assert res.stats["max_drawdown"] > bh_curve.pct_change().add(1).cumprod().pipe(
        lambda s: (s / s.cummax() - 1).min())


def test_strategy_participates_in_at_least_one_clean_winner():
    universe = _build_universe()
    bench = pd.Series(100.0, index=universe["WINNER1"].index)
    res = run_breakout_backtest(universe, bench, BreakoutBacktestConfig(
        params=BreakoutParams(atr_compression_threshold=0.85, bbw_percentile_threshold=0.35)))
    bought = set(res.trades.loc[res.trades["side"] == "BUY", "ticker"]) if not res.trades.empty else set()
    assert bought & {"WINNER1", "WINNER2"}, (
        f"strategy never bought either clean-breakout ticker; trades={res.trades}")


def test_no_lookahead_truncating_the_dataset_never_changes_earlier_trades():
    """The full-system regression test: run the backtest on the full
    universe, then again on a version truncated 200 trading days earlier.
    Every trade dated on-or-before the truncation point must be IDENTICAL
    between the two runs -- if truncating the future changed a past
    decision, that decision was leaking information from ahead of itself.
    """
    universe = _build_universe()
    bench_full = pd.Series(100.0, index=universe["WINNER1"].index)
    cfg = BreakoutBacktestConfig(
        params=BreakoutParams(atr_compression_threshold=0.85, bbw_percentile_threshold=0.35))

    res_full = run_breakout_backtest(universe, bench_full, cfg)

    cutoff = universe["WINNER1"].index[-200]
    universe_trunc = {t: df.loc[:cutoff] for t, df in universe.items()}
    bench_trunc = bench_full.loc[:cutoff]
    res_trunc = run_breakout_backtest(universe_trunc, bench_trunc, cfg)

    full_before_cutoff = res_full.trades[res_full.trades["date"] <= cutoff].reset_index(drop=True)
    trunc_all = res_trunc.trades.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        full_before_cutoff.sort_values(["date", "ticker"]).reset_index(drop=True),
        trunc_all.sort_values(["date", "ticker"]).reset_index(drop=True),
    )

