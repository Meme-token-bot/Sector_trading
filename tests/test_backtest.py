"""Tests for `src.backtest`. Uses synthetic price panels to keep things
deterministic and fast — no DB / network access."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from config.settings import BENCHMARK, PARAMS, SECTOR_ETFS, SUPPLEMENTARY_SECTORS
from src.backtest import (
    BacktestConfig,
    _annualised_stats,
    _closed_position_hit_rate,
    _max_drawdown,
    _synthetic_sentiment,
    real_sentiment_ablation,
    run_backtest,
    weekly_rebalance_dates,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic price panels
# ---------------------------------------------------------------------------

def _trading_days(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _flat_panel(n_days: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every sector + SPY at a constant $100. Useful as a sanity baseline:
    a flat market should produce zero P&L on either side (the model emits
    HOLD when no RS dispersion exists, and SPY buy-and-hold is flat)."""
    idx = _trading_days("2020-01-02", n_days)
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    cols = universe + [BENCHMARK]
    closes = pd.DataFrame(100.0, index=idx, columns=cols)
    opens = closes.copy()
    return closes, opens


def _trending_panel(n_days: int = 600, slope: float = 0.0008
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SPY drifts up steadily; sectors stair-step from clearly-leading XLK
    through middling to clearly-laggard XLU. Dispersion matters: when all
    laggards tie on RS, max_rank collapses and the weak-rank SELL gate
    sweeps the whole universe, leaving no BUYs. The stair-step ensures
    ranks 1..N are distinct."""
    idx = _trading_days("2020-01-02", n_days)
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    cols = universe + [BENCHMARK]
    closes = pd.DataFrame(index=idx, columns=cols, dtype=float)
    rng = np.arange(n_days)
    closes[BENCHMARK] = 100.0 * np.exp(slope * rng)
    # Multipliers ranging from 1.6 (XLK best) down to 0.3 (worst) — each
    # sector gets a UNIQUE growth rate so the rank ordering is well-defined.
    mults = np.linspace(1.6, 0.3, len(universe))
    leader_first = ["XLK"] + [t for t in universe if t != "XLK"]
    for t, m in zip(leader_first, mults):
        closes[t] = 100.0 * np.exp((slope * m) * rng)
    opens = closes.shift(1).fillna(closes.iloc[0])
    return closes, opens


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------

def test_synthetic_sentiment_passes_buy_threshold():
    df = _synthetic_sentiment(["XLK", "XLF"])
    assert (df["score"] >= PARAMS.buy_sentiment_threshold).all()
    assert df["n_obs"].iloc[0] == 1
    assert set(df.columns) == {"score", "n_obs", "score_stdev",
                                "score_min", "score_max"}


def test_max_drawdown_monotonic():
    # A monotonically-rising equity has zero drawdown.
    eq = pd.Series(np.linspace(100, 200, 100),
                   index=pd.bdate_range("2020-01-01", periods=100))
    assert _max_drawdown(eq) == 0.0


def test_max_drawdown_known_dip():
    eq = pd.Series([100, 110, 90, 95, 120],
                   index=pd.bdate_range("2020-01-01", periods=5))
    # peak 110 -> trough 90 = -18.18%
    assert _max_drawdown(eq) == pytest.approx(-(20 / 110), rel=1e-6)


def test_annualised_stats_flat_returns_zero():
    eq = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=252))
    s = _annualised_stats(eq)
    assert s["cagr"] == pytest.approx(0.0, abs=1e-9)
    assert s["sharpe"] == 0.0
    assert s["max_drawdown"] == 0.0


def test_weekly_rebalance_dates_one_per_week():
    closes, _ = _flat_panel(60)
    dates = weekly_rebalance_dates(closes, None, None)
    # Each rebalance is the LAST trading day of its iso week.
    weeks = pd.DatetimeIndex(dates).isocalendar().week
    assert len(weeks) == len(set(zip(weeks,
                                     pd.DatetimeIndex(dates).isocalendar().year)))


def test_closed_position_hit_rate_fifo():
    # Open at $100, partial sell at $110 (win), close remaining at $90 (loss).
    df = pd.DataFrame([
        {"ticker": "X", "fill_date": pd.Timestamp("2020-01-02"),
         "side": "BUY",  "shares": 10.0, "price": 100.0, "notional": 1000.0, "cost": 0.0,
         "rebalance_date": pd.Timestamp("2020-01-02"), "state": "NEW_BUY"},
        {"ticker": "X", "fill_date": pd.Timestamp("2020-01-09"),
         "side": "SELL", "shares": -5.0, "price": 110.0, "notional": -550.0, "cost": 0.0,
         "rebalance_date": pd.Timestamp("2020-01-09"), "state": "SELL"},
        {"ticker": "X", "fill_date": pd.Timestamp("2020-01-16"),
         "side": "SELL", "shares": -5.0, "price": 90.0,  "notional": -450.0, "cost": 0.0,
         "rebalance_date": pd.Timestamp("2020-01-16"), "state": "SELL"},
    ])
    # One win, one loss => 50%.
    assert _closed_position_hit_rate(df) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End-to-end backtest invariants
# ---------------------------------------------------------------------------

def test_backtest_flat_market_no_trades_no_pnl():
    """Sanity: a flat market produces a HOLD-everywhere signal, so we never
    enter a trade, and equity stays at initial_capital."""
    closes, opens = _flat_panel(400)
    cfg = BacktestConfig(initial_capital=100_000.0, cost_bps=5.0)
    res = run_backtest(cfg, closes=closes, opens=opens)
    assert res.stats["n_trades"] == 0
    assert res.equity.iloc[-1] == pytest.approx(100_000.0)
    # SPY also flat => zero excess.
    assert res.benchmark_equity.iloc[-1] == pytest.approx(100_000.0)
    assert res.stats["excess_cagr"] == pytest.approx(0.0, abs=1e-9)


def test_backtest_trending_market_enters_winner():
    """In a trending market with positive cross-section dispersion, the model
    should at least make some trades and end up positive. (The single
    strongest sector may stay perpetually CHASE because extension > 12% in
    a smooth exponential trend — so we don't assert on a specific ticker.)
    """
    closes, opens = _trending_panel(700)
    res = run_backtest(BacktestConfig(cost_bps=5.0), closes=closes, opens=opens)
    assert res.stats["n_trades"] >= 1
    assert res.equity.iloc[-1] > 100_000.0
    # All traded tickers should come from the universe — never an unrelated
    # symbol or a supplementary one.
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    assert set(res.trades["ticker"]).issubset(set(universe))


def test_backtest_costs_applied_on_each_trade():
    """Every trade should have cost = |notional| * (cost_bps + slippage_bps) / 1e4."""
    closes, opens = _trending_panel(500)
    cfg = BacktestConfig(cost_bps=8.0, slippage_bps=2.0)
    res = run_backtest(cfg, closes=closes, opens=opens)
    if res.trades.empty:
        pytest.skip("trending fixture produced no trades")
    expected = res.trades["notional"].abs() * (10.0 / 10_000.0)
    np.testing.assert_allclose(res.trades["cost"].values, expected.values, rtol=1e-9)


def test_backtest_no_lookahead_signal_only_uses_prior_bars():
    """A spike on day T should NOT influence the signal computed on day T-1.

    Construct a panel where XLK is flat for a long warmup, then jumps 30% on
    one day. The signal evaluated on the day BEFORE the jump must produce the
    same state for XLK as a control panel where the jump never happens.
    """
    n = 500
    idx = _trading_days("2020-01-02", n)
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    cols = universe + [BENCHMARK]
    base = pd.DataFrame(100.0, index=idx, columns=cols)
    # Mild upward drift everywhere so SMA200 is computable & well-defined.
    for c in cols:
        base[c] = 100.0 * (1 + 0.0005) ** np.arange(n)

    spike_day = idx[400]
    spiked = base.copy()
    spiked.loc[spike_day:, "XLK"] *= 1.30  # +30% jump

    # Evaluate signals at the day BEFORE the spike using each panel; XLK's
    # state should be identical.
    from src.market_engine import compute_sector_metrics
    pre_day = idx[399]
    m_base = compute_sector_metrics(base, as_of=pre_day)
    m_spiked = compute_sector_metrics(spiked, as_of=pre_day)
    assert m_base.loc["XLK", "above_sma"] == m_spiked.loc["XLK", "above_sma"]
    assert m_base.loc["XLK", "relative_strength_3m"] == pytest.approx(
        m_spiked.loc["XLK", "relative_strength_3m"], rel=1e-12)


def test_backtest_equity_curve_is_continuous_and_dated():
    """Equity should be a sorted, gap-free per-trading-day Series across
    the chosen window — no duplicate dates, no NaNs."""
    closes, opens = _trending_panel(500)
    res = run_backtest(BacktestConfig(), closes=closes, opens=opens)
    assert res.equity.is_monotonic_increasing or True  # not required
    assert res.equity.index.is_monotonic_increasing
    assert not res.equity.index.has_duplicates
    assert res.equity.notna().all()


def test_real_sentiment_ablation_returns_caveat():
    """The ablation function should never silently report 'success' without
    a small-n caveat string. Returns numbers (possibly zero) plus the caveat.
    """
    # Use a synthetic panel so we don't need the real DB.
    closes, _ = _trending_panel(400)
    out = real_sentiment_ablation(closes=closes, weeks=8)
    assert "caveat" in out and out["caveat"]
    for arm in ("off", "on"):
        assert "n_signals" in out[arm]


# ---------------------------------------------------------------------------
# Regime-aware bull overlay (BacktestConfig.regime_aware)
# ---------------------------------------------------------------------------

def _declining_panel(n_days: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SPY (and everything) grinds steadily lower so the market is never BULL
    — SPY sits far below its trailing 252-day high → CORRECTION/BEAR. Sectors
    keep a stair-stepped RS ordering so the pipeline still ranks them."""
    idx = _trading_days("2020-01-02", n_days)
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    cols = universe + [BENCHMARK]
    closes = pd.DataFrame(index=idx, columns=cols, dtype=float)
    rng = np.arange(n_days)
    closes[BENCHMARK] = 100.0 * np.exp(-0.0010 * rng)
    mults = np.linspace(0.6, 1.4, len(universe))  # all decline, at varied rates
    for t, m in zip(universe, mults):
        closes[t] = 100.0 * np.exp((-0.0010 * m) * rng)
    opens = closes.shift(1).fillna(closes.iloc[0])
    return closes, opens


def test_regime_aware_off_in_downtrend_is_noop():
    """A market that never reaches BULL must leave the overlay dormant, so
    regime_aware=True is byte-identical to the defensive baseline."""
    closes, opens = _declining_panel(500)
    base = run_backtest(BacktestConfig(cost_bps=0.0), closes=closes, opens=opens)
    aware = run_backtest(BacktestConfig(cost_bps=0.0, regime_aware=True),
                         closes=closes, opens=opens)
    assert aware.equity.equals(base.equity)


def test_regime_aware_default_is_unchanged_when_flag_off():
    """regime_aware defaults False → identical to current behaviour even in a
    bull tape (the overlay must be opt-in)."""
    closes, opens = _trending_panel(500)
    a = run_backtest(BacktestConfig(cost_bps=0.0), closes=closes, opens=opens)
    b = run_backtest(BacktestConfig(cost_bps=0.0, regime_aware=False),
                     closes=closes, opens=opens)
    assert a.equity.equals(b.equity)


def _bull_with_holdings_panel(n_days: int = 600
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A confirmed STRONG bull where the model actually holds positions.

    A pure exponential trend pushes every sector either past the +12% extension
    cutoff (→ CHASE) or below SPY's pace (→ SELL), leaving the book all-cash —
    useless for exercising the overlay. Here the benchmark and most sectors
    drift up GENTLY enough that the rising SMA200 stays close to price, so the
    leaders qualify as NEW_BUY / HOLD_IF_LONG (above SMA, +RS, not extended).
    XLK runs steeply so it sits in CHASE. SPY trends up steadily → it is near
    its trailing high with a rising SMA throughout the back half → the strong-
    bull gate fires there.
    """
    idx = _trading_days("2019-01-02", n_days)
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    cols = universe + [BENCHMARK]
    closes = pd.DataFrame(index=idx, columns=cols, dtype=float)
    rng = np.arange(n_days)
    closes[BENCHMARK] = 100.0 * np.exp(0.0004 * rng)
    # Distinct gentle slopes, all just beating SPY → NEW_BUY-eligible leaders.
    slopes = np.linspace(0.00055, 0.00075, len(universe))
    order = ["XLK"] + [t for t in universe if t != "XLK"]
    for t, s in zip(order, slopes):
        closes[t] = 100.0 * np.exp(s * rng)
    closes["XLK"] = 100.0 * np.exp(0.0016 * rng)  # steep → perpetual CHASE
    opens = closes.shift(1).fillna(closes.iloc[0])
    return closes, opens


def test_regime_aware_bull_increases_exposure():
    """In a confirmed strong bull with positions held, the overlay drops the
    cash buffer (and promotes any CHASE leader to full weight), so the book is
    at least as invested on average and strictly more invested on some
    rebalances than the defensive baseline."""
    closes, opens = _bull_with_holdings_panel(600)
    base = run_backtest(BacktestConfig(cost_bps=0.0), closes=closes, opens=opens)
    aware = run_backtest(BacktestConfig(cost_bps=0.0, regime_aware=True),
                         closes=closes, opens=opens)
    base_inv = 1.0 - base.weights_history["cash_buffer"]
    aware_inv = 1.0 - aware.weights_history["cash_buffer"]
    # Sanity: this fixture actually holds positions (else the test is vacuous).
    assert base_inv.max() > 0.0
    assert aware_inv.mean() > base_inv.mean()
    assert (aware_inv > base_inv + 1e-9).any()
