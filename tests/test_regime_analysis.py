"""Pure-function tests for `src.regime_analysis`. Uses synthetic SPY series
so the regime/drawdown logic is checked against known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime_analysis import (
    classify_regimes,
    drawdown_attribution,
    identify_drawdowns,
    regime_conditional_stats,
    regime_episodes,
)


def _bdays(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def test_classify_regimes_flat_market_is_bull():
    s = pd.Series(100.0, index=_bdays("2020-01-01", 300))
    r = classify_regimes(s)
    assert (r == "BULL").all()


def test_classify_regimes_thresholds_at_5_and_15():
    # Build a panel that touches each band cleanly.
    idx = _bdays("2020-01-01", 100)
    # Start at 100, fall to 90 (-10% = CORRECTION), then to 80 (-20% = BEAR),
    # then back up.
    s = pd.Series([100.0] * 10 + [90.0] * 10 + [80.0] * 10 + [100.0] * 70,
                  index=idx)
    r = classify_regimes(s)
    # First 10 = BULL.
    assert r.iloc[5] == "BULL"
    # At index 10 (after the drop to 90 = -10%), should be CORRECTION.
    assert r.iloc[10] == "CORRECTION"
    # At index 20 (after the drop to 80 = -20%), should be BEAR.
    assert r.iloc[20] == "BEAR"
    # After recovery to 100, should be BULL again.
    assert r.iloc[-1] == "BULL"


def test_regime_episodes_collapses_runs():
    s = pd.Series(["BULL", "BULL", "CORRECTION", "BULL", "BEAR", "BEAR"],
                   index=_bdays("2020-01-01", 6))
    eps = regime_episodes(s)
    assert list(eps["regime"]) == ["BULL", "CORRECTION", "BULL", "BEAR"]
    assert list(eps["n_days"]) == [2, 1, 1, 2]


# ---------------------------------------------------------------------------
# Drawdown identification
# ---------------------------------------------------------------------------

def test_identify_drawdowns_finds_known_peak_trough():
    idx = _bdays("2020-01-01", 50)
    # Rise 100 → 110 over 10 days, fall to 80 over 20 days (-27% from peak),
    # recover to 115 over 20 days.
    vals = (list(np.linspace(100, 110, 10))
            + list(np.linspace(110, 80, 20))
            + list(np.linspace(80, 115, 20)))
    s = pd.Series(vals, index=idx)
    eps = identify_drawdowns(s, min_dd_pct=0.05, min_days=5)
    assert len(eps) == 1
    ep = eps[0]
    assert ep.peak_value == pytest.approx(110)
    assert ep.trough_value == pytest.approx(80)
    assert ep.drawdown_pct == pytest.approx(-30 / 110, rel=1e-3)
    assert ep.recovery_date is not None


def test_identify_drawdowns_filters_shallow_episodes():
    """A 2% dip is dropped (below threshold); a 22.7% drop is kept."""
    idx = _bdays("2020-01-01", 25)
    vals = [100, 99, 98, 99, 100,                  # 0-4: small dip filtered
            105, 110,                                # 5-6: new high
            110, 100, 95, 90, 85,                    # 7-11: -22.7% DD from 110
            85, 85, 85, 85, 85,                      # 12-16: hang at trough
            90, 95, 100, 105, 110, 115, 120, 120]    # 17-24: full recovery
    s = pd.Series(vals, index=idx, dtype=float)
    eps = identify_drawdowns(s, min_dd_pct=0.05, min_days=2)
    assert len(eps) == 1
    # peak=110, trough=85 → -25/110 ≈ -22.7%
    assert eps[0].drawdown_pct == pytest.approx(-25 / 110, rel=1e-3)
    assert eps[0].recovery_date is not None


def test_identify_drawdowns_handles_unrecovered_tail():
    idx = _bdays("2020-01-01", 30)
    vals = list(np.linspace(100, 110, 10)) + list(np.linspace(110, 90, 20))
    s = pd.Series(vals, index=idx)
    eps = identify_drawdowns(s, min_dd_pct=0.05, min_days=5)
    assert len(eps) == 1
    assert eps[0].recovery_date is None
    assert eps[0].days_to_recover is None


# ---------------------------------------------------------------------------
# Regime conditional stats
# ---------------------------------------------------------------------------

def test_regime_conditional_stats_compounds_within_regime():
    """Strategy +1%/day during BULL, 0% during BEAR; SPY 0% in BULL, +1% in BEAR.
    The per-regime compound returns should match exactly."""
    idx = _bdays("2020-01-01", 21)
    # 10 BULL days, then 10 BEAR days, then 1 final tick.
    regimes = pd.Series(["BULL"] * 10 + ["BEAR"] * 11, index=idx)
    strat_rets = [0.0] + [0.01] * 9 + [0.0] * 11    # +1%/day for 9 BULL days
    spy_rets = [0.0] * 10 + [0.01] * 11             # +1%/day for 10 BEAR days
    strat_eq = (1 + pd.Series(strat_rets, index=idx)).cumprod() * 100
    spy_eq = (1 + pd.Series(spy_rets, index=idx)).cumprod() * 100
    out = regime_conditional_stats(strat_eq, spy_eq, regimes)
    # Strategy compounded +1% for 9 days during BULL.
    assert out.loc["BULL", "strategy_cum"] == pytest.approx(1.01 ** 9 - 1)
    # SPY was flat during BULL.
    assert out.loc["BULL", "spy_cum"] == pytest.approx(0.0, abs=1e-9)
    # During BEAR (11 days), strategy flat, SPY +1%/day for 11 days.
    assert out.loc["BEAR", "spy_cum"] == pytest.approx(1.01 ** 11 - 1)


def test_regime_conditional_stats_orders_bull_correction_bear():
    idx = _bdays("2020-01-01", 30)
    regimes = pd.Series(["BEAR"] * 10 + ["BULL"] * 10 + ["CORRECTION"] * 10,
                         index=idx)
    eq = pd.Series(100.0, index=idx)
    out = regime_conditional_stats(eq, eq, regimes)
    assert list(out.index) == ["BULL", "CORRECTION", "BEAR"]


# ---------------------------------------------------------------------------
# Drawdown attribution structural check (uses a tiny fake BacktestResult)
# ---------------------------------------------------------------------------

def test_drawdown_attribution_returns_empty_when_no_qualifying_dd():
    from src.backtest import BacktestConfig, BacktestResult
    idx = _bdays("2020-01-01", 30)
    spy = pd.Series(100.0, index=idx)
    eq = pd.Series(100.0, index=idx)
    # Build a minimal valid BacktestResult.
    r = BacktestResult(
        config=BacktestConfig(), equity=eq, benchmark_equity=spy,
        trades=pd.DataFrame(), stats={}, weights_history=pd.DataFrame(),
        states_history=pd.DataFrame(),
    )
    out = drawdown_attribution(r, spy, spy, min_dd_pct=0.05)
    assert out == []
