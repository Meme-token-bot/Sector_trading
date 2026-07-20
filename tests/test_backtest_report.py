"""Regression tests for the two B5 narrative-drift fixes in
src.backtest_report (see TRADING_EDGE_AUDIT.md):

1. The verdict paragraph must quote the ANNUALIZED per-regime excess
   (`excess_cagr`), not the cumulative multi-year compounding gap
   (`excess_cum`) — the latter produced numbers like "-127.23%" that read
   as an impossible "lost 127%" claim for a long-only, unlevered strategy.

2. The "never tested in a sustained bear" caveat must be computed from the
   actual `drawdown_attribution` output, not hand-typed — it must not
   claim "only corrections (5-19%)" when a qualifying >30% drawdown (e.g.
   a fast COVID-style crash) is present in the sample, and it must
   correctly distinguish a fast crash from a long grinding bear by
   duration, not just depth.

Builds a minimal synthetic HeadlineReport rather than running a real
backtest — this module has no network/DB dependency once the report
object exists, matching the pure-render contract of `render_markdown`.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, BacktestResult
from src.backtest_report import DBFindings, HeadlineReport, render_markdown


def _fake_stats(cagr: float, ann_vol: float, sharpe: float, mdd: float,
                total_return: float, n_days: int, cost_bps: float = 5.0,
                slippage_bps: float = 0.0, cash_buffer: float = 0.05,
                sentiment_gate: str = "off") -> dict:
    return {
        "cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_drawdown": mdd, "total_return": total_return, "n_days": n_days,
    }


def _fake_result(strat_cagr, spy_cagr, strat_mdd, spy_mdd) -> BacktestResult:
    idx = pd.bdate_range("2019-01-18", periods=1300)
    t = np.arange(len(idx))
    equity = pd.Series(100_000 * (1 + strat_cagr / 252) ** t, index=idx)
    bench = pd.Series(100_000 * (1 + spy_cagr / 252) ** t, index=idx)
    stats = {
        "window_start": str(idx[0].date()), "window_end": str(idx[-1].date()),
        "initial_capital": 100_000.0, "final_equity": float(equity.iloc[-1]),
        "strategy": _fake_stats(strat_cagr, 0.17, 0.7, strat_mdd, 1.2, len(idx)),
        "spy": _fake_stats(spy_cagr, 0.19, 0.9, spy_mdd, 2.0, len(idx)),
        "excess_cagr": strat_cagr - spy_cagr,
        "excess_total_return": -0.5,
        "annualised_turnover": 16.5, "n_trades": 553, "total_costs": 9218.0,
        "closed_position_hit_rate": 0.486,
        "config": {"execution": "next_open", "cost_bps": 5.0,
                   "slippage_bps": 0.0, "cash_buffer": 0.05,
                   "sentiment_gate": "off"},
    }
    return BacktestResult(
        config=BacktestConfig(), equity=equity, benchmark_equity=bench,
        trades=pd.DataFrame({"cost": [1.0, 2.0]}),
        stats=stats, weights_history=pd.DataFrame(),
        states_history=pd.DataFrame(columns=["date", "ticker", "state", "extension_pct"]),
    )


def _fake_findings() -> DBFindings:
    return DBFindings(
        prices_window=("2018-01-02", "2026-05-28"),
        prices_coverage=[],
        sectors_with_daily=["SPY", "XLK"],
        n_newsletters=100, n_sector_ratings=200,
        sentiment_window=("2026-02-08", "2026-05-30"),
        newsletters_by_date=[("2026-05-29", 3), ("2026-05-30", 2)],
        sector_coverage=[{"ticker": "XLK", "n": 36,
                         "first_date": "2026-05-05", "last_date": "2026-05-29"}],
    )


def _fake_report(strat_cagr=0.1168, spy_cagr=0.1692,
                 strat_mdd=-0.2611, spy_mdd=-0.3372,
                 bull_excess_cum=-1.2723, bull_excess_cagr=-0.024,
                 dd_rows=None) -> HeadlineReport:
    ed = _fake_result(strat_cagr, spy_cagr, strat_mdd, spy_mdd)
    rbt = _fake_result(0.0417, spy_cagr, -0.30, spy_mdd)
    regime_stats = pd.DataFrame({
        "n_days": [1316, 358, 174],
        "years": [5.2, 1.4, 0.7],
        "strategy_cum": [2.7651, -0.1409, -0.3033],
        "spy_cum": [4.0374, -0.0136, -0.3645],
        "excess_cum": [bull_excess_cum, -0.1273, 0.0612],
        "strategy_cagr": [0.30, -0.10, -0.35],
        "spy_cagr": [0.30 - bull_excess_cagr, -0.09, -0.40],
        "excess_cagr": [bull_excess_cagr, -0.01, 0.05],
        "strategy_ann_vol": [0.15, 0.20, 0.30],
        "spy_ann_vol": [0.17, 0.22, 0.35],
        "strategy_mdd_in_regime": [-0.094, -0.206, -0.293],
        "spy_mdd_in_regime": [-0.049, -0.218, -0.311],
        "capture_up": [0.83, 0.68, 0.59],
        "capture_down": [0.83, 0.75, 0.65],
    }, index=pd.Index(["BULL", "CORRECTION", "BEAR"], name="regime"))

    if dd_rows is None:
        dd_rows = [
            {"peak_date": date(2020, 2, 19), "trough_date": date(2020, 3, 23),
             "days_to_trough": 33, "days_to_recover": 90,
             "spy_drawdown": -0.3372, "strategy_drawdown": -0.2592,
             "excess_drawdown": 0.0780,
             "held_at_peak": [], "held_at_trough": [],
             "rotated_in_during_dd": [], "rotated_out_during_dd": []},
            {"peak_date": date(2022, 1, 3), "trough_date": date(2022, 10, 12),
             "days_to_trough": 282, "days_to_recover": None,
             "spy_drawdown": -0.2450, "strategy_drawdown": -0.2043,
             "excess_drawdown": 0.0407,
             "held_at_peak": [], "held_at_trough": [],
             "rotated_in_during_dd": [], "rotated_out_during_dd": []},
        ]

    return HeadlineReport(
        findings=_fake_findings(), ed_result=ed, rbt_result=rbt,
        ablation={"window_weeks": 14, "n_rebalances": 14,
                 "off": {"n_signals": 26, "mean_excess_1w": 0.0034, "hit_rate": 0.46},
                 "on": {"n_signals": 6, "mean_excess_1w": 0.0095, "hit_rate": 0.67},
                 "caveat": "Sample sizes are TINY."},
        chase_share_by_ticker=pd.DataFrame([
            {"ticker": "XLK", "n_chase": 136, "share_pct": 35.4,
             "max_ext_pct": 28.9, "median_ext_when_chase_pct": 16.1},
        ]),
        state_distribution=pd.DataFrame(),
        n_weekly_snapshots=384,
        cost_drag_decomp={"years": 5.0, "ed_cost_drag_pct_yr": 1.25,
                          "rbt_cost_drag_pct_yr": 1.64,
                          "cost_drag_diff_pct_yr": 0.39,
                          "cagr_diff_pct": 7.50,
                          "structural_residual_pct_yr": 7.11},
        recent_newsletter_rate={"mean_per_day": 15.1, "window_days": 14,
                                "total": 212, "first": "2026-05-17",
                                "last": "2026-05-30"},
        regime_stats=regime_stats,
        regime_distribution={"BULL": 1316, "CORRECTION": 358, "BEAR": 174},
        drawdown_attribution=dd_rows,
    )


# ---------------------------------------------------------------------------
# Fix 1 — bull-regime excess must be annualized, not cumulative
# ---------------------------------------------------------------------------

def test_verdict_uses_annualized_excess_not_cumulative():
    report = _fake_report(bull_excess_cum=-1.2723, bull_excess_cagr=-0.024)
    md = render_markdown(report, as_of=date(2026, 7, 8))
    # The old bug: the cumulative figure ("-127.23%") appearing in the
    # verdict paragraph's "gave up X of upside" sentence.
    assert "-127.23%" not in md.split("## Blunt assessment")[1].split("**Caveats")[0]
    # The fix: the annualized figure, with a "/yr" unit so it can't be
    # misread as a lifetime loss.
    assert "-2.40%/yr" in md
    assert "gave up" in md


def test_regime_table_caption_flags_excess_cum_as_cumulative():
    """The per-regime TABLE still legitimately shows cumulative excess (it's
    labelled `Excess`, matching the table's own `excess_cum` column) — but
    the reader needs to be told, right there, that it's cumulative and to
    look at the verdict paragraph for the comparable annualized figure."""
    report = _fake_report()
    md = render_markdown(report, as_of=date(2026, 7, 8))
    assert "CUMULATIVE" in md
    assert "ANNUALIZED" in md


# ---------------------------------------------------------------------------
# Fix 2 — drawdown caveat is computed from drawdown_attribution, not hand-typed
# ---------------------------------------------------------------------------

def test_caveat_does_not_contradict_a_fast_deep_crash_in_sample():
    """Regression for the exact bug found in BACKTEST_REPORT.md: a -33.72%
    SPY drawdown (COVID, 33 days peak-to-trough) was present in
    drawdown_attribution while the caveat claimed 'only corrections
    (5-19% SPY moves), not crashes'. That specific contradiction must not
    reappear."""
    report = _fake_report()  # includes the -33.72%/33-day COVID-shaped row
    md = render_markdown(report, as_of=date(2026, 7, 8))
    caveats_section = md.split("**Caveats this measurement cannot escape:**")[1]
    first_caveat = caveats_section.split("2.")[0]
    assert "5–19%" not in first_caveat
    assert "5-19%" not in first_caveat
    # It should instead correctly acknowledge the >30% crash that's present...
    assert "-33.72%" in first_caveat or "33.72" in first_caveat
    # ...while still being honest that no GRINDING multi-year bear is in
    # the sample (33 and 282 days are both under the 252-day-plus
    # "grinding bear" bar this fix uses).
    assert "grinding" in first_caveat.lower()


def test_caveat_switches_tone_when_a_genuinely_long_drawdown_is_present():
    """If a qualifying drawdown's days_to_trough exceeds the long-bear
    threshold, the caveat must stop claiming no sustained bear exists."""
    long_bear_row = {
        "peak_date": date(2000, 3, 1), "trough_date": date(2002, 10, 1),
        "days_to_trough": 950, "days_to_recover": None,
        "spy_drawdown": -0.45, "strategy_drawdown": -0.30,
        "excess_drawdown": 0.15,
        "held_at_peak": [], "held_at_trough": [],
        "rotated_in_during_dd": [], "rotated_out_during_dd": [],
    }
    report = _fake_report(dd_rows=[long_bear_row])
    md = render_markdown(report, as_of=date(2026, 7, 8))
    caveats_section = md.split("**Caveats this measurement cannot escape:**")[1]
    first_caveat = caveats_section.split("2.")[0]
    assert "sustained, deep drawdown" in first_caveat
    assert "950 days" in first_caveat


def test_caveat_handles_no_qualifying_drawdowns_gracefully():
    report = _fake_report(dd_rows=[])
    md = render_markdown(report, as_of=date(2026, 7, 8))
    # Must not crash, and must not fabricate a depth/duration it doesn't have.
    assert "has not seen a crash-scale" in md


def test_small_sample_caveat_does_not_repeat_the_same_count_as_a_fraction():
    """Regression for a self-inflicted bug caught during implementation:
    an earlier draft rendered '{dd_total} of {dd_total} drawdown episodes'
    (e.g. '2 of 2'), which is meaningless — always 100% by construction —
    where the intent was just to flag the total count as small."""
    report = _fake_report()  # 2 drawdown rows in the default fixture
    md = render_markdown(report, as_of=date(2026, 7, 8))
    assert "2 of 2 drawdown" not in md
    assert "drawdown episodes is still a small sample" in md
