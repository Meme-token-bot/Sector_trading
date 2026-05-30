"""Tests for `src.walk_forward`. Focus on the pure logic (winner selection,
fold scheduling) — full sweep runs are too slow for the test loop and are
exercised by `scripts/run_walk_forward.py` directly."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.walk_forward import (
    FoldResult,
    ParamSweepResult,
    WalkForwardConfig,
    build_folds,
    pick_robust_winner,
)


def _fake_fold(fold: int, winner, w_oos: float, d_oos: float) -> FoldResult:
    return FoldResult(
        fold=fold,
        train_start=date(2024, 1, 1), train_end=date(2024, 6, 30),
        test_start=date(2024, 7, 1), test_end=date(2024, 12, 31),
        param="extension_pct_cutoff",
        candidate_train_scores={winner: 0.5},
        winner=winner,
        winner_test_score=w_oos,
        default_test_score=d_oos,
        delta_vs_default=(w_oos - d_oos),
    )


def _result(param: str, default: float, folds: list[FoldResult]) -> ParamSweepResult:
    deltas = [f.delta_vs_default for f in folds]
    return ParamSweepResult(
        param=param, candidates=[default],
        default_value=default, folds=folds,
        winners_by_fold=[f.winner for f in folds],
        mean_winner_oos=sum(f.winner_test_score for f in folds) / len(folds),
        mean_default_oos=sum(f.default_test_score for f in folds) / len(folds),
        mean_delta=sum(deltas) / len(deltas),
        n_folds=len(folds),
    )


# ---------------------------------------------------------------------------
# Robustness rule
# ---------------------------------------------------------------------------

def test_winner_with_positive_oos_lift_is_applied():
    """Modal winner across folds + meaningful (>+0.5pp) lift → apply."""
    folds = [_fake_fold(i, 0.25, 0.05, 0.0) for i in range(6)]  # +5pp lift
    r = _result("extension_pct_cutoff", 0.12, folds)
    w, why = pick_robust_winner(r)
    assert w == 0.25
    assert "+5.00pp" in why or "modal pick" in why


def test_winner_with_negative_oos_lift_keeps_default():
    """If the train optimizer's picks UNDERPERFORM default OOS, never apply."""
    folds = [_fake_fold(i, 0.40, -0.03, 0.0) for i in range(6)]  # -3pp lift
    r = _result("extension_pct_cutoff", 0.12, folds)
    w, why = pick_robust_winner(r)
    assert w == 0.12
    assert "keep default" in why
    assert "negative" in why or "-3.00pp" in why


def test_winner_with_marginal_positive_lift_keeps_default():
    """A <+0.5pp lift is below noise — don't churn live params for it."""
    folds = [_fake_fold(i, 0.18, 0.002, 0.0) for i in range(6)]  # +0.2pp
    r = _result("extension_pct_cutoff", 0.12, folds)
    w, why = pick_robust_winner(r)
    assert w == 0.12
    assert "keep default" in why


def test_winner_modal_equals_default_passes_through():
    """If the train optimizer keeps picking the live default, just say so."""
    folds = [_fake_fold(i, 0.12, 0.0, 0.0) for i in range(6)]
    r = _result("extension_pct_cutoff", 0.12, folds)
    w, why = pick_robust_winner(r)
    assert w == 0.12


def test_empty_folds_returns_default():
    # `_result` divides by len(folds) so build the ParamSweepResult by hand.
    r = ParamSweepResult(
        param="extension_pct_cutoff", candidates=[0.12],
        default_value=0.12, folds=[], winners_by_fold=[],
        mean_winner_oos=float("nan"), mean_default_oos=float("nan"),
        mean_delta=float("nan"), n_folds=0,
    )
    w, why = pick_robust_winner(r)
    assert w == 0.12


# ---------------------------------------------------------------------------
# Fold scheduling
# ---------------------------------------------------------------------------

def test_build_folds_respects_window_sizes():
    """train_years + test_months fold widths come out close to spec."""
    idx = pd.bdate_range("2020-01-02", periods=2000)
    closes = pd.DataFrame({"X": 1.0}, index=idx)
    wfc = WalkForwardConfig(train_years=2.0, test_months=6.0, step_months=6.0)
    folds = build_folds(closes, wfc)
    assert len(folds) >= 3
    for tr_s, tr_e, te_s, te_e in folds:
        # Train ~ 2 years (730 days ± a couple from int rounding)
        assert 720 <= (tr_e - tr_s).days <= 735
        # Test ~ 6 months
        assert 180 <= (te_e - te_s).days <= 185


def test_build_folds_empty_when_window_too_short():
    idx = pd.bdate_range("2020-01-02", periods=10)
    closes = pd.DataFrame({"X": 1.0}, index=idx)
    wfc = WalkForwardConfig(train_years=2.0)
    assert build_folds(closes, wfc) == []
