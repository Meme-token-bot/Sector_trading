"""Tests for `src.walk_forward`. Focus on the pure logic (winner selection,
fold scheduling, and the new status-payload builder) — full sweep runs are
too slow for the test loop and are exercised by `scripts/run_walk_forward.py`
directly."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.walk_forward import (
    FoldResult,
    ParamSweepResult,
    WalkForwardConfig,
    build_folds,
    build_status_payload,
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
    folds = [_fake_fold(i, 0.25, 0.05, 0.0) for i in range(6)]
    r = _result("extension_pct_cutoff", 0.12, folds)
    w, why = pick_robust_winner(r)
    assert w == 0.25
    assert "+5.00pp" in why or "modal pick" in why


def test_winner_with_negative_oos_lift_keeps_default():
    folds = [_fake_fold(i, 0.40, -0.03, 0.0) for i in range(6)]
    r = _result("extension_pct_cutoff", 0.12, folds)
    w, why = pick_robust_winner(r)
    assert w == 0.12
    assert "keep default" in why
    assert "negative" in why or "-3.00pp" in why


def test_winner_with_marginal_positive_lift_keeps_default():
    folds = [_fake_fold(i, 0.18, 0.002, 0.0) for i in range(6)]
    r = _result("extension_pct_cutoff", 0.12, folds)
    w, why = pick_robust_winner(r)
    assert w == 0.12
    assert "keep default" in why


def test_empty_folds_returns_default():
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
    idx = pd.bdate_range("2020-01-02", periods=2000)
    closes = pd.DataFrame({"X": 1.0}, index=idx)
    wfc = WalkForwardConfig(train_years=2.0, test_months=6.0, step_months=6.0)
    folds = build_folds(closes, wfc)
    assert len(folds) >= 3
    for tr_s, tr_e, te_s, te_e in folds:
        assert 720 <= (tr_e - tr_s).days <= 735
        assert 180 <= (te_e - te_s).days <= 185


def test_build_folds_empty_when_window_too_short():
    idx = pd.bdate_range("2020-01-02", periods=10)
    closes = pd.DataFrame({"X": 1.0}, index=idx)
    wfc = WalkForwardConfig(train_years=2.0)
    assert build_folds(closes, wfc) == []


# ---------------------------------------------------------------------------
# build_status_payload — new, for the Dashboard's walk-forward trust badge
# ---------------------------------------------------------------------------

def test_status_payload_marks_kept_defaults_as_unchanged():
    """Mirrors the real WALK_FORWARD_REPORT.md outcome: every swept param
    kept its default (mean OOS lift negative or ~0), so every verdict's
    `changed` flag should be False."""
    folds = [_fake_fold(i, 0.40, -0.03, 0.0) for i in range(6)]
    result = _result("extension_pct_cutoff", 0.12, folds)
    winner, why = pick_robust_winner(result)  # -> 0.12 (default kept)
    payload = build_status_payload(
        results={"extension_pct_cutoff": result},
        winners={"extension_pct_cutoff": winner},
        reasons={"extension_pct_cutoff": why},
        wfc=WalkForwardConfig(),
        generated_at=date(2026, 5, 30),
    )
    v = payload["verdicts"]["extension_pct_cutoff"]
    assert v["changed"] is False
    assert v["winner"] == 0.12
    assert v["default"] == 0.12
    assert payload["generated_at"] == "2026-05-30"


def test_status_payload_marks_a_real_change_as_changed():
    folds = [_fake_fold(i, 0.25, 0.05, 0.0) for i in range(6)]
    result = _result("extension_pct_cutoff", 0.12, folds)
    winner, why = pick_robust_winner(result)  # -> 0.25 (a real change)
    payload = build_status_payload(
        results={"extension_pct_cutoff": result},
        winners={"extension_pct_cutoff": winner},
        reasons={"extension_pct_cutoff": why},
        wfc=WalkForwardConfig(),
    )
    v = payload["verdicts"]["extension_pct_cutoff"]
    assert v["changed"] is True
    assert v["winner"] == 0.25


def test_status_payload_is_json_serializable():
    """The whole point is that app.py can json.loads() this straight back —
    guard against accidentally leaking a non-serializable type (numpy
    scalar, NaN, etc.) into the payload."""
    import json
    folds = [_fake_fold(i, 0.25, 0.05, 0.0) for i in range(6)]
    result = _result("extension_pct_cutoff", 0.12, folds)
    payload = build_status_payload(
        results={"extension_pct_cutoff": result},
        winners={"extension_pct_cutoff": 0.25},
        reasons={"extension_pct_cutoff": "test"},
        wfc=WalkForwardConfig(),
        joint={"mean_default_excess": -0.05, "mean_new_excess": -0.05,
              "mean_delta_excess": 0.0},
    )
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped == payload


def test_status_payload_handles_nan_mean_delta_as_null():
    """A param with zero successful folds has mean_delta == NaN; NaN is not
    valid JSON, so it must be coerced to null rather than crashing json.dumps
    or round-tripping as the string 'NaN'."""
    import json
    empty_result = ParamSweepResult(
        param="stale_buy_weeks", candidates=[4], default_value=4,
        folds=[], winners_by_fold=[],
        mean_winner_oos=float("nan"), mean_default_oos=float("nan"),
        mean_delta=float("nan"), n_folds=0,
    )
    payload = build_status_payload(
        results={"stale_buy_weeks": empty_result},
        winners={"stale_buy_weeks": 4},
        reasons={"stale_buy_weeks": "no folds"},
        wfc=WalkForwardConfig(),
    )
    assert payload["verdicts"]["stale_buy_weeks"]["mean_delta"] is None
    # Must actually be JSON-serializable, not just look right in Python.
    json.dumps(payload)


def test_status_payload_omits_joint_key_when_not_provided():
    result = _result("extension_pct_cutoff", 0.12,
                     [_fake_fold(0, 0.12, 0.0, 0.0)])
    payload = build_status_payload(
        results={"extension_pct_cutoff": result},
        winners={"extension_pct_cutoff": 0.12},
        reasons={"extension_pct_cutoff": "x"},
        wfc=WalkForwardConfig(),
    )
    assert "joint" not in payload
