"""Tests for src.signals — conviction score + sentiment-quality propagation.

Focuses on the new additive columns. The existing BUY/HOLD/SELL convergence
rules are not retested here (they remain governed by the project's hand-
verified scenarios).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import PARAMS
from src.signals import build_signals, refine_signals


# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic frames that match what compute_sector_metrics
# would produce (only the columns refine_signals reads).
# ---------------------------------------------------------------------------

def _base_signal_row(
    *,
    signal: str = "BUY",
    reasons: str = "",
    rs3: float = 0.0,
    above_sma: bool = True,
    extension_pct: float = 0.02,
    sentiment_score: float = 0.0,
    n_obs: int = 1,
    score_stdev: float = 0.0,
    score_min: float = 0.0,
    score_max: float = 0.0,
) -> dict:
    return {
        "name": "Test",
        "price": 100.0,
        "sma200": 98.0,
        "above_sma": above_sma,
        "extension_pct": extension_pct,
        "return_3m": rs3 + 0.02,
        "spy_return_3m": 0.02,
        "relative_strength_3m": rs3,
        "rs_rank": 1,
        "sentiment_score": sentiment_score,
        "n_obs": n_obs,
        "score_stdev": score_stdev,
        "score_min": score_min,
        "score_max": score_max,
        "signal": signal,
        "reasons": reasons,
    }


def _frame(**kwargs) -> pd.DataFrame:
    """Build a one-row signals frame indexed by ticker 'XLK'."""
    row = _base_signal_row(**kwargs)
    return pd.DataFrame({"XLK": row}).T


def _macro(ratio: float) -> pd.DataFrame:
    """Single-sector macro_alignment frame."""
    return pd.DataFrame(
        {"tailwinds": [1], "headwinds": [0], "neutral": [0],
         "ratio": [ratio], "detail": [[]]},
        index=pd.Index(["XLK"], name="sector"),
    )


# ---------------------------------------------------------------------------
# Conviction scoring — each marginal point in isolation.
# ---------------------------------------------------------------------------

def test_conviction_zero_baseline():
    """Nothing positive — conviction == 0."""
    df = _frame(rs3=-0.01, sentiment_score=0.0)
    out = refine_signals(df)
    assert out.loc["XLK", "conviction"] == 0


def test_conviction_point_for_positive_rs():
    """rs3 > 0 but below strong_rs_margin → +1 only."""
    df = _frame(rs3=0.01, sentiment_score=0.0)
    assert PARAMS.strong_rs_margin > 0.01
    out = refine_signals(df)
    assert out.loc["XLK", "conviction"] == 1


def test_conviction_point_for_strong_rs():
    """rs3 above strong_rs_margin → +1 (positive) +1 (strong) = 2."""
    df = _frame(rs3=PARAMS.strong_rs_margin + 0.01, sentiment_score=0.0)
    out = refine_signals(df)
    assert out.loc["XLK", "conviction"] == 2


def test_conviction_point_for_high_sentiment():
    """sentiment >= buy_threshold+1 → +1."""
    df = _frame(rs3=-0.01,
                sentiment_score=PARAMS.buy_sentiment_threshold + 1)
    out = refine_signals(df)
    assert out.loc["XLK", "conviction"] == 1


def test_conviction_point_for_consecutive_buy_weeks():
    """consecutive_buy_weeks >= 2 → +1. Achieved via a 2-row BUY history."""
    df = _frame(rs3=-0.01, sentiment_score=0.0, signal="BUY")
    history = pd.DataFrame(
        [{"XLK": "BUY"}, {"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-08", "2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    # Only the 2-consec-BUY component fires; ext/sma keep state on a buy-class
    # value but that's irrelevant to conviction.
    assert out.loc["XLK", "consecutive_buy_weeks"] == 2
    assert out.loc["XLK", "conviction"] == 1


def test_conviction_point_for_macro_alignment():
    """macro ratio >= 0.5 → +1."""
    df = _frame(rs3=-0.01, sentiment_score=0.0)
    out = refine_signals(df, macro_alignment=_macro(ratio=0.6))
    assert out.loc["XLK", "conviction"] == 1


def test_conviction_no_point_when_macro_ratio_below_threshold():
    df = _frame(rs3=-0.01, sentiment_score=0.0)
    out = refine_signals(df, macro_alignment=_macro(ratio=0.4))
    assert out.loc["XLK", "conviction"] == 0


def test_conviction_max_without_macro_is_four():
    """All four non-macro components active → 4 (max without macro frame)."""
    df = _frame(
        rs3=PARAMS.strong_rs_margin + 0.02,
        sentiment_score=PARAMS.buy_sentiment_threshold + 1.5,
        signal="BUY",
    )
    history = pd.DataFrame(
        [{"XLK": "BUY"}, {"XLK": "BUY"}, {"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-01", "2026-05-08", "2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    assert out.loc["XLK", "conviction"] == 4


def test_conviction_max_with_macro_is_five():
    df = _frame(
        rs3=PARAMS.strong_rs_margin + 0.02,
        sentiment_score=PARAMS.buy_sentiment_threshold + 1.5,
        signal="BUY",
    )
    history = pd.DataFrame(
        [{"XLK": "BUY"}, {"XLK": "BUY"}, {"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-01", "2026-05-08", "2026-05-15"]),
    )
    out = refine_signals(df, history=history, macro_alignment=_macro(0.8))
    assert out.loc["XLK", "conviction"] == 5


def test_refine_callable_with_no_macro_data():
    """No macro frame supplied → macro component is 0; function still runs."""
    df = _frame(rs3=0.05, sentiment_score=3.5)
    out = refine_signals(df)  # history=None, macro_alignment=None
    # conviction should be computable without macro_alignment.
    assert "conviction" in out.columns
    assert 0 <= out.loc["XLK", "conviction"] <= 4


# ---------------------------------------------------------------------------
# Sentiment-quality propagation through build_signals.
# ---------------------------------------------------------------------------

def test_build_signals_includes_quality_columns_when_available():
    metrics = pd.DataFrame({
        "XLK": {
            "name": "Tech", "price": 100.0, "sma200": 98.0,
            "above_sma": True, "extension_pct": 0.02,
            "return_3m": 0.05, "spy_return_3m": 0.02,
            "relative_strength_3m": 0.03, "rs_rank": 1,
        }
    }).T
    sentiment = pd.DataFrame({
        "score": [2.5], "n_obs": [4],
        "score_stdev": [0.5], "score_min": [2.0], "score_max": [3.0],
    }, index=pd.Index(["XLK"], name="ticker"))

    out = build_signals(metrics, sentiment)
    assert out.loc["XLK", "score_stdev"] == pytest.approx(0.5)
    assert out.loc["XLK", "score_min"] == pytest.approx(2.0)
    assert out.loc["XLK", "score_max"] == pytest.approx(3.0)


def test_build_signals_defaults_quality_columns_when_sentiment_empty():
    metrics = pd.DataFrame({
        "XLK": {
            "name": "Tech", "price": 100.0, "sma200": 98.0,
            "above_sma": True, "extension_pct": 0.02,
            "return_3m": 0.05, "spy_return_3m": 0.02,
            "relative_strength_3m": 0.03, "rs_rank": 1,
        }
    }).T
    out = build_signals(metrics, pd.DataFrame())
    assert out.loc["XLK", "score_stdev"] == pytest.approx(0.0)
    assert np.isnan(out.loc["XLK", "score_min"])
    assert np.isnan(out.loc["XLK", "score_max"])
