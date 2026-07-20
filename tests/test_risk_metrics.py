"""Tests for src.risk_metrics. Deterministic weekly-return construction
throughout (prices built via cumprod of a CHOSEN return array, sampled
exactly on W-FRI dates) so correlation/VaR/ES have known, hand-checkable
true values rather than "looks plausible."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    annualized_vol_by_ticker,
    average_pairwise_correlation,
    compute_correlation_matrix,
    concentration_metrics,
    historical_var_es,
)


def _weekly_price_frame(returns_by_ticker: dict[str, list[float]],
                        start: str = "2024-01-05", base: float = 100.0) -> pd.DataFrame:
    """Build a price DataFrame indexed on exact W-FRI dates, where each
    ticker's weekly returns are EXACTLY `returns_by_ticker[ticker]` (via
    cumprod) — no resampling ambiguity, since the index already IS weekly."""
    n = len(next(iter(returns_by_ticker.values())))
    idx = pd.date_range(start=start, periods=n + 1, freq="W-FRI")
    cols = {}
    for tk, rets in returns_by_ticker.items():
        prices = [base]
        for r in rets:
            prices.append(prices[-1] * (1 + r))
        cols[tk] = prices
    return pd.DataFrame(cols, index=idx)


# ---------------------------------------------------------------------------
# compute_correlation_matrix / average_pairwise_correlation
# ---------------------------------------------------------------------------

def test_perfectly_correlated_series_give_correlation_one():
    base_rets = [0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.015, -0.005, 0.01]
    prices = _weekly_price_frame({
        "A": base_rets,
        "B": [2.0 * r for r in base_rets],   # positive scalar multiple -> corr == 1
    })
    corr = compute_correlation_matrix(prices, ["A", "B"], lookback_weeks=52)
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-9)


def test_perfectly_anticorrelated_series_give_correlation_minus_one():
    base_rets = [0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.015, -0.005, 0.01]
    prices = _weekly_price_frame({
        "A": base_rets,
        "C": [-1.5 * r for r in base_rets],  # negative scalar multiple -> corr == -1
    })
    corr = compute_correlation_matrix(prices, ["A", "C"], lookback_weeks=52)
    assert corr.loc["A", "C"] == pytest.approx(-1.0, abs=1e-9)


def test_correlation_matrix_empty_with_one_ticker():
    prices = _weekly_price_frame({"A": [0.01] * 10})
    assert compute_correlation_matrix(prices, ["A"]).empty


def test_correlation_matrix_empty_with_too_little_history():
    prices = _weekly_price_frame({"A": [0.01, 0.02], "B": [0.01, -0.02]})
    assert compute_correlation_matrix(prices, ["A", "B"]).empty


def test_average_pairwise_correlation_uniform_offdiag():
    corr = pd.DataFrame(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    assert average_pairwise_correlation(corr) == pytest.approx(0.5)


def test_average_pairwise_correlation_identity_is_zero():
    corr = pd.DataFrame(np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"])
    assert average_pairwise_correlation(corr) == pytest.approx(0.0)


def test_average_pairwise_correlation_single_name_is_nan():
    corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    assert np.isnan(average_pairwise_correlation(corr))


# ---------------------------------------------------------------------------
# concentration_metrics — the "four sectors, one bet" property
# ---------------------------------------------------------------------------

def test_equal_weight_four_names_naive_effective_n_is_four():
    w = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    out = concentration_metrics(w)
    assert out["hhi"] == pytest.approx(0.25)
    assert out["effective_n_naive"] == pytest.approx(4.0)
    assert out["effective_n_corr_adjusted"] is None  # no corr/vols supplied


def test_single_name_full_weight_effective_n_is_one():
    w = pd.Series({"A": 1.0})
    out = concentration_metrics(w)
    assert out["effective_n_naive"] == pytest.approx(1.0)


def test_weights_renormalize_when_they_dont_sum_to_one():
    """A target-weights Series with a 20% cash buffer (sums to 0.8, not 1.0)
    must still read as fully-invested-and-equal-weight among the four
    NAMES, not be silently penalized for the cash sitting outside it."""
    w = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2})  # sums to 0.8
    out = concentration_metrics(w)
    assert out["effective_n_naive"] == pytest.approx(4.0)


def test_perfect_correlation_collapses_effective_n_to_one_regardless_of_count():
    """THE key property this module exists for: four equally-weighted
    sectors that are perfectly correlated are one bet, not four — the
    correlation-adjusted effective-N must say so even though the naive
    (weight-only) effective-N says 4."""
    tickers = ["A", "B", "C", "D"]
    w = pd.Series({t: 0.25 for t in tickers})
    corr = pd.DataFrame(1.0, index=tickers, columns=tickers)  # perfectly correlated
    vols = pd.Series({t: 0.15 for t in tickers})  # equal vols
    out = concentration_metrics(w, corr=corr, vols=vols)
    assert out["effective_n_naive"] == pytest.approx(4.0)
    assert out["diversification_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert out["effective_n_corr_adjusted"] == pytest.approx(1.0, abs=1e-9)


def test_zero_correlation_converges_corr_adjusted_to_naive():
    """With zero correlation and equal vols/weights, the correlation-adjusted
    effective-N should equal the naive one exactly — this is the sanity
    check that the heuristic doesn't systematically over- or under-state
    diversification in the no-correlation limit."""
    tickers = ["A", "B", "C", "D"]
    w = pd.Series({t: 0.25 for t in tickers})
    corr = pd.DataFrame(np.eye(4), index=tickers, columns=tickers)
    vols = pd.Series({t: 0.15 for t in tickers})
    out = concentration_metrics(w, corr=corr, vols=vols)
    assert out["effective_n_corr_adjusted"] == pytest.approx(out["effective_n_naive"], rel=1e-6)


def test_concentration_metrics_empty_weights():
    out = concentration_metrics(pd.Series(dtype=float))
    assert np.isnan(out["hhi"])
    assert out["effective_n_corr_adjusted"] is None


# ---------------------------------------------------------------------------
# historical_var_es
# ---------------------------------------------------------------------------

def test_single_asset_portfolio_matches_that_assets_own_percentile():
    rets = [0.03, -0.05, 0.02, -0.01, 0.04, -0.06, 0.01, -0.02, 0.05, -0.03,
           0.02, -0.04, 0.01, -0.01, 0.03, -0.02, 0.02, -0.03, 0.04, -0.01]
    prices = _weekly_price_frame({"A": rets})
    w = pd.Series({"A": 1.0})
    out = historical_var_es(prices, w, lookback_weeks=52, confidence=0.90)

    # Hand-check against the SAME returns, computed independently.
    arr = np.array(rets)
    expected_var = float(np.percentile(arr, 10))
    expected_es = float(arr[arr <= expected_var].mean())
    assert out["var"] == pytest.approx(expected_var, abs=1e-9)
    assert out["es"] == pytest.approx(expected_es, abs=1e-9)
    assert out["n_weeks"] == len(rets)


def test_zero_weight_asset_is_excluded_from_the_blend():
    rets_a = [0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.015, -0.005, 0.01]
    rets_b = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]  # very different
    prices = _weekly_price_frame({"A": rets_a, "B": rets_b})
    w = pd.Series({"A": 1.0, "B": 0.0})
    out = historical_var_es(prices, w, lookback_weeks=52, confidence=0.8)

    only_a = historical_var_es(prices, pd.Series({"A": 1.0}), lookback_weeks=52, confidence=0.8)
    assert out["var"] == pytest.approx(only_a["var"], abs=1e-9)


def test_es_is_never_better_than_var():
    """Expected Shortfall is the average of the tail beyond VaR — it must
    never be a smaller loss (i.e. algebraically larger) than VaR itself."""
    rng = np.random.default_rng(42)
    rets = list(rng.normal(0.001, 0.03, size=60))
    prices = _weekly_price_frame({"A": rets})
    out = historical_var_es(prices, pd.Series({"A": 1.0}), lookback_weeks=60, confidence=0.95)
    assert out["es"] <= out["var"] + 1e-12


def test_insufficient_history_returns_nan():
    prices = _weekly_price_frame({"A": [0.01, 0.02, -0.01]})  # only 3 weeks
    out = historical_var_es(prices, pd.Series({"A": 1.0}))
    assert np.isnan(out["var"])
    assert out["n_weeks"] == 0


def test_empty_weights_returns_nan():
    prices = _weekly_price_frame({"A": [0.01] * 20})
    out = historical_var_es(prices, pd.Series(dtype=float))
    assert np.isnan(out["var"])


# ---------------------------------------------------------------------------
# annualized_vol_by_ticker
# ---------------------------------------------------------------------------

def test_annualized_vol_matches_hand_computed_stdev():
    rets = [0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02]  # stdev known exactly
    prices = _weekly_price_frame({"A": rets})
    vols = annualized_vol_by_ticker(prices, ["A"], lookback_weeks=52)
    expected = float(np.std(rets, ddof=0) * np.sqrt(52))
    assert vols["A"] == pytest.approx(expected, rel=1e-6)
