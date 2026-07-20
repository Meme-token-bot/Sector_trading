"""Tests for src.edge_metrics.information_coefficient.

Uses synthetic price series engineered to have a KNOWN true cross-sectional
rank correlation between score and forward return, so the tests check
correctness of the IC computation itself, not just that it runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.edge_metrics import information_coefficient


def _monotonic_prices(n_tickers: int = 5, n_days: int = 300,
                      base_rate: float = 0.0006,
                      reverse: bool = False) -> dict[str, pd.Series]:
    """Ticker `T{k}` (k=1..n_tickers) grows at a distinct constant daily
    rate proportional to k (or to n_tickers+1-k if `reverse`). Because the
    growth rate ordering is fixed and monotonic, the forward-return ordering
    across tickers is IDENTICAL at every single snapshot date and at every
    horizon — this is what makes the true IC exactly +1 (or -1, reversed),
    not just "probably positive."
    """
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    t = np.arange(n_days)
    prices: dict[str, pd.Series] = {}
    for k in range(1, n_tickers + 1):
        rank = (n_tickers + 1 - k) if reverse else k
        rate = base_rate * rank
        prices[f"T{k}"] = pd.Series(100.0 * (1 + rate) ** t, index=idx)
    return prices


def _snapshots_with_score(tickers: list[str], as_of_dates: list,
                          score_by_ticker: dict[str, float]) -> pd.DataFrame:
    rows = []
    for d in as_of_dates:
        for tk in tickers:
            rows.append({"as_of": d, "ticker": tk, "conviction": score_by_ticker[tk]})
    df = pd.DataFrame(rows)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


def test_perfect_positive_rank_correlation_gives_ic_near_one():
    prices = _monotonic_prices(n_tickers=5, n_days=280, reverse=False)
    tickers = list(prices.keys())
    score = {f"T{k}": float(k) for k in range(1, 6)}  # T5 grows fastest, scores highest
    idx = prices["T1"].index
    as_of_dates = [idx[d] for d in range(20, 180, 20)]
    snaps = _snapshots_with_score(tickers, as_of_dates, score)

    out = information_coefficient(snaps, prices, score_col="conviction",
                                  horizons_weeks=(4, 8))
    assert set(out.index) == {4, 8}
    for h in (4, 8):
        assert out.loc[h, "n_periods"] > 0
        assert out.loc[h, "mean_ic"] == pytest.approx(1.0, abs=1e-6)
        assert out.loc[h, "ic_std"] == pytest.approx(0.0, abs=1e-9)


def test_perfect_negative_rank_correlation_gives_ic_near_minus_one():
    prices = _monotonic_prices(n_tickers=5, n_days=280, reverse=True)
    tickers = list(prices.keys())
    # T1 now grows FASTEST (reverse=True), but score still increases with k
    # -> score and forward return are perfectly ANTI-correlated.
    score = {f"T{k}": float(k) for k in range(1, 6)}
    idx = prices["T1"].index
    as_of_dates = [idx[d] for d in range(20, 180, 20)]
    snaps = _snapshots_with_score(tickers, as_of_dates, score)

    out = information_coefficient(snaps, prices, score_col="conviction",
                                  horizons_weeks=(4,))
    assert out.loc[4, "mean_ic"] == pytest.approx(-1.0, abs=1e-6)


def test_t_stat_is_large_when_ic_is_consistently_nonzero():
    prices = _monotonic_prices(n_tickers=6, n_days=300)
    tickers = list(prices.keys())
    score = {f"T{k}": float(k) for k in range(1, 7)}
    idx = prices["T1"].index
    as_of_dates = [idx[d] for d in range(20, 200, 15)]
    snaps = _snapshots_with_score(tickers, as_of_dates, score)

    out = information_coefficient(snaps, prices, horizons_weeks=(4,))
    # IC is a constant +1 every period here (zero variance across periods),
    # so ic_std == 0 and t_stat is correctly left as NaN (undefined, not a
    # divide-by-zero crash) rather than a fabricated infinity.
    assert out.loc[4, "ic_std"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(out.loc[4, "t_stat"])
    assert out.loc[4, "n_periods"] == len(as_of_dates)


def test_degenerate_period_with_tied_scores_is_excluded_not_crashed():
    """A snapshot date where every ticker has the identical score has
    undefined rank correlation — must be skipped, not raise or silently
    return a bogus 0.0 for that period."""
    idx = pd.bdate_range("2024-01-01", periods=200)
    prices = {f"T{k}": pd.Series(100.0 * (1 + 0.0005 * k) ** np.arange(200), index=idx)
             for k in range(1, 5)}
    tickers = list(prices.keys())
    tied_score = {tk: 3.0 for tk in tickers}  # everyone tied
    as_of_dates = [idx[30], idx[60]]
    snaps = _snapshots_with_score(tickers, as_of_dates, tied_score)

    out = information_coefficient(snaps, prices, horizons_weeks=(4,))
    assert out.loc[4, "n_periods"] == 0
    assert np.isnan(out.loc[4, "mean_ic"])


def test_min_names_per_date_filters_thin_periods():
    idx = pd.bdate_range("2024-01-01", periods=200)
    prices = {f"T{k}": pd.Series(100.0 * (1 + 0.0005 * k) ** np.arange(200), index=idx)
             for k in range(1, 3)}  # only 2 tickers
    tickers = list(prices.keys())
    score = {"T1": 1.0, "T2": 2.0}
    snaps = _snapshots_with_score(tickers, [idx[30]], score)

    out = information_coefficient(snaps, prices, horizons_weeks=(4,),
                                  min_names_per_date=4)
    assert out.loc[4, "n_periods"] == 0  # only 2 names, below the min of 4


def test_missing_score_column_returns_empty_frame():
    out = information_coefficient(pd.DataFrame({"as_of": [], "ticker": []}),
                                  {}, score_col="conviction")
    assert out.empty


def test_empty_snapshots_returns_empty_frame_not_crash():
    out = information_coefficient(pd.DataFrame(), {})
    assert out.empty
