"""Synthetic-series tests for src.expression_signals.

No DB, no yfinance. All series are built with `pd.date_range` + arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.expressions import Expression
from config.settings import PARAMS
from src.expression_signals import (
    ExpressionSignal,
    blend_theme_sentiment,
    build_theme_sentiment_loader,
    compute_expression_signal,
    compute_expressions_for_sector,
    rank_expressions,
)


def _series(values, end: str = "2026-05-20") -> pd.Series:
    """Build an ascending date-indexed close series from a list/array."""
    arr = np.asarray(values, dtype=float)
    idx = pd.date_range(end=end, periods=len(arr), freq="B")
    return pd.Series(arr, index=idx, name="close")


def _flat(level: float, n: int) -> pd.Series:
    return _series([level] * n)


def _linear(start: float, end: float, n: int) -> pd.Series:
    return _series(np.linspace(start, end, n))


def _make_expr(ticker: str = "FOO", beta: float = 1.0) -> Expression:
    return Expression(ticker=ticker, label="test", kind="thematic", beta_hint=beta)


def test_no_data():
    expr = _make_expr()
    sig = compute_expression_signal(
        expr, "NEW_BUY", pd.Series(dtype=float), _flat(100, 300)
    )
    assert sig.state == "NO_DATA"
    assert "Update price data" in sig.reason
    assert sig.above_own_sma is None
    assert sig.own_extension_pct is None
    assert sig.own_return_3m is None


def test_warming_up_short_history():
    # 100 bars: not enough for SMA200, but enough for 3m return (63 bars).
    expr = _make_expr()
    own = _linear(100, 110, 100)
    parent = _linear(100, 105, 300)
    sig = compute_expression_signal(expr, "NEW_BUY", own, parent)
    assert sig.state == "WARMING_UP"
    assert "100 bars" in sig.reason
    assert sig.own_return_3m is not None
    assert sig.parent_return_3m is not None
    assert sig.rs_vs_parent is not None
    assert sig.above_own_sma is None
    assert sig.own_extension_pct is None


def test_warming_up_very_short():
    expr = _make_expr()
    own = _linear(100, 102, 20)
    parent = _linear(100, 105, 300)
    sig = compute_expression_signal(expr, "NEW_BUY", own, parent)
    assert sig.state == "WARMING_UP"
    assert sig.own_return_3m is None
    assert sig.rs_vs_parent is None


def test_parent_inactive_passthrough():
    expr = _make_expr()
    own = _linear(80, 110, 300)
    parent = _linear(80, 105, 300)
    sig = compute_expression_signal(expr, "HOLD", own, parent)
    assert sig.state == "PARENT_INACTIVE"
    assert "HOLD" in sig.reason
    # All diagnostics populated.
    assert sig.above_own_sma is True
    assert sig.own_extension_pct is not None
    assert sig.own_return_3m is not None
    assert sig.parent_return_3m is not None
    assert sig.rs_vs_parent is not None
    assert sig.beta_scaled_cutoff == pytest.approx(PARAMS.extension_pct_cutoff)


def test_confirmed_leads_parent():
    # Own outperforms parent on 3m, mild extension below cutoff.
    expr = _make_expr(beta=1.0)
    # SMA200 ≈ mean of last 200 bars; engineer modest extension.
    own = _linear(95, 105, 300)
    parent = _linear(98, 102, 300)
    sig = compute_expression_signal(expr, "NEW_BUY", own, parent)
    assert sig.state == "CONFIRMED"
    assert sig.rs_vs_parent is not None and sig.rs_vs_parent > 0
    assert sig.above_own_sma is True


def test_confirmed_matches_parent():
    # Boundary: identical trajectories → rs_vs_parent == 0 → CONFIRMED.
    expr = _make_expr(beta=1.0)
    own = _linear(95, 105, 300)
    parent = own.copy()
    sig = compute_expression_signal(expr, "NEW_BUY", own, parent)
    assert sig.state == "CONFIRMED"
    assert sig.rs_vs_parent == pytest.approx(0.0)


def test_lagging():
    # Both uptrend & above own SMA, but own < parent on 3m.
    expr = _make_expr(beta=1.0)
    own = _linear(98, 102, 300)        # +4% total, modest 3m
    parent = _linear(90, 110, 300)     # much stronger 3m
    sig = compute_expression_signal(expr, "NEW_BUY", own, parent)
    assert sig.state == "LAGGING"
    assert sig.rs_vs_parent is not None and sig.rs_vs_parent < 0


def test_broken():
    # Last close below own SMA200.
    expr = _make_expr(beta=1.0)
    # Strong uptrend then sharp recent drop: SMA stays high, last < SMA.
    body = list(np.linspace(80, 130, 250))
    tail = list(np.linspace(130, 60, 50))
    own = _series(body + tail)
    parent = _linear(100, 110, 300)
    sig = compute_expression_signal(expr, "NEW_BUY", own, parent)
    assert sig.state == "BROKEN"
    assert sig.above_own_sma is False
    assert sig.own_extension_pct is not None and sig.own_extension_pct < 0


def test_stretched_default_beta():
    # 15% above own SMA200, beta_hint 1.0, cutoff = 12% → STRETCHED.
    expr = _make_expr(beta=1.0)
    # Make SMA200 ≈ 100, last ≈ 115.
    body = list(np.full(280, 100.0))
    tail = list(np.linspace(100, 115, 20))
    own = _series(body + tail)
    parent = _linear(100, 105, 300)
    sig = compute_expression_signal(expr, "NEW_BUY", own, parent)
    assert sig.state == "STRETCHED"
    assert sig.own_extension_pct is not None and sig.own_extension_pct > PARAMS.extension_pct_cutoff


def test_stretched_high_beta_passes():
    # Same 15% extension, beta_hint 2.5, cutoff = 30% → not STRETCHED.
    expr = _make_expr(beta=2.5)
    body = list(np.full(280, 100.0))
    tail = list(np.linspace(100, 115, 20))
    own = _series(body + tail)
    parent = _linear(100, 105, 300)
    sig = compute_expression_signal(expr, "NEW_BUY", own, parent)
    assert sig.state != "STRETCHED"
    # Should land CONFIRMED — own outperforms parent on 3m here.
    assert sig.state == "CONFIRMED"


def test_baseline_sector_etf():
    # When the expression IS the sector, parent_close == expression_close,
    # rs_vs_parent == 0, and rule 7 (strict <) falls through to CONFIRMED.
    expr = _make_expr(ticker="XLK", beta=1.0)
    series = _linear(95, 105, 300)
    sig = compute_expression_signal(expr, "NEW_BUY", series, series)
    assert sig.state == "CONFIRMED"
    assert sig.rs_vs_parent == pytest.approx(0.0)


def test_compute_expressions_for_sector_no_parent_data(monkeypatch):
    """If parent close is empty, every expression should return NO_DATA."""
    def loader(ticker: str) -> pd.Series:
        # Parent (XLK) has no data; expressions also empty for simplicity.
        return pd.Series(dtype=float)

    results = compute_expressions_for_sector("XLK", "NEW_BUY", loader)
    assert results, "should still return one signal per expression"
    assert all(r.state == "NO_DATA" for r in results)
    assert all("parent ETF has no price data" in r.reason for r in results)


# ---------------------------------------------------------------------------
# Theme-news overlay
# ---------------------------------------------------------------------------

def test_theme_key_resolved_from_ticker():
    # SOXX is a SEMIS expression; FOO maps to no theme.
    sig = compute_expression_signal(
        _make_expr(ticker="SOXX"), "NEW_BUY", _linear(95, 105, 300),
        _linear(98, 102, 300))
    assert sig.theme_key == "SEMIS"
    sig2 = compute_expression_signal(
        _make_expr(ticker="FOO"), "NEW_BUY", _linear(95, 105, 300),
        _linear(98, 102, 300))
    assert sig2.theme_key is None


def test_news_flag_contradicts_on_confirmed_bad_news():
    sig = compute_expression_signal(
        _make_expr(ticker="SOXX"), "NEW_BUY", _linear(95, 105, 300),
        _linear(98, 102, 300), theme_sentiment=-3.0, theme_n_obs=4)
    assert sig.state == "CONFIRMED"
    assert sig.news_flag == "NEWS_CONTRADICTS"


def test_news_flag_divergence_on_broken_good_news():
    body = list(np.linspace(80, 130, 250))
    tail = list(np.linspace(130, 60, 50))
    sig = compute_expression_signal(
        _make_expr(ticker="GDX"), "NEW_BUY", _series(body + tail),
        _linear(100, 110, 300), theme_sentiment=3.0, theme_n_obs=3)
    assert sig.state == "BROKEN"
    assert sig.news_flag == "NEWS_DIVERGENCE"


def test_no_flag_without_theme_sentiment():
    sig = compute_expression_signal(
        _make_expr(ticker="SOXX"), "NEW_BUY", _linear(95, 105, 300),
        _linear(98, 102, 300))
    assert sig.news_flag is None
    assert sig.theme_sentiment is None


def test_blend_theme_sentiment():
    # both: (1-0.4)*4 + 0.4*(-1) = 2.0
    assert blend_theme_sentiment(4.0, 2, -1.0, 5, 0.4) == (2.0, 7)
    # one-sided
    assert blend_theme_sentiment(3.0, 2, None, 0) == (3.0, 2)
    assert blend_theme_sentiment(None, 0, -2.0, 4) == (-2.0, 4)
    # neither
    assert blend_theme_sentiment(None, 0, None, 0) == (None, 0)


def _sig(ticker, state, theme_sentiment=None):
    return ExpressionSignal(
        ticker=ticker, state=state, reason="", above_own_sma=None,
        own_extension_pct=None, own_return_3m=None, parent_return_3m=None,
        rs_vs_parent=None, beta_scaled_cutoff=None,
        theme_sentiment=theme_sentiment)


def test_rank_expressions_state_then_news():
    sigs = [
        _sig("A", "LAGGING", 1.0),
        _sig("B", "CONFIRMED", -2.0),
        _sig("C", "CONFIRMED", 4.0),
        _sig("D", "BROKEN", 5.0),
    ]
    ranked = [s.ticker for s in rank_expressions(sigs)]
    # CONFIRMED before LAGGING before BROKEN; within CONFIRMED, higher news first.
    assert ranked == ["C", "B", "A", "D"]


def test_build_theme_sentiment_loader_blends_and_skips_plain():
    nl = pd.DataFrame({"score": [3.0], "n_obs": [2]},
                      index=pd.Index(["URANIUM"], name="theme_key"))
    news = pd.DataFrame({"score": [4.0], "n_headlines": [5]},
                        index=pd.Index(["URANIUM"], name="theme_key"))
    loader = build_theme_sentiment_loader(nl, news, weight=0.4)
    score, n = loader("URA")        # URA is a URANIUM expression
    assert score == pytest.approx(0.6 * 3.0 + 0.4 * 4.0)
    assert n == 7
    assert loader("XLB") == (None, 0)   # plain sector proxy → no theme


def test_compute_expressions_for_sector_passes_theme_overlay():
    def ohlcv(ticker: str) -> pd.Series:
        return _linear(95, 105, 300)

    def theme_loader(ticker: str):
        return (-4.0, 3) if ticker in ("SOXX", "SMH") else (None, 0)

    results = compute_expressions_for_sector(
        "XLK", "NEW_BUY", ohlcv, theme_sentiment_loader=theme_loader)
    by = {r.ticker: r for r in results}
    assert by["SOXX"].theme_sentiment == -4.0
    assert by["SOXX"].news_flag == "NEWS_CONTRADICTS"  # CONFIRMED + bad news
    # plain proxy carries no theme overlay
    assert by["XLK"].theme_sentiment is None
