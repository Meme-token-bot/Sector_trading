"""Fund Screener — composite 0-100 score per candidate ticker within a theme.

Pure functions, no IO — same convention as src/industry_rotation.py and
src/risk_metrics.py. Never imports sqlite3, requests, or yfinance; IO
(price history via src.price_store, fund metadata via src.fund_data)
happens entirely in the caller (app.py), matching src/regime_snapshot.py's
split between pure computation and the cached IO wrapper that feeds it.

Never wired into position sizing, src/signals.py, or target_weights — this
is a discovery tool, not the trading model (see config/fund_screener_themes.py's
module docstring for the full non-goals list).
"""
from __future__ import annotations

import pandas as pd

from config.fund_screener_themes import ScreenerTheme
from config.settings import FUND_SCORE_PARAMS, FundScoreParams
from src.fund_data import FundRecord

_OUTPUT_COLUMNS = ["name", "expense_ratio", "aum", "return_1y", "score", "score_label"]


def _total_return(close: pd.Series | None, lookback_days: int) -> float | None:
    """Simple total return over the trailing `lookback_days`.

    Degrades to whatever history is actually available rather than
    returning None outright — but only down to HALF the requested lookback
    (below that, a return figure would be comparing too short a window to
    be meaningful against funds that DO have full history, so it's
    excluded instead of silently distorting the ranking).
    """
    if close is None:
        return None
    close = close.dropna()
    min_needed = max(20, lookback_days // 2)
    if len(close) < min_needed + 1:
        return None
    offset = min(lookback_days, len(close) - 1)
    start, end = float(close.iloc[-offset - 1]), float(close.iloc[-1])
    if start == 0:
        return None
    return end / start - 1.0


def _percentile_rank(s: pd.Series) -> pd.Series:
    """0..1 percentile rank within `s`, NaN-safe.

    A missing value gets the neutral midpoint (0.5) — never a penalty for
    data the fetcher simply couldn't get. When there are fewer than 2
    distinct real values to rank (e.g. a single-candidate theme, or every
    candidate missing the same field), EVERYONE gets 0.5 rather than an
    arbitrary tie-break order, since there's no genuine spread to rank on.
    """
    if s.dropna().nunique() < 2:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True).fillna(0.5)


def score_label(score: int) -> str:
    """'68 · Good' style label for a 0-100 score."""
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 45:
        return "Fair"
    return "Weak"


def score_theme_funds(
    theme: ScreenerTheme,
    prices: pd.DataFrame | None,
    fund_metadata: dict[str, FundRecord] | None,
    params: FundScoreParams = FUND_SCORE_PARAMS,
) -> pd.DataFrame:
    """Composite 0-100 score for every ticker in `theme.tickers`.

    `prices` is a wide close-price frame (columns = tickers, whichever are
    actually available — missing columns degrade that ticker's return
    component to the neutral midpoint, never a crash). `fund_metadata` is
    `{ticker: FundRecord}` from src.fund_data.fetch_fund_metadata; a
    missing or error-flagged entry degrades expense_ratio/aum to None the
    same way.

    Returns a DataFrame indexed by ticker (covering EVERY ticker in
    `theme.tickers`, even ones with zero usable data — score defaults to
    the neutral ~50 in that case rather than being silently dropped),
    columns: name, expense_ratio, aum, return_1y, score, score_label.
    Sorted by score descending; ties broken by ticker for determinism.
    """
    tickers = list(theme.tickers)
    fund_metadata = fund_metadata or {}

    returns: dict[str, float | None] = {}
    for t in tickers:
        col = prices[t] if (prices is not None and t in prices.columns) else None
        returns[t] = _total_return(col, params.return_lookback_days)

    names = {t: (fund_metadata[t].name if t in fund_metadata and fund_metadata[t].name else t)
             for t in tickers}
    expense_ratios = {t: (fund_metadata[t].expense_ratio if t in fund_metadata else None)
                       for t in tickers}
    aums = {t: (fund_metadata[t].aum if t in fund_metadata else None) for t in tickers}

    idx = pd.Index(tickers, name="ticker")
    ret_s = pd.Series(returns, index=idx, dtype=float)
    er_s = pd.Series(expense_ratios, index=idx, dtype=float)
    aum_s = pd.Series(aums, index=idx, dtype=float)

    return_component = _percentile_rank(ret_s)
    liquidity_component = _percentile_rank(aum_s)
    fee_component = 1.0 - _percentile_rank(er_s)  # LOWER expense ratio scores higher

    raw = (params.return_weight * return_component
          + params.liquidity_weight * liquidity_component
          + params.fee_weight * fee_component)
    score = (raw * 100.0).round().clip(lower=0, upper=100).astype(int)

    out = pd.DataFrame({
        "name": pd.Series(names, index=idx),
        "expense_ratio": er_s,
        "aum": aum_s,
        "return_1y": ret_s,
        "score": score,
    }, index=idx)
    out["score_label"] = out["score"].map(score_label)
    out = out.sort_values(["score", "ticker"], ascending=[False, True],
                          kind="mergesort")  # stable sort, deterministic ticker tiebreak
    return out[_OUTPUT_COLUMNS]