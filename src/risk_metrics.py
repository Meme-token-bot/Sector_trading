"""Portfolio-level risk block: correlation, concentration, historical VaR/ES.

Pure functions, no IO — same convention as macro_alignment.py /
regime_analysis.py / regime_snapshot.py. Answers TRADING_EDGE_AUDIT.md items
A1/A2: four NEW_BUY sectors that each look independently fine can still be
one bet wearing four hats, and nothing in the pipeline before this module
could tell you that.

Scope note, deliberately: this is the Phase-2 stopgap, not a full risk
system. It uses HISTORICAL (empirical) VaR/ES off realized weekly returns —
no parametric-normal assumption, since sector-ETF returns at this sample
size are not obviously normal and a fat-tailed empirical method is the
honest default. It does NOT do PCA-based factor decomposition (deferred to
Phase 3 per the audit — the diversification-ratio heuristic below gets most
of the practical value for a fraction of the complexity, and stays
auditable by eye, which matters more than precision at this stage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _weekly_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Weekly (Friday-anchored) simple returns for each ticker — matches the
    strategy's own weekly-rebalance cadence rather than a daily return
    scaled by sqrt(5), which would silently assume i.i.d. daily returns."""
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return pd.DataFrame()
    sub = prices[cols].dropna(how="all")
    weekly = sub.resample("W-FRI").last().ffill()
    return weekly.pct_change().dropna(how="all")


def compute_correlation_matrix(prices: pd.DataFrame, tickers: list[str],
                               lookback_weeks: int = 52) -> pd.DataFrame:
    """Trailing `lookback_weeks` correlation matrix of weekly returns across
    `tickers`. Empty frame if fewer than 2 usable tickers or too little
    history to form a meaningful correlation (< 4 weekly observations)."""
    rets = _weekly_returns(prices, tickers)
    if rets.empty:
        return pd.DataFrame()
    rets = rets.tail(lookback_weeks).dropna(axis=1, how="all")
    if rets.shape[1] < 2 or len(rets) < 4:
        return pd.DataFrame()
    return rets.corr()


def average_pairwise_correlation(corr: pd.DataFrame) -> float:
    """Mean of the off-diagonal entries of a correlation matrix. NaN if
    `corr` has fewer than 2 names."""
    if corr is None or corr.empty or corr.shape[0] < 2:
        return float("nan")
    n = corr.shape[0]
    arr = corr.to_numpy()
    off_diag_sum = arr.sum() - np.trace(arr)
    n_pairs = n * (n - 1)
    return float(off_diag_sum / n_pairs) if n_pairs else float("nan")


def concentration_metrics(weights: pd.Series,
                          corr: pd.DataFrame | None = None,
                          vols: pd.Series | None = None) -> dict:
    """HHI / effective-N on `weights`, plus a correlation-adjusted
    effective-N when `corr` and `vols` (annualized per-ticker vol) are both
    supplied.

    Returns:
      {
        "hhi": float,                          # sum(w_i^2), w renormalized over w>0
        "effective_n_naive": float,             # 1/hhi — ignores correlation entirely
        "effective_n_corr_adjusted": float|None,# heuristic, see docstring
        "diversification_ratio": float|None,
      }

    The correlation-adjusted figure is a HEURISTIC — the squared
    diversification ratio, (weighted-avg vol / portfolio vol)^2 — not an
    exact PCA-derived quantity. It equals 1.0 when every held name is
    perfectly correlated (no diversification benefit no matter how many
    names you hold) and rises toward `effective_n_naive` as correlation
    falls, converging to it exactly when correlation is zero. Good enough
    to catch "four sectors, one bet" at a glance; not a substitute for real
    factor analysis.
    """
    w = weights[weights > 0]
    if w.empty:
        return {"hhi": float("nan"), "effective_n_naive": float("nan"),
                "effective_n_corr_adjusted": None, "diversification_ratio": None}
    w_norm = w / w.sum()
    hhi = float((w_norm ** 2).sum())
    eff_n_naive = float(1.0 / hhi) if hhi > 0 else float("nan")

    eff_n_corr = None
    div_ratio = None
    if corr is not None and vols is not None and not corr.empty:
        names = [t for t in w_norm.index if t in corr.index and t in vols.index]
        if len(names) >= 2:
            wv = w_norm.loc[names]
            wv = wv / wv.sum()
            sigma = vols.loc[names].astype(float)
            weighted_avg_vol = float((wv * sigma).sum())
            sub_corr = corr.loc[names, names].to_numpy()
            cov = sub_corr * np.outer(sigma.to_numpy(), sigma.to_numpy())
            wv_arr = wv.to_numpy()
            port_var = float(wv_arr @ cov @ wv_arr)
            port_vol = float(np.sqrt(max(port_var, 0.0)))
            if port_vol > 0:
                div_ratio = weighted_avg_vol / port_vol
                eff_n_corr = float(div_ratio ** 2)

    return {"hhi": hhi, "effective_n_naive": eff_n_naive,
           "effective_n_corr_adjusted": eff_n_corr,
           "diversification_ratio": div_ratio}


def historical_var_es(prices: pd.DataFrame, weights: pd.Series,
                      lookback_weeks: int = 104,
                      confidence: float = 0.95) -> dict:
    """Empirical (historical) 1-week VaR and Expected Shortfall for a
    weighted portfolio, off REALIZED weekly returns — no parametric-normal
    assumption.

    Returns {"var": float, "es": float, "n_weeks": int}. Both `var` and `es`
    are NEGATIVE numbers (a loss), matching how drawdowns are already
    signed elsewhere in this codebase (e.g. `src/backtest.py::_max_drawdown`)
    — `var=-0.024` means "a 2.4% weekly loss at the 95th percentile," not a
    positive magnitude you have to remember to negate. NaN / n_weeks=0 when
    there's fewer than 8 weeks of usable history or the weights are empty.
    """
    w = weights[weights > 0]
    if w.empty:
        return {"var": float("nan"), "es": float("nan"), "n_weeks": 0}
    w_norm = w / w.sum()
    tickers = [t for t in w_norm.index if t in prices.columns]
    if not tickers:
        return {"var": float("nan"), "es": float("nan"), "n_weeks": 0}

    rets = _weekly_returns(prices, tickers).tail(lookback_weeks)
    rets = rets.reindex(columns=tickers).fillna(0.0)
    if len(rets) < 8:
        return {"var": float("nan"), "es": float("nan"), "n_weeks": 0}

    w_aligned = w_norm.reindex(tickers).fillna(0.0)
    port_rets = (rets * w_aligned).sum(axis=1)
    alpha = 1.0 - confidence
    var_threshold = float(np.percentile(port_rets.to_numpy(), alpha * 100))
    tail = port_rets[port_rets <= var_threshold]
    es = float(tail.mean()) if not tail.empty else var_threshold
    return {"var": var_threshold, "es": es, "n_weeks": int(len(port_rets))}


def annualized_vol_by_ticker(prices: pd.DataFrame, tickers: list[str],
                             lookback_weeks: int = 52) -> pd.Series:
    """Annualized vol (weekly return stdev * sqrt(52)) per ticker — the
    `vols` input `concentration_metrics` wants. Separated out so callers who
    already have a vol estimate elsewhere aren't forced to recompute it."""
    rets = _weekly_returns(prices, tickers).tail(lookback_weeks)
    if rets.empty:
        return pd.Series(dtype=float)
    return rets.std(ddof=0) * np.sqrt(52)
