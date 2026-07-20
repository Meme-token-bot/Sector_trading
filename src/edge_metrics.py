"""Signal-quality metrics: information coefficient and decay curve.

Answers a question nothing in the pipeline answers today: is the model's
cross-sectional RANKING itself predictive, independent of the discrete
BUY/HOLD/SELL convergence rule? The IC — here, the cross-sectional Spearman
rank correlation between a score at time t and each ticker's forward return
at t+k — is the standard signal-quality metric in cross-sectional equity
work, and it's completely absent from this project today
(TRADING_EDGE_AUDIT.md item B2).

Pure functions, no IO — same convention as `regime_analysis.py` /
`macro_alignment.py` / `regime_snapshot.py`. Reads whatever `signal_snapshots`
-shaped DataFrame and price history the caller passes in; callers are
responsible for sourcing both (typically `src.db.load_signal_snapshots()`
and the cached price panel already loaded elsewhere in the app).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_return(prices: dict[str, pd.Series], ticker: str,
                    as_of: pd.Timestamp, horizon_weeks: int) -> float | None:
    """Simple forward total return for `ticker` from `as_of` to
    `as_of + horizon_weeks`. None if the ticker or the horizon's price data
    isn't available (e.g. too close to the end of the loaded price history).
    """
    s = prices.get(ticker)
    if s is None:
        return None
    s = s.dropna()
    if s.empty:
        return None
    entry_i = s.index.searchsorted(as_of, side="left")
    exit_ts = as_of + pd.Timedelta(weeks=horizon_weeks)
    exit_i = s.index.searchsorted(exit_ts, side="left")
    if entry_i >= len(s) or exit_i >= len(s) or exit_i <= entry_i:
        return None
    e, x = float(s.iloc[entry_i]), float(s.iloc[exit_i])
    if e == 0:
        return None
    return x / e - 1.0


def information_coefficient(
    snapshots: pd.DataFrame,
    prices: dict[str, pd.Series],
    score_col: str = "conviction",
    horizons_weeks: tuple[int, ...] = (1, 2, 4, 8, 12),
    min_names_per_date: int = 4,
) -> pd.DataFrame:
    """Mean cross-sectional Spearman IC of `score_col` vs. forward return,
    at each horizon in `horizons_weeks` — i.e. the signal-decay curve.

    Methodology (standard practitioner convention, not invented here):
    within each snapshot date, rank tickers by `score_col` and separately by
    forward return, Spearman-correlate the two rankings (the "period IC"),
    then average the period ICs across all dates that had at least
    `min_names_per_date` tickers with both a score and a resolvable forward
    return. Averaging PERIOD ICs — rather than pooling every (score, return)
    pair across all dates into one giant correlation — is deliberate: it
    stops a handful of unusually volatile weeks from dominating the
    estimate, and it's what "IC" conventionally means when someone reports
    one.

    `score_col` defaults to `conviction` (0-5) but any numeric column already
    on `snapshots` works, e.g. `relative_strength_3m` or `rs_rank` (rank
    reversed — lower rank = stronger, so expect a NEGATIVE IC there, not a
    sign error).

    Returns a DataFrame indexed by `horizon_weeks` with columns:
      mean_ic, ic_std, n_periods, t_stat
    `t_stat = mean_ic / (ic_std / sqrt(n_periods))` is the standard "is this
    distinguishable from zero" statistic — treat it as a first-pass filter,
    NOT a significance verdict, given how few independent weekly periods
    this system has accumulated so far. `n_periods` is exactly the number
    you should look at before trusting `mean_ic` at all: with only a
    handful of periods, a big-looking IC and a t_stat near zero are both
    telling you the same thing — not enough data yet, keep collecting.
    """
    empty_cols = ["mean_ic", "ic_std", "n_periods", "t_stat"]
    if snapshots is None or snapshots.empty or score_col not in snapshots.columns:
        return pd.DataFrame(columns=empty_cols).rename_axis("horizon_weeks")

    rows: list[dict] = []
    for h in horizons_weeks:
        period_ics: list[float] = []
        for as_of, grp in snapshots.groupby("as_of"):
            valid = grp.dropna(subset=[score_col])
            if len(valid) < min_names_per_date:
                continue
            fwd = {
                t: _forward_return(prices, t, pd.Timestamp(as_of), h)
                for t in valid["ticker"]
            }
            fwd_series = pd.Series(fwd).dropna()
            if len(fwd_series) < min_names_per_date:
                continue
            scores = valid.set_index("ticker")[score_col].reindex(fwd_series.index)
            scores = scores.dropna()
            fwd_series = fwd_series.reindex(scores.index)
            if len(scores) < min_names_per_date:
                continue
            # Spearman correlation is undefined (or a meaningless divide-by-
            # zero) when either side has zero variance — e.g. every sector
            # tied at the same conviction score that week.
            if scores.nunique() < 2 or fwd_series.nunique() < 2:
                continue
            ic = scores.corr(fwd_series, method="spearman")
            if pd.notna(ic):
                period_ics.append(float(ic))

        if period_ics:
            arr = np.array(period_ics, dtype=float)
            n = len(arr)
            std = float(arr.std(ddof=1)) if n > 1 else float("nan")
            t_stat = (float(arr.mean()) / (std / np.sqrt(n))
                      if n > 1 and std > 0 else float("nan"))
            rows.append({"horizon_weeks": h, "mean_ic": float(arr.mean()),
                        "ic_std": std, "n_periods": n, "t_stat": t_stat})
        else:
            rows.append({"horizon_weeks": h, "mean_ic": float("nan"),
                        "ic_std": float("nan"), "n_periods": 0,
                        "t_stat": float("nan")})

    return pd.DataFrame(rows).set_index("horizon_weeks")
