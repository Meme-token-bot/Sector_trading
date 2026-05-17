"""Replay weekly BUY/HOLD/SELL signals over recent history.

Used by the state-aware classifier in src/signals.py to detect:
- Sectors that just turned BUY (NEW_BUY — fresh entry OK)
- Sectors that have been BUY for too long (HOLD_IF_LONG — don't chase)
- Sectors that recently degraded from BUY (REDUCE)

Pure function over (prices, sentiment_db). Stateless. Same inputs → same output.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from config.settings import PARAMS, SECTOR_ETFS
from src.db import aggregate_sentiment
from src.market_engine import compute_sector_metrics
from src.signals import build_signals


def build_signal_history(prices: pd.DataFrame,
                         end: date | None = None,
                         n_weeks: int | None = None) -> pd.DataFrame:
    """Weekly signal snapshots, oldest -> newest.

    Returns DataFrame with index=week date, columns=sector tickers,
    values=signal label ('BUY' / 'HOLD' / 'SELL'). Missing data => NaN.
    """
    end = end or date.today()
    n = n_weeks or PARAMS.history_weeks

    snapshots: dict[pd.Timestamp, pd.Series] = {}
    for i in range(n - 1, -1, -1):
        d = end - timedelta(weeks=i)
        m = compute_sector_metrics(prices, as_of=pd.Timestamp(d))
        if m.empty:
            continue
        s = aggregate_sentiment(as_of=d)
        sig = build_signals(m, s)
        snapshots[pd.Timestamp(d)] = sig["signal"]

    if not snapshots:
        return pd.DataFrame(columns=list(SECTOR_ETFS.keys()))

    df = pd.DataFrame(snapshots).T
    df = df.reindex(columns=list(SECTOR_ETFS.keys()))
    df.index.name = "week"
    return df.sort_index()


def consecutive_buy_weeks(history: pd.DataFrame) -> pd.Series:
    """For each sector, count how many consecutive most-recent weekly
    snapshots had signal == 'BUY'. Returns Series indexed by ticker.
    """
    if history.empty:
        return pd.Series(dtype=int)

    # Walk backwards from the latest snapshot; stop on first non-BUY per ticker.
    rev = history.iloc[::-1]
    counts: dict[str, int] = {}
    for tkr in rev.columns:
        n = 0
        for v in rev[tkr]:
            if v == "BUY":
                n += 1
            else:
                break
        counts[tkr] = n
    return pd.Series(counts, name="consecutive_buy_weeks")
