"""Sentiment trend reconstruction.

Walks weekly snapshot dates and rebuilds the rolling-window aggregate at
each snapshot. Pure function over the database — no schema changes.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pandas as pd

from config.settings import DB_PATH, PARAMS, SECTOR_ETFS


def _earliest_newsletter_date() -> date | None:
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute("SELECT MIN(publication_date) FROM newsletters").fetchone()
    if not row or not row[0]:
        return None
    return date.fromisoformat(row[0])


def _aggregate_at(conn: sqlite3.Connection, as_of: date,
                  lookback_days: int) -> pd.Series:
    cutoff = as_of - timedelta(days=lookback_days)
    df = pd.read_sql_query(
        """SELECT sr.ticker, AVG(sr.sentiment_score) AS score
           FROM sector_ratings sr
           JOIN newsletters n ON n.id = sr.newsletter_id
           WHERE n.publication_date BETWEEN ? AND ?
           GROUP BY sr.ticker""",
        conn, params=(cutoff.isoformat(), as_of.isoformat()),
    )
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("ticker")["score"]


def build_sentiment_trend(lookback_days: int | None = None,
                          end: date | None = None) -> pd.DataFrame:
    """Weekly snapshots from earliest newsletter to `end` (default today).

    Returns a DataFrame indexed by week-ending date with one column per
    sector ticker. Cells are the rolling-window average sentiment as it
    would have been computed on that snapshot date. NaN means no coverage
    in that sector's window for that week.
    """
    lookback = lookback_days or PARAMS.sentiment_lookback_days
    end = end or date.today()

    start = _earliest_newsletter_date()
    if start is None:
        return pd.DataFrame(columns=list(SECTOR_ETFS.keys()))

    snapshots: list[date] = []
    cur = start
    while cur <= end:
        snapshots.append(cur)
        cur += timedelta(days=7)
    if snapshots[-1] != end:
        snapshots.append(end)

    rows: dict[date, pd.Series] = {}
    with sqlite3.connect(DB_PATH) as conn:
        for d in snapshots:
            rows[d] = _aggregate_at(conn, d, lookback)

    df = pd.DataFrame(rows).T
    df = df.reindex(columns=list(SECTOR_ETFS.keys()))
    df.index = pd.to_datetime(df.index)
    df.index.name = "week"
    return df
