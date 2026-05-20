"""SQLite persistence for OHLCV price data.

Separate from `sentiment.db` because price history is large, churns differently,
and is purely a cache of yfinance-adjusted bars. Schema is intentionally narrow
(no fundamentals, no intraday). yfinance returns split/dividend-adjusted prices
when `auto_adjust=True` — so a split that happens *today* retroactively rewrites
*every historical close*. Naive append would silently corrupt history; see
`update_ticker` for the reconciliation logic.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Callable, Iterator

import pandas as pd

from config.settings import DATA_DIR

PRICES_DB_PATH = DATA_DIR / "prices.db"

# Cold-start window. ~5 years of business days.
COLD_START_DAYS = 1825
# Overlap window for incremental updates — also the split-detection window.
OVERLAP_DAYS = 60
# Tolerance for split detection. >0.5% close mismatch on an overlapping bar
# is treated as a split/dividend re-adjustment.
SPLIT_TOL = 0.005


SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker       TEXT     NOT NULL,
    timeframe    TEXT     NOT NULL CHECK (timeframe IN ('1d', '1wk')),
    bar_date     DATE     NOT NULL,
    open         REAL     NOT NULL,
    high         REAL     NOT NULL,
    low          REAL     NOT NULL,
    close        REAL     NOT NULL,
    volume       INTEGER  NOT NULL,
    fetched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, timeframe, bar_date)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_query ON ohlcv(ticker, timeframe, bar_date DESC);

CREATE TABLE IF NOT EXISTS fetch_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    started_at   TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    rows_written INTEGER,
    status       TEXT,
    notes        TEXT
);
"""


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(PRICES_DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with _conn() as c:
        c.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------

def last_bar_date(ticker: str, timeframe: str) -> date | None:
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(bar_date) AS d FROM ohlcv "
            "WHERE ticker = ? AND timeframe = ?",
            (ticker, timeframe),
        ).fetchone()
    if not row or row["d"] is None:
        return None
    val = row["d"]
    if isinstance(val, date):
        return val
    return date.fromisoformat(str(val))


def load_ohlcv(ticker: str, timeframe: str,
               start: date | None = None,
               end: date | None = None) -> pd.DataFrame:
    """Load a single ticker's OHLCV as a DataFrame.

    Returns columns ['open','high','low','close','volume'] indexed by
    DatetimeIndex named 'bar_date' (ascending). Empty frame if nothing stored.
    """
    init_db()
    q = ("SELECT bar_date, open, high, low, close, volume "
         "FROM ohlcv WHERE ticker = ? AND timeframe = ?")
    params: list = [ticker, timeframe]
    if start is not None:
        q += " AND bar_date >= ?"
        params.append(start.isoformat())
    if end is not None:
        q += " AND bar_date <= ?"
        params.append(end.isoformat())
    q += " ORDER BY bar_date ASC"

    with _conn() as c:
        df = pd.read_sql_query(q, c, params=params, parse_dates=["bar_date"])
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return df.set_index("bar_date")


def load_ohlcv_multi(tickers: list[str], timeframe: str,
                     start: date | None = None,
                     end: date | None = None) -> pd.DataFrame:
    """Load many tickers as a single frame with MultiIndex columns.

    Columns: MultiIndex with level 0 = ticker, level 1 = field
    ('open','high','low','close','volume'). Index is DatetimeIndex named
    'bar_date'. Missing bars are NaN-aligned across tickers.
    """
    if not tickers:
        return pd.DataFrame()
    init_db()
    placeholders = ",".join("?" for _ in tickers)
    q = (f"SELECT ticker, bar_date, open, high, low, close, volume "
         f"FROM ohlcv WHERE timeframe = ? AND ticker IN ({placeholders})")
    params: list = [timeframe, *tickers]
    if start is not None:
        q += " AND bar_date >= ?"
        params.append(start.isoformat())
    if end is not None:
        q += " AND bar_date <= ?"
        params.append(end.isoformat())
    q += " ORDER BY bar_date ASC"

    with _conn() as c:
        df = pd.read_sql_query(q, c, params=params, parse_dates=["bar_date"])
    if df.empty:
        return pd.DataFrame()

    wide = df.pivot(index="bar_date", columns="ticker",
                    values=["open", "high", "low", "close", "volume"])
    # Re-order to (ticker, field) for nicer slicing: df['XLK']['close']
    wide = wide.swaplevel(axis=1).sort_index(axis=1)
    return wide


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------

def upsert_ohlcv(rows: list[dict]) -> int:
    """Insert-or-update OHLCV rows. Each row must have keys:
    ticker, timeframe, bar_date, open, high, low, close, volume.
    Returns count of rows written.
    """
    if not rows:
        return 0
    init_db()
    sql = (
        "INSERT INTO ohlcv "
        "(ticker, timeframe, bar_date, open, high, low, close, volume, fetched_at) "
        "VALUES (:ticker, :timeframe, :bar_date, :open, :high, :low, :close, "
        ":volume, CURRENT_TIMESTAMP) "
        "ON CONFLICT(ticker, timeframe, bar_date) DO UPDATE SET "
        "  open = excluded.open,"
        "  high = excluded.high,"
        "  low = excluded.low,"
        "  close = excluded.close,"
        "  volume = excluded.volume,"
        "  fetched_at = CURRENT_TIMESTAMP"
    )
    with _conn() as c:
        c.executemany(sql, rows)
    return len(rows)


def wipe_ticker(ticker: str, timeframe: str) -> int:
    """Delete every row for one ticker/timeframe. Used by the split-replace
    branch of `update_ticker`. Returns number of rows deleted.
    """
    init_db()
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM ohlcv WHERE ticker = ? AND timeframe = ?",
            (ticker, timeframe),
        )
        return cur.rowcount


def _log_fetch(ticker: str, timeframe: str, started_at: datetime,
               completed_at: datetime, rows_written: int,
               status: str, notes: str = "") -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO fetch_log "
            "(ticker, timeframe, started_at, completed_at, rows_written, status, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticker, timeframe, started_at, completed_at, rows_written, status, notes),
        )


# ---------------------------------------------------------------------------
# Update orchestration
# ---------------------------------------------------------------------------

def update_ticker(ticker: str, timeframe: str) -> dict:
    """Cold-start or incremental update for one (ticker, timeframe) pair.

    Cold start
    ----------
    If no rows exist yet, pull ~5 years of history and upsert.

    Incremental
    -----------
    Refetch from `last_date - OVERLAP_DAYS` (60 days). The overlap window
    serves two purposes:
      1. fills in any small gaps near the tail (late-arriving bars, etc.);
      2. detects retroactive history rewrites.

    Why the overlap matters (the non-obvious bit)
    ---------------------------------------------
    yfinance with `auto_adjust=True` returns SPLIT- AND DIVIDEND-ADJUSTED
    historical prices. When a ticker splits 2-for-1 today, every close
    yfinance returns for every prior date is divided by 2 retroactively.
    If we naively appended only the new bars, the recent history would be
    in split-adjusted units and the older history in pre-split units —
    a discontinuity that would poison every SMA, RSI, momentum, and
    relative-strength calculation downstream.

    To guard against this:
      * fetch a 60-day overlap window;
      * compare each overlapping bar's close to what we already have;
      * if ANY pair differs by more than `SPLIT_TOL` (0.5%), declare a
        split/dividend re-adjustment, WIPE the ticker's full history
        for this timeframe, and re-pull from scratch (~5 years);
      * log the event with `status='split_detected'`.

    A 60-day overlap costs roughly one yfinance request per ticker per run.
    A corrupted history costs hours of debugging. Do not "optimize" this away.

    Returns
    -------
    {"status": "ok" | "partial" | "split_detected" | "error",
     "rows_written": int,
     "notes": str}
    """
    from src.market_engine import fetch_ohlcv_yf  # local import to avoid cycles

    started_at = datetime.utcnow()
    last = last_bar_date(ticker, timeframe)
    today = date.today()

    # ---- cold start ----
    if last is None:
        start = today - timedelta(days=COLD_START_DAYS)
        try:
            df = fetch_ohlcv_yf([ticker], timeframe, start=start, end=today)
        except Exception as exc:
            _log_fetch(ticker, timeframe, started_at, datetime.utcnow(),
                       0, "error", f"cold-start fetch failed: {exc}")
            return {"status": "error", "rows_written": 0, "notes": str(exc)}
        if df.empty:
            _log_fetch(ticker, timeframe, started_at, datetime.utcnow(),
                       0, "error", "cold-start returned empty")
            return {"status": "error", "rows_written": 0,
                    "notes": "yfinance returned no rows"}
        rows = _df_to_rows(df, ticker, timeframe)
        n = upsert_ohlcv(rows)
        _log_fetch(ticker, timeframe, started_at, datetime.utcnow(),
                   n, "ok", f"cold start, {n} rows")
        return {"status": "ok", "rows_written": n,
                "notes": f"cold start: {n} rows"}

    # ---- incremental ----
    refetch_start = last - timedelta(days=OVERLAP_DAYS)
    try:
        df = fetch_ohlcv_yf([ticker], timeframe, start=refetch_start, end=today)
    except Exception as exc:
        _log_fetch(ticker, timeframe, started_at, datetime.utcnow(),
                   0, "error", f"incremental fetch failed: {exc}")
        return {"status": "error", "rows_written": 0, "notes": str(exc)}

    if df.empty:
        _log_fetch(ticker, timeframe, started_at, datetime.utcnow(),
                   0, "partial", "incremental returned empty")
        return {"status": "partial", "rows_written": 0,
                "notes": "yfinance returned no rows in incremental window"}

    # ---- split detection ----
    stored = load_ohlcv(ticker, timeframe, start=refetch_start, end=today)
    if not stored.empty:
        new_idx = pd.to_datetime(df["bar_date"]).dt.normalize()
        new_close = pd.Series(df["close"].values, index=new_idx)
        stored_close = stored["close"]
        stored_close.index = pd.to_datetime(stored_close.index).normalize()
        common = new_close.index.intersection(stored_close.index)
        if len(common):
            diff = (new_close.loc[common] - stored_close.loc[common]).abs() \
                   / stored_close.loc[common].replace(0, pd.NA)
            max_diff = float(diff.max(skipna=True)) if not diff.empty else 0.0
            if pd.notna(max_diff) and max_diff > SPLIT_TOL:
                # Wipe and re-pull from scratch.
                wipe_ticker(ticker, timeframe)
                start = today - timedelta(days=COLD_START_DAYS)
                try:
                    full = fetch_ohlcv_yf([ticker], timeframe,
                                          start=start, end=today)
                except Exception as exc:
                    _log_fetch(ticker, timeframe, started_at, datetime.utcnow(),
                               0, "error",
                               f"split-replace re-pull failed: {exc}")
                    return {"status": "error", "rows_written": 0,
                            "notes": f"split-replace fetch failed: {exc}"}
                rows = _df_to_rows(full, ticker, timeframe)
                n = upsert_ohlcv(rows)
                notes = (f"split_detected: max overlap diff={max_diff:.4f}, "
                         f"wiped and re-pulled {n} rows")
                _log_fetch(ticker, timeframe, started_at, datetime.utcnow(),
                           n, "split_detected", notes)
                return {"status": "split_detected", "rows_written": n,
                        "notes": notes}

    # Normal incremental upsert.
    rows = _df_to_rows(df, ticker, timeframe)
    n = upsert_ohlcv(rows)
    _log_fetch(ticker, timeframe, started_at, datetime.utcnow(),
               n, "ok", f"incremental, {n} rows in overlap+new window")
    return {"status": "ok", "rows_written": n,
            "notes": f"incremental: {n} rows touched"}


def update_all(tickers: list[str] | None = None,
               progress: Callable[[str, str, str], None] | None = None
               ) -> list[dict]:
    """Update both `1d` and `1wk` for every ticker.

    Parameters
    ----------
    tickers : list of symbols, or None to use SECTOR_ETFS + BENCHMARK.
    progress : optional callback(ticker, timeframe, status) for UI hooks.

    Returns one result dict per (ticker, timeframe) pair.
    """
    from config.settings import BENCHMARK, SECTOR_ETFS
    if tickers is None:
        tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]

    results: list[dict] = []
    for tkr in tickers:
        for tf in ("1d", "1wk"):
            res = update_ticker(tkr, tf)
            res["ticker"] = tkr
            res["timeframe"] = tf
            results.append(res)
            if progress is not None:
                progress(tkr, tf, res["status"])
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_rows(df: pd.DataFrame, ticker: str, timeframe: str) -> list[dict]:
    """Turn a fetch_ohlcv_yf result into the row-dict format `upsert_ohlcv`
    expects. Drops any row with a missing close.
    """
    if df.empty:
        return []
    out: list[dict] = []
    for _, r in df.iterrows():
        if pd.isna(r["close"]):
            continue
        bd = r["bar_date"]
        if isinstance(bd, pd.Timestamp):
            bd = bd.date()
        elif isinstance(bd, str):
            bd = date.fromisoformat(bd)
        out.append({
            "ticker": ticker,
            "timeframe": timeframe,
            "bar_date": bd,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": int(r["volume"]) if pd.notna(r["volume"]) else 0,
        })
    return out


# ---------------------------------------------------------------------------
# Live sanity check
# ---------------------------------------------------------------------------

def sanity_check_latest(ticker: str) -> dict:
    """Cross-check the most recent stored daily close against a live yfinance
    quote. Useful for verifying the DB isn't stale or split-corrupted.

    Returns
    -------
    {"status": "ok" | "stale" | "mismatch" | "error",
     "stored_close": float | None,
     "live_close":   float | None,
     "diff_pct":     float | None}
    """
    import yfinance as yf

    try:
        stored = load_ohlcv(ticker, "1d")
        if stored.empty:
            return {"status": "error", "stored_close": None,
                    "live_close": None, "diff_pct": None}
        stored_close = float(stored["close"].iloc[-1])

        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1d", auto_adjust=True)
        if hist.empty:
            return {"status": "error", "stored_close": stored_close,
                    "live_close": None, "diff_pct": None}
        live_close = float(hist["Close"].iloc[-1])
        diff_pct = abs(live_close - stored_close) / stored_close \
            if stored_close else None

        if diff_pct is None:
            status = "error"
        elif diff_pct > 0.02:
            status = "mismatch"
        else:
            status = "ok"
        return {"status": status, "stored_close": stored_close,
                "live_close": live_close, "diff_pct": diff_pct}
    except Exception as exc:
        return {"status": "error", "stored_close": None,
                "live_close": None, "diff_pct": None, "error": str(exc)}
