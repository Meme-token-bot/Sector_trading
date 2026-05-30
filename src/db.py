"""SQLite persistence for parsed newsletter sentiment."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Iterator

import pandas as pd

from config.settings import DB_PATH, PARAMS
from src.schemas import NewsletterAnalysis


SCHEMA = """
CREATE TABLE IF NOT EXISTS newsletters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash    TEXT NOT NULL UNIQUE,
    author          TEXT NOT NULL,
    publication_date DATE NOT NULL,
    overall_macro_bias TEXT NOT NULL,
    summary         TEXT,
    raw_text        TEXT,
    gmail_message_id TEXT,
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sector_ratings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    newsletter_id   INTEGER NOT NULL,
    ticker          TEXT NOT NULL,
    sentiment_score INTEGER NOT NULL,
    reasoning       TEXT,
    FOREIGN KEY (newsletter_id) REFERENCES newsletters(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sr_ticker ON sector_ratings(ticker);
CREATE INDEX IF NOT EXISTS idx_nl_date   ON newsletters(publication_date);

CREATE TABLE IF NOT EXISTS theme_ratings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    newsletter_id   INTEGER NOT NULL,
    theme_key       TEXT NOT NULL,
    sentiment_score INTEGER NOT NULL,
    reasoning       TEXT,
    FOREIGN KEY (newsletter_id) REFERENCES newsletters(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tr_theme ON theme_ratings(theme_key);

CREATE TABLE IF NOT EXISTS theme_news (
    theme_key    TEXT NOT NULL,
    as_of        DATE NOT NULL,
    score        REAL NOT NULL,
    n_headlines  INTEGER NOT NULL,
    top_headline TEXT,
    fetched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (theme_key, as_of)
);

CREATE TABLE IF NOT EXISTS weekly_recaps (
    as_of_iso       TEXT NOT NULL,
    model           TEXT NOT NULL,
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    n_newsletters   INTEGER NOT NULL,
    payload_json    TEXT NOT NULL,
    PRIMARY KEY (as_of_iso, model)
);

CREATE INDEX IF NOT EXISTS idx_wr_as_of ON weekly_recaps(as_of_iso DESC);

-- Persisted refined-state snapshots. One row per (as_of, ticker). Written
-- by the dashboard + the weekly script (`scripts/run_signals.py`) so we
-- have a tamper-resistant record of WHAT THE MODEL EMITTED, separate from
-- what gets re-computed at view time. Forward-perf tracking reads from
-- here, not from a re-replay of build_signal_history (which could quietly
-- shift if PARAMS, the sentiment window, or upstream code changes).
CREATE TABLE IF NOT EXISTS signal_snapshots (
    as_of                 DATE    NOT NULL,
    ticker                TEXT    NOT NULL,
    state                 TEXT    NOT NULL,   -- refined: NEW_BUY / HOLD_IF_LONG / CHASE / REDUCE / HOLD / SELL / WATCH
    signal                TEXT    NOT NULL,   -- raw: BUY / HOLD / SELL
    above_sma             INTEGER NOT NULL,   -- 0/1
    extension_pct         REAL,
    relative_strength_3m  REAL,
    rs_rank               INTEGER,
    sentiment_score       REAL,
    n_sentiment_obs       INTEGER,
    macro_tailwinds       INTEGER,
    macro_headwinds       INTEGER,
    conviction            INTEGER,
    consecutive_buy_weeks INTEGER,
    written_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of, ticker)
);
CREATE INDEX IF NOT EXISTS idx_snap_ticker ON signal_snapshots(ticker, as_of);
CREATE INDEX IF NOT EXISTS idx_snap_state  ON signal_snapshots(state, as_of);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the original schema. Idempotent."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(newsletters)")}
    if "gmail_message_id" not in cols:
        conn.execute("ALTER TABLE newsletters ADD COLUMN gmail_message_id TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nl_gmail "
                 "ON newsletters(gmail_message_id)")


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)
        _migrate(c)


def gmail_message_already_ingested(gmail_message_id: str) -> bool:
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM newsletters WHERE gmail_message_id = ?",
            (gmail_message_id,),
        ).fetchone()
    return row is not None


def attach_gmail_message_id(newsletter_id: int, gmail_message_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE newsletters SET gmail_message_id = ? WHERE id = ?",
            (gmail_message_id, newsletter_id),
        )


def _hash_content(text: str, author: str, pub_date: date) -> str:
    h = hashlib.sha256()
    h.update(f"{author}|{pub_date.isoformat()}|".encode())
    h.update(text.strip().encode())
    return h.hexdigest()


def save_analysis(analysis: NewsletterAnalysis, raw_text: str) -> int | None:
    init_db()
    content_hash = _hash_content(raw_text, analysis.author, analysis.publication_date)
    with _conn() as c:
        existing = c.execute(
            "SELECT id FROM newsletters WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if existing:
            return None

        cur = c.execute(
            """INSERT INTO newsletters
               (content_hash, author, publication_date, overall_macro_bias, summary, raw_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (content_hash, analysis.author, analysis.publication_date,
             analysis.overall_macro_bias.value, analysis.summary, raw_text),
        )
        nid = cur.lastrowid
        c.executemany(
            """INSERT INTO sector_ratings
               (newsletter_id, ticker, sentiment_score, reasoning)
               VALUES (?, ?, ?, ?)""",
            [(nid, r.ticker, r.sentiment_score, r.reasoning)
             for r in analysis.sector_ratings],
        )
        c.executemany(
            """INSERT INTO theme_ratings
               (newsletter_id, theme_key, sentiment_score, reasoning)
               VALUES (?, ?, ?, ?)""",
            [(nid, r.theme_key, r.sentiment_score, r.reasoning)
             for r in getattr(analysis, "theme_ratings", [])],
        )
        return nid


def aggregate_sentiment(as_of: date | None = None,
                        lookback_days: int | None = None) -> pd.DataFrame:
    """Per-ticker rolling sentiment aggregation.

    Returns DataFrame indexed by ticker with columns (in this order):
      score:       float — mean of per-newsletter sentiment scores
      n_obs:       int   — count of contributing observations
      score_stdev: float — population stdev (ddof=0); 0.0 when n_obs < 2
      score_min:   float — min observed score; NaN when n_obs == 0
      score_max:   float — max observed score; NaN when n_obs == 0
    """
    as_of = as_of or date.today()
    lookback = lookback_days or PARAMS.sentiment_lookback_days
    cutoff = as_of - timedelta(days=lookback)

    empty_cols = ["score", "n_obs", "score_stdev", "score_min", "score_max"]
    init_db()
    with _conn() as c:
        df = pd.read_sql_query(
            """
            SELECT sr.ticker, sr.sentiment_score
            FROM sector_ratings sr
            JOIN newsletters n ON n.id = sr.newsletter_id
            WHERE n.publication_date >= ?
              AND n.publication_date <= ?
            """,
            c, params=(cutoff.isoformat(), as_of.isoformat()),
        )
    if df.empty:
        return pd.DataFrame(columns=empty_cols).rename_axis("ticker")

    grouped = df.groupby("ticker")["sentiment_score"]
    # Population stdev (ddof=0). pandas returns NaN for groups of size 1,
    # which we coerce to 0.0 per spec.
    stdev = grouped.std(ddof=0).fillna(0.0)
    out = pd.DataFrame({
        "score":       grouped.mean().astype(float),
        "n_obs":       grouped.size().astype(int),
        "score_stdev": stdev.astype(float),
        "score_min":   grouped.min().astype(float),
        "score_max":   grouped.max().astype(float),
    })
    out.index.name = "ticker"
    # Enforce canonical column order.
    return out[empty_cols]


def aggregate_theme_sentiment(as_of: date | None = None,
                              lookback_days: int | None = None) -> pd.DataFrame:
    """Per-theme rolling newsletter sentiment, mirroring aggregate_sentiment.

    Returns DataFrame indexed by theme_key with columns: score (mean), n_obs.
    Empty frame (those columns) when no theme ratings fall in the window.
    """
    as_of = as_of or date.today()
    lookback = lookback_days or PARAMS.sentiment_lookback_days
    cutoff = as_of - timedelta(days=lookback)

    init_db()
    with _conn() as c:
        df = pd.read_sql_query(
            """
            SELECT tr.theme_key, tr.sentiment_score
            FROM theme_ratings tr
            JOIN newsletters n ON n.id = tr.newsletter_id
            WHERE n.publication_date >= ? AND n.publication_date <= ?
            """,
            c, params=(cutoff.isoformat(), as_of.isoformat()),
        )
    if df.empty:
        return pd.DataFrame(columns=["score", "n_obs"]).rename_axis("theme_key")

    grouped = df.groupby("theme_key")["sentiment_score"]
    out = pd.DataFrame({
        "score": grouped.mean().astype(float),
        "n_obs": grouped.size().astype(int),
    })
    out.index.name = "theme_key"
    return out


def save_theme_news(as_of: date, rows: list[dict]) -> None:
    """Upsert per-theme news scores for `as_of`.

    Each row: {theme_key, score, n_headlines, top_headline}. Idempotent — a
    re-run for the same day overwrites that day's scores.
    """
    if not rows:
        return
    init_db()
    with _conn() as c:
        c.executemany(
            """INSERT INTO theme_news
                   (theme_key, as_of, score, n_headlines, top_headline)
               VALUES (:theme_key, :as_of, :score, :n_headlines, :top_headline)
               ON CONFLICT(theme_key, as_of) DO UPDATE SET
                   score=excluded.score,
                   n_headlines=excluded.n_headlines,
                   top_headline=excluded.top_headline,
                   fetched_at=CURRENT_TIMESTAMP""",
            [{"theme_key": r["theme_key"], "as_of": as_of.isoformat(),
              "score": float(r["score"]), "n_headlines": int(r["n_headlines"]),
              "top_headline": r.get("top_headline", "")} for r in rows],
        )


def latest_theme_news(max_age_days: int = 14) -> pd.DataFrame:
    """Most-recent news score per theme within `max_age_days`.

    Returns DataFrame indexed by theme_key with columns: score, n_headlines,
    top_headline, as_of. Empty when nothing fresh enough.
    """
    cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
    init_db()
    with _conn() as c:
        df = pd.read_sql_query(
            """
            SELECT tn.theme_key, tn.score, tn.n_headlines, tn.top_headline,
                   tn.as_of
            FROM theme_news tn
            JOIN (SELECT theme_key, MAX(as_of) AS mx
                  FROM theme_news WHERE as_of >= ? GROUP BY theme_key) latest
              ON latest.theme_key = tn.theme_key AND latest.mx = tn.as_of
            """,
            c, params=(cutoff,),
        )
    if df.empty:
        return pd.DataFrame(
            columns=["score", "n_headlines", "top_headline", "as_of"]
        ).rename_axis("theme_key")
    return df.set_index("theme_key")


def recent_newsletters(limit: int = 25) -> pd.DataFrame:
    init_db()
    with _conn() as c:
        return pd.read_sql_query(
            """SELECT id, author, publication_date, overall_macro_bias,
                      summary, ingested_at
               FROM newsletters
               ORDER BY publication_date DESC, ingested_at DESC
               LIMIT ?""",
            c, params=(limit,),
        )


def delete_newsletter(newsletter_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM newsletters WHERE id = ?", (newsletter_id,))


# ---------------------------------------------------------------------------
# Weekly recap persistence — one row per (as_of_iso, model). Lets the user
# regenerate the dashboard tab without re-spending OpenAI tokens and keeps a
# browsable history of past recaps.
# ---------------------------------------------------------------------------

def save_weekly_recap(as_of_iso: str, model: str,
                      payload: dict, n_newsletters: int) -> None:
    """Upsert one recap. `payload` is the JSON-mode model_dump() of WeeklyRecap."""
    init_db()
    with _conn() as c:
        c.execute(
            """INSERT INTO weekly_recaps
                   (as_of_iso, model, n_newsletters, payload_json,
                    generated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(as_of_iso, model) DO UPDATE SET
                   n_newsletters = excluded.n_newsletters,
                   payload_json  = excluded.payload_json,
                   generated_at  = CURRENT_TIMESTAMP""",
            (as_of_iso, model, n_newsletters, json.dumps(payload)),
        )


def load_weekly_recap(as_of_iso: str, model: str) -> dict | None:
    """Return the stored payload dict, or None if no row exists."""
    init_db()
    with _conn() as c:
        row = c.execute(
            """SELECT payload_json FROM weekly_recaps
               WHERE as_of_iso = ? AND model = ?""",
            (as_of_iso, model),
        ).fetchone()
    return json.loads(row["payload_json"]) if row else None


def list_weekly_recaps(limit: int = 25) -> pd.DataFrame:
    """Browsable history: most recent first, all models."""
    init_db()
    with _conn() as c:
        return pd.read_sql_query(
            """SELECT as_of_iso, model, generated_at, n_newsletters
               FROM weekly_recaps
               ORDER BY as_of_iso DESC, generated_at DESC
               LIMIT ?""",
            c, params=(limit,),
        )


def delete_weekly_recap(as_of_iso: str, model: str) -> None:
    with _conn() as c:
        c.execute(
            "DELETE FROM weekly_recaps WHERE as_of_iso = ? AND model = ?",
            (as_of_iso, model),
        )


# ---------------------------------------------------------------------------
# Signal snapshots
# ---------------------------------------------------------------------------

def save_signal_snapshot(as_of: date, refined: pd.DataFrame,
                         macro_alignment: pd.DataFrame | None = None) -> int:
    """Persist one weekly snapshot of the refined-signals frame.

    Idempotent on (as_of, ticker) — re-running the same as_of overwrites.
    `refined` is the frame returned by `src.signals.refine_signals` (must
    carry the `state` and `signal` columns at minimum; the rest are stored
    as NULL when absent so older callers don't break).

    Returns count of rows written.
    """
    if refined is None or refined.empty:
        return 0
    init_db()
    rows: list[dict] = []
    for tkr, row in refined.iterrows():
        macro_tw = macro_hw = None
        if macro_alignment is not None and not macro_alignment.empty and tkr in macro_alignment.index:
            macro_tw = int(macro_alignment.loc[tkr, "tailwinds"] or 0)
            macro_hw = int(macro_alignment.loc[tkr, "headwinds"] or 0)
        rows.append({
            "as_of": as_of.isoformat(),
            "ticker": str(tkr),
            "state": str(row.get("state", row.get("signal", ""))),
            "signal": str(row.get("signal", "")),
            "above_sma": int(bool(row.get("above_sma", False))),
            "extension_pct": _safe_float(row.get("extension_pct")),
            "relative_strength_3m": _safe_float(row.get("relative_strength_3m")),
            "rs_rank": _safe_int(row.get("rs_rank")),
            "sentiment_score": _safe_float(row.get("sentiment_score")),
            "n_sentiment_obs": _safe_int(row.get("n_obs")),
            "macro_tailwinds": macro_tw,
            "macro_headwinds": macro_hw,
            "conviction": _safe_int(row.get("conviction")),
            "consecutive_buy_weeks": _safe_int(row.get("consecutive_buy_weeks")),
        })
    with _conn() as c:
        c.executemany(
            """
            INSERT INTO signal_snapshots
              (as_of, ticker, state, signal, above_sma, extension_pct,
               relative_strength_3m, rs_rank, sentiment_score, n_sentiment_obs,
               macro_tailwinds, macro_headwinds, conviction,
               consecutive_buy_weeks, written_at)
            VALUES
              (:as_of, :ticker, :state, :signal, :above_sma, :extension_pct,
               :relative_strength_3m, :rs_rank, :sentiment_score, :n_sentiment_obs,
               :macro_tailwinds, :macro_headwinds, :conviction,
               :consecutive_buy_weeks, CURRENT_TIMESTAMP)
            ON CONFLICT(as_of, ticker) DO UPDATE SET
              state = excluded.state,
              signal = excluded.signal,
              above_sma = excluded.above_sma,
              extension_pct = excluded.extension_pct,
              relative_strength_3m = excluded.relative_strength_3m,
              rs_rank = excluded.rs_rank,
              sentiment_score = excluded.sentiment_score,
              n_sentiment_obs = excluded.n_sentiment_obs,
              macro_tailwinds = excluded.macro_tailwinds,
              macro_headwinds = excluded.macro_headwinds,
              conviction = excluded.conviction,
              consecutive_buy_weeks = excluded.consecutive_buy_weeks,
              written_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
    return len(rows)


def load_signal_snapshots(
    state: str | list[str] | None = None,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    """Load persisted signal snapshots, optionally filtered by state / window.

    Returns DataFrame with one row per (as_of, ticker). Empty frame if nothing
    matches. Caller should treat as_of as a date.
    """
    init_db()
    q = "SELECT * FROM signal_snapshots WHERE 1=1"
    params: list = []
    if state is not None:
        states = [state] if isinstance(state, str) else list(state)
        placeholders = ",".join("?" for _ in states)
        q += f" AND state IN ({placeholders})"
        params.extend(states)
    if since is not None:
        q += " AND as_of >= ?"
        params.append(since.isoformat())
    if until is not None:
        q += " AND as_of <= ?"
        params.append(until.isoformat())
    q += " ORDER BY as_of ASC, ticker ASC"
    with _conn() as c:
        df = pd.read_sql_query(q, c, params=params, parse_dates=["as_of"])
    return df


def _safe_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _safe_int(v) -> int | None:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(i):
        return None
    return i
