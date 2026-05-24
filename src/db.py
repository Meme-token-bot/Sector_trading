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

CREATE TABLE IF NOT EXISTS weekly_recaps (
    as_of_iso       TEXT NOT NULL,
    model           TEXT NOT NULL,
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    n_newsletters   INTEGER NOT NULL,
    payload_json    TEXT NOT NULL,
    PRIMARY KEY (as_of_iso, model)
);

CREATE INDEX IF NOT EXISTS idx_wr_as_of ON weekly_recaps(as_of_iso DESC);
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


def recent_newsletters(limit: int = 25) -> pd.DataFrame:
    init_db()
    with _conn() as c:
        return pd.read_sql_query(
            """SELECT id, author, publication_date, overall_macro_bias,
                      substr(summary, 1, 200) AS summary, ingested_at
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
