"""SQLite persistence for parsed newsletter sentiment."""
from __future__ import annotations

import hashlib
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
    as_of = as_of or date.today()
    lookback = lookback_days or PARAMS.sentiment_lookback_days
    cutoff = as_of - timedelta(days=lookback)

    init_db()
    with _conn() as c:
        df = pd.read_sql_query(
            """
            SELECT sr.ticker,
                   AVG(sr.sentiment_score) AS score,
                   COUNT(*)                AS n_obs
            FROM sector_ratings sr
            JOIN newsletters n ON n.id = sr.newsletter_id
            WHERE n.publication_date >= ?
              AND n.publication_date <= ?
            GROUP BY sr.ticker
            """,
            c, params=(cutoff.isoformat(), as_of.isoformat()),
        )
    return df.set_index("ticker") if not df.empty else pd.DataFrame(
        columns=["score", "n_obs"]).rename_axis("ticker")


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
