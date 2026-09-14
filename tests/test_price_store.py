"""Tests for src.price_store, focused on the split-detection guard in
update_ticker (see that function's docstring and _exclude_forming_bar).

Same DB-redirection pattern already used elsewhere in this suite (e.g.
tests/test_weekly_recap.py's temp_db fixture): price_store computes
PRICES_DB_PATH once at import time, so tests monkeypatch the module's own
attribute rather than config.settings, which wouldn't reach the already-bound
name. fetch_ohlcv_yf is monkeypatched at its source (src.market_engine),
matching the local-import-inside-the-function pattern update_ticker uses.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import src.market_engine as market_engine
import src.price_store as price_store
from src.price_store import _exclude_forming_bar, update_ticker, upsert_ohlcv


@pytest.fixture
def temp_prices_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_prices.db"
    monkeypatch.setattr(price_store, "PRICES_DB_PATH", db_file)
    price_store.init_db()
    yield db_file


def _seed_weekly_bars(ticker: str, closes: list[float], end: date) -> list[date]:
    """Upsert `len(closes)` weekly bars, 7 days apart, ending on `end`
    (the most recent bar). Returns the bar dates, oldest first."""
    n = len(closes)
    dates = [end - timedelta(days=7 * (n - 1 - i)) for i in range(n)]
    rows = [
        {"ticker": ticker, "timeframe": "1wk", "bar_date": d,
         "open": c, "high": c, "low": c, "close": c, "volume": 1_000_000}
        for d, c in zip(dates, closes)
    ]
    upsert_ohlcv(rows)
    return dates


def _fake_fetch(dates: list[date], closes: list[float]) -> pd.DataFrame:
    """Build the shape fetch_ohlcv_yf returns: ticker, bar_date, OHLCV."""
    return pd.DataFrame({
        "ticker": ["TST"] * len(dates),
        "bar_date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1_000_000] * len(dates),
    })


# ---------------------------------------------------------------------------
# _exclude_forming_bar — pure function, no DB/network involved
# ---------------------------------------------------------------------------

def test_exclude_forming_bar_empty_index_stays_empty():
    empty = pd.DatetimeIndex([])
    out = _exclude_forming_bar(empty)
    assert len(out) == 0


def test_exclude_forming_bar_single_element_excludes_it():
    idx = pd.DatetimeIndex(["2026-08-14"])
    out = _exclude_forming_bar(idx)
    assert len(out) == 0


def test_exclude_forming_bar_drops_only_the_max():
    idx = pd.DatetimeIndex(["2026-08-07", "2026-08-14", "2026-08-21"])
    out = _exclude_forming_bar(idx)
    assert list(out) == [pd.Timestamp("2026-08-07"), pd.Timestamp("2026-08-14")]


# ---------------------------------------------------------------------------
# update_ticker — the actual bug: an in-progress last bar must not trip
# split_detected, but a genuine split (moving every bar) still must.
# ---------------------------------------------------------------------------

def test_in_progress_last_bar_does_not_trigger_false_split(temp_prices_db, monkeypatch):
    """The exact failure mode found in the live fetch_log: six settled
    weekly bars are unchanged, but the newest (still-forming) bar comes
    back from yfinance revised by 2% — comfortably above SPLIT_TOL (0.5%).
    Before the fix this wiped and re-pulled the ticker's full history;
    after the fix it's just a normal incremental update."""
    today = date.today()
    stored_closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    dates = _seed_weekly_bars("TST", stored_closes, end=today)

    # Same five older bars, unchanged; only the newest (today's) bar is
    # revised, mimicking more of the week's trading having landed since
    # it was first stored.
    revised_closes = [100.0, 100.0, 100.0, 100.0, 100.0, 102.0]
    fake_df = _fake_fetch(dates, revised_closes)
    monkeypatch.setattr(market_engine, "fetch_ohlcv_yf",
                        lambda tickers, timeframe, start, end: fake_df)

    result = update_ticker("TST", "1wk")

    assert result["status"] == "ok"
    # The revised close for the forming bar should still be captured via
    # the normal incremental upsert -- the fix skips the false split
    # trigger, it doesn't discard the fresh data.
    updated = price_store.load_ohlcv("TST", "1wk")
    assert updated["close"].iloc[-1] == pytest.approx(102.0)


def test_genuine_split_across_settled_bars_is_still_caught(temp_prices_db, monkeypatch):
    """A real split/dividend re-adjustment moves EVERY historical close, not
    just the newest one -- so it still shows up in the settled bars the fix
    continues to compare, and must still be flagged."""
    today = date.today()
    stored_closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    dates = _seed_weekly_bars("TST", stored_closes, end=today)

    # A clean 2-for-1 split: every bar, including the newest, is halved.
    split_closes = [c / 2 for c in stored_closes]
    fake_df = _fake_fetch(dates, split_closes)

    def _fake_fetch_ohlcv_yf(tickers, timeframe, start, end):
        # First call is the incremental overlap fetch; the split branch
        # then re-pulls "from scratch" -- return the same split-adjusted
        # series either way, exactly like a real yfinance response would.
        return fake_df

    monkeypatch.setattr(market_engine, "fetch_ohlcv_yf", _fake_fetch_ohlcv_yf)

    result = update_ticker("TST", "1wk")

    assert result["status"] == "split_detected"
    assert "split_detected" in result["notes"]


def test_only_the_most_recent_bar_differing_by_a_lot_is_not_enough(temp_prices_db, monkeypatch):
    """Even a large (not just borderline) revision on ONLY the newest bar
    must not trip the guard -- confirms the exclusion isn't accidentally
    tolerance-dependent."""
    today = date.today()
    stored_closes = [50.0, 50.0, 50.0, 50.0]
    dates = _seed_weekly_bars("TST", stored_closes, end=today)

    revised_closes = [50.0, 50.0, 50.0, 65.0]  # +30% on the newest bar only
    fake_df = _fake_fetch(dates, revised_closes)
    monkeypatch.setattr(market_engine, "fetch_ohlcv_yf",
                        lambda tickers, timeframe, start, end: fake_df)

    result = update_ticker("TST", "1wk")
    assert result["status"] == "ok"
