"""Tests for src.preflight_checks. Uses temp SQLite DBs (same pattern as
tests/test_weekly_recap_persistence.py) plus targeted mocking of the
downstream pipeline calls, since these checks legitimately touch DB/Tiger/
price-panel boundaries — that's the whole point of a readiness check.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

import src.preflight_checks as pc


# ---------------------------------------------------------------------------
# check_data_freshness
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dbs(tmp_path, monkeypatch):
    """Redirect both DB_PATH (sentiment.db) and PRICES_DB_PATH used inside
    src.preflight_checks to temp files, and create their minimal schemas."""
    sentiment_path = tmp_path / "sentiment.db"
    prices_path = tmp_path / "prices.db"

    with sqlite3.connect(sentiment_path) as c:
        c.execute("""CREATE TABLE newsletters (
            id INTEGER PRIMARY KEY, publication_date DATE)""")
        c.execute("""CREATE TABLE signal_snapshots (
            as_of DATE, ticker TEXT)""")
    with sqlite3.connect(prices_path) as c:
        c.execute("""CREATE TABLE ohlcv (
            ticker TEXT, timeframe TEXT, bar_date DATE)""")

    monkeypatch.setattr(pc, "DB_PATH", sentiment_path)
    monkeypatch.setattr(pc, "PRICES_DB_PATH", prices_path)
    return sentiment_path, prices_path


def _seed_prices(prices_path, universe: list[str], bar_date: date):
    with sqlite3.connect(prices_path) as c:
        c.executemany(
            "INSERT INTO ohlcv (ticker, timeframe, bar_date) VALUES (?, '1d', ?)",
            [(t, bar_date.isoformat()) for t in universe],
        )


def _seed_newsletters(sentiment_path, pub_date: date, n: int = 1):
    with sqlite3.connect(sentiment_path) as c:
        c.executemany(
            "INSERT INTO newsletters (publication_date) VALUES (?)",
            [(pub_date.isoformat(),)] * n,
        )


def test_freshness_fails_when_a_ticker_has_no_price_rows(temp_dbs):
    sentiment_path, prices_path = temp_dbs
    from config.settings import SECTOR_ETFS, SUPPLEMENTARY_SECTORS
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS][:-1]  # miss one
    _seed_prices(prices_path, universe, date.today())
    rows = pc.check_data_freshness()
    price_row = next(r for r in rows if r["label"] == "prices.db universe coverage")
    assert price_row["status"] == pc.FAIL
    assert "missing" in price_row["detail"]


def test_freshness_ok_when_bars_are_recent(temp_dbs):
    sentiment_path, prices_path = temp_dbs
    from config.settings import SECTOR_ETFS, SUPPLEMENTARY_SECTORS, BENCHMARK
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS] + [BENCHMARK]
    _seed_prices(prices_path, universe, date.today())
    _seed_newsletters(sentiment_path, date.today())
    rows = pc.check_data_freshness()
    price_row = next(r for r in rows if r["label"] == "prices.db freshness")
    assert price_row["status"] == pc.OK
    news_row = next(r for r in rows if r["label"] == "sentiment.db newsletters")
    assert news_row["status"] == pc.OK


def test_freshness_warns_when_data_is_a_week_stale(temp_dbs):
    sentiment_path, prices_path = temp_dbs
    from config.settings import SECTOR_ETFS, SUPPLEMENTARY_SECTORS, BENCHMARK
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS] + [BENCHMARK]
    stale_date = date.today() - timedelta(days=5)
    _seed_prices(prices_path, universe, stale_date)
    rows = pc.check_data_freshness()
    price_row = next(r for r in rows if r["label"] == "prices.db freshness")
    assert price_row["status"] == pc.WARN


def test_freshness_handles_empty_sentiment_db_gracefully(temp_dbs):
    sentiment_path, prices_path = temp_dbs
    from config.settings import SECTOR_ETFS, SUPPLEMENTARY_SECTORS, BENCHMARK
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS] + [BENCHMARK]
    _seed_prices(prices_path, universe, date.today())
    rows = pc.check_data_freshness()
    news_row = next(r for r in rows if r["label"] == "sentiment.db newsletters")
    assert news_row["status"] == pc.WARN
    assert news_row["detail"] == "(empty)"
    snap_row = next(r for r in rows if r["label"] == "signal_snapshots")
    assert snap_row["status"] == pc.WARN


# ---------------------------------------------------------------------------
# check_model_state
# ---------------------------------------------------------------------------

def test_model_state_fails_gracefully_on_pipeline_error():
    """load_price_panel isn't wired to real data in this test environment —
    the check must degrade to a single FAIL row, not raise."""
    rows, state = pc.check_model_state()
    assert any(r["status"] == pc.FAIL for r in rows)
    assert state["current_regime"] == "—"
    assert state["signals"].empty
    assert state["targets"].empty


def test_model_state_success_path_reports_regime_and_states():
    idx = pd.bdate_range("2024-01-01", periods=300)
    closes = pd.DataFrame({"SPY": 100.0, "XLK": 105.0}, index=idx)
    fake_signals = pd.DataFrame(
        {"state": ["NEW_BUY", "HOLD"]}, index=["XLK", "XLF"])
    fake_targets = pd.Series({"XLK": 0.5})

    with patch("src.backtest.load_price_panel", return_value=(closes, closes)), \
         patch("src.db.aggregate_sentiment", return_value=pd.DataFrame()), \
         patch("src.market_engine.compute_sector_metrics", return_value=pd.DataFrame()), \
         patch("src.regime_analysis.classify_regimes",
               return_value=pd.Series(["BULL"] * len(idx), index=idx)), \
         patch("src.signal_history.build_signal_history", return_value=pd.DataFrame()), \
         patch("src.signals.build_signals", return_value=pd.DataFrame()), \
         patch("src.signals.refine_signals", return_value=fake_signals), \
         patch("src.signals.target_weights", return_value=fake_targets):
        rows, state = pc.check_model_state()

    regime_row = next(r for r in rows if r["label"] == "current regime")
    assert regime_row["status"] == pc.OK
    assert "BULL" in regime_row["detail"]
    assert state["current_regime"] == "BULL"
    assert state["targets"].equals(fake_targets)


# ---------------------------------------------------------------------------
# check_tiger
# ---------------------------------------------------------------------------

def test_tiger_not_configured_warns_and_stops():
    with patch.object(pc, "tiger_configured", return_value=False):
        rows = pc.check_tiger({})
    assert len(rows) == 1
    assert rows[0]["status"] == pc.WARN
    assert "not configured" in rows[0]["detail"] or ".env" in rows[0]["detail"]


def test_tiger_connection_failure_is_reported_as_fail():
    with patch.object(pc, "tiger_configured", return_value=True), \
         patch("src.tiger_client.fetch_account_snapshot",
               side_effect=RuntimeError("boom")):
        rows = pc.check_tiger({})
    assert any(r["status"] == pc.FAIL for r in rows)


def test_tiger_cash_coverage_ok_when_sells_fund_the_buys():
    from src.tiger_client import AccountSnapshot
    snap = AccountSnapshot(net_liquidation=100_000.0, cash=5_000.0,
                           positions=pd.DataFrame())
    drift = pd.DataFrame({"trade_value": [10_000.0, -8_000.0]},
                         index=["XLK", "XLF"])
    targets = pd.Series({"XLK": 0.5})

    with patch.object(pc, "tiger_configured", return_value=True), \
         patch("src.tiger_client.fetch_account_snapshot", return_value=snap), \
         patch("src.tiger_client.compute_drift_by_sector", return_value=drift):
        rows = pc.check_tiger({"targets": targets})

    cash_row = next(r for r in rows if r["label"] == "Cash coverage for rotation")
    # need $10,000, free from sells $8,000, cash on hand $5,000 -> 13,000 >= 9,500 -> OK
    assert cash_row["status"] == pc.OK


def test_tiger_cash_coverage_warns_when_short():
    from src.tiger_client import AccountSnapshot
    snap = AccountSnapshot(net_liquidation=100_000.0, cash=500.0,
                           positions=pd.DataFrame())
    drift = pd.DataFrame({"trade_value": [50_000.0]}, index=["XLK"])
    targets = pd.Series({"XLK": 0.5})

    with patch.object(pc, "tiger_configured", return_value=True), \
         patch("src.tiger_client.fetch_account_snapshot", return_value=snap), \
         patch("src.tiger_client.compute_drift_by_sector", return_value=drift):
        rows = pc.check_tiger({"targets": targets})

    cash_row = next(r for r in rows if r["label"] == "Cash coverage for rotation")
    assert cash_row["status"] == pc.WARN


def test_tiger_no_targets_warns_and_stops_before_drift():
    with patch.object(pc, "tiger_configured", return_value=True), \
         patch("src.tiger_client.fetch_account_snapshot") as m:
        from src.tiger_client import AccountSnapshot
        m.return_value = AccountSnapshot(100_000.0, 5_000.0, pd.DataFrame())
        rows = pc.check_tiger({"targets": pd.Series(dtype=float)})
    assert any("no targets" in r["detail"] for r in rows)


def test_tiger_accepts_a_prefetched_snapshot_without_hitting_the_api():
    """The Dashboard passes its own already-cached snapshot in — this must
    not call fetch_account_snapshot() at all in that case."""
    from src.tiger_client import AccountSnapshot
    prefetched = AccountSnapshot(net_liquidation=50_000.0, cash=2_000.0,
                                 positions=pd.DataFrame())
    drift = pd.DataFrame({"trade_value": [1000.0]}, index=["XLK"])
    targets = pd.Series({"XLK": 0.5})

    with patch.object(pc, "tiger_configured", return_value=True), \
         patch("src.tiger_client.fetch_account_snapshot") as mock_fetch, \
         patch("src.tiger_client.compute_drift_by_sector", return_value=drift):
        rows = pc.check_tiger({"targets": targets}, snapshot=prefetched)

    mock_fetch.assert_not_called()
    nlv_row = next(r for r in rows if r["label"] == "Tiger connection")
    assert "50,000" in nlv_row["detail"]


# ---------------------------------------------------------------------------
# list_monday_orders
# ---------------------------------------------------------------------------

def test_monday_orders_classifies_states_correctly():
    signals = pd.DataFrame({
        "state": ["NEW_BUY", "SELL", "REDUCE", "HOLD", "CHASE"],
    }, index=["XLK", "XLF", "XLE", "XLU", "XLB"])
    targets = pd.Series({"XLK": 0.3, "XLB": 0.1})

    summary, orders = pc.list_monday_orders({"signals": signals, "targets": targets})
    actions = {o["ticker"]: o["action"] for o in orders}
    assert actions["XLK"] == "BUY"
    assert actions["XLF"] == "SELL"
    assert actions["XLE"] == "REDUCE"
    assert "XLU" not in actions  # HOLD generates no order
    # CHASE only orders if PARAMS.chase_weight_fraction > 0 (it's 0.25 by default)
    from config.settings import PARAMS
    if PARAMS.chase_weight_fraction > 0:
        assert actions["XLB"] == "CHASE"
    assert summary[0]["detail"].endswith("action(s)")


def test_monday_orders_empty_when_no_actionable_states():
    signals = pd.DataFrame({"state": ["HOLD", "HOLD_IF_LONG"]}, index=["XLK", "XLF"])
    summary, orders = pc.list_monday_orders({"signals": signals, "targets": pd.Series(dtype=float)})
    assert orders == []
    assert summary[0]["status"] == pc.OK
    assert "aligned" in summary[0]["detail"]


def test_monday_orders_no_model_output():
    summary, orders = pc.list_monday_orders({"signals": None})
    assert summary[0]["status"] == pc.WARN
    assert orders == []


# ---------------------------------------------------------------------------
# overall_verdict
# ---------------------------------------------------------------------------

def test_overall_verdict_ready():
    rows = [pc._row(pc.OK, "a"), pc._row(pc.OK, "b")]
    v = pc.overall_verdict(rows)
    assert v["verdict"] == "ready"
    assert v["n_ok"] == 2 and v["n_warn"] == 0 and v["n_fail"] == 0


def test_overall_verdict_ready_with_warnings():
    rows = [pc._row(pc.OK, "a"), pc._row(pc.WARN, "b")]
    v = pc.overall_verdict(rows)
    assert v["verdict"] == "ready_with_warnings"


def test_overall_verdict_not_ready_on_any_fail():
    rows = [pc._row(pc.OK, "a"), pc._row(pc.WARN, "b"), pc._row(pc.FAIL, "c")]
    v = pc.overall_verdict(rows)
    assert v["verdict"] == "not_ready"
