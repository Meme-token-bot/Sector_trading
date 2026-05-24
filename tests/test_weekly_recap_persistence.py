"""Tests for the weekly_recaps persistence layer in src.db."""
from __future__ import annotations

import pytest

from src.db import (
    delete_weekly_recap, list_weekly_recaps,
    load_weekly_recap, save_weekly_recap,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Same pattern as tests/test_weekly_recap.py — redirect DB_PATH to tmp."""
    db_file = tmp_path / "test_recaps.db"
    import config.settings as settings_mod
    import src.db as db_mod
    monkeypatch.setattr(settings_mod, "DB_PATH", db_file)
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    db_mod.init_db()
    yield db_file


SAMPLE = {
    "weekly_summary": "Macro neutral, sentiment mixed.",
    "macro": {"regime_label": "Risk-on", "summary": "Spreads tight."},
    "sectors": [],
    "allocation": [],
}


def test_save_then_load_round_trip(temp_db):
    save_weekly_recap("2026-05-24", "gpt-4o-mini", SAMPLE, n_newsletters=3)
    out = load_weekly_recap("2026-05-24", "gpt-4o-mini")
    assert out == SAMPLE


def test_load_missing_returns_none(temp_db):
    assert load_weekly_recap("2026-05-24", "gpt-4o-mini") is None


def test_distinct_models_do_not_collide(temp_db):
    a = dict(SAMPLE, weekly_summary="mini")
    b = dict(SAMPLE, weekly_summary="4o")
    save_weekly_recap("2026-05-24", "gpt-4o-mini", a, n_newsletters=3)
    save_weekly_recap("2026-05-24", "gpt-4o", b, n_newsletters=3)
    assert load_weekly_recap("2026-05-24", "gpt-4o-mini")["weekly_summary"] == "mini"
    assert load_weekly_recap("2026-05-24", "gpt-4o")["weekly_summary"] == "4o"


def test_upsert_overwrites_payload_and_count(temp_db):
    save_weekly_recap("2026-05-24", "gpt-4o-mini", SAMPLE, n_newsletters=3)
    updated = dict(SAMPLE, weekly_summary="rewritten")
    save_weekly_recap("2026-05-24", "gpt-4o-mini", updated, n_newsletters=5)
    out = load_weekly_recap("2026-05-24", "gpt-4o-mini")
    assert out["weekly_summary"] == "rewritten"
    hist = list_weekly_recaps()
    row = hist[hist["as_of_iso"] == "2026-05-24"].iloc[0]
    assert int(row["n_newsletters"]) == 5
    # Single row per (date, model) — upsert, not append.
    assert len(hist[(hist["as_of_iso"] == "2026-05-24")
                    & (hist["model"] == "gpt-4o-mini")]) == 1


def test_list_orders_most_recent_first(temp_db):
    save_weekly_recap("2026-05-10", "gpt-4o-mini", SAMPLE, n_newsletters=1)
    save_weekly_recap("2026-05-24", "gpt-4o-mini", SAMPLE, n_newsletters=2)
    save_weekly_recap("2026-05-17", "gpt-4o-mini", SAMPLE, n_newsletters=3)
    hist = list_weekly_recaps()
    assert hist["as_of_iso"].tolist() == [
        "2026-05-24", "2026-05-17", "2026-05-10",
    ]


def test_delete_removes_only_targeted_row(temp_db):
    save_weekly_recap("2026-05-24", "gpt-4o-mini", SAMPLE, n_newsletters=3)
    save_weekly_recap("2026-05-24", "gpt-4o", SAMPLE, n_newsletters=3)
    delete_weekly_recap("2026-05-24", "gpt-4o-mini")
    assert load_weekly_recap("2026-05-24", "gpt-4o-mini") is None
    assert load_weekly_recap("2026-05-24", "gpt-4o") is not None
