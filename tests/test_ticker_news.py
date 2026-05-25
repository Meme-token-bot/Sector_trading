"""Tests for src.ticker_news — RSS parsing + batched scoring + persistence.

No network, no OpenAI: the http_get and llm_client seams are stubbed.
"""
from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from src.schemas import ThemeNewsBatch, ThemeNewsScore
from src.ticker_news import fetch_headlines, score_themes, refresh_theme_news


_RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Google News</title>
  <item><title>Nvidia earnings crush estimates</title><link>http://a</link></item>
  <item><title>TSMC raises capex on AI demand</title><link>http://b</link></item>
  <item><title>Nvidia earnings crush estimates</title><link>http://c</link></item>
</channel></rss>"""


def test_fetch_headlines_parses_and_dedupes():
    heads = fetch_headlines("semis", http_get=lambda url: _RSS_FIXTURE)
    assert heads == ["Nvidia earnings crush estimates", "TSMC raises capex on AI demand"]


def test_fetch_headlines_limit():
    heads = fetch_headlines("semis", limit=1, http_get=lambda url: _RSS_FIXTURE)
    assert heads == ["Nvidia earnings crush estimates"]


def test_fetch_headlines_bad_xml_returns_empty():
    assert fetch_headlines("x", http_get=lambda url: "not xml <<<") == []


def test_fetch_headlines_http_error_returns_empty():
    def _boom(url):
        raise OSError("network down")
    assert fetch_headlines("x", http_get=_boom) == []


class _StubMessage:
    def __init__(self, parsed):
        self.parsed = parsed
        self.refusal = None


class _StubCompletion:
    def __init__(self, parsed):
        self.choices = [type("C", (), {"message": _StubMessage(parsed)})()]


def _stub_client(batch: ThemeNewsBatch):
    class _Parse:
        def parse(self, *, model, messages, response_format, temperature):
            # Sanity: only populated themes reach the model.
            assert "SEMIS" in messages[-1]["content"]
            return _StubCompletion(batch)

    class _Beta:
        class chat:
            completions = _Parse()

    return type("Client", (), {"beta": _Beta()})()


def test_score_themes_empty_input_skips_llm():
    # No headlines anywhere → no call, empty result.
    assert score_themes({"SEMIS": [], "URANIUM": []}) == {}


def test_score_themes_maps_scores():
    batch = ThemeNewsBatch(scores=[
        ThemeNewsScore(theme_key="SEMIS", score=4, n_headlines=2,
                       top_headline="Nvidia earnings crush estimates"),
    ])
    out = score_themes({"SEMIS": ["Nvidia earnings crush estimates"]},
                       llm_client=_stub_client(batch))
    assert out["SEMIS"]["score"] == 4
    assert out["SEMIS"]["n_headlines"] == 2
    assert out["SEMIS"]["top_headline"].startswith("Nvidia")


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "news.db"
    import config.settings as settings_mod
    import src.db as db_mod
    monkeypatch.setattr(settings_mod, "DB_PATH", db_file)
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    db_mod.init_db()
    yield db_file


def test_refresh_theme_news_persists(temp_db):
    from config.themes import THEMES
    batch = ThemeNewsBatch(scores=[
        ThemeNewsScore(theme_key="SEMIS", score=3, n_headlines=2, top_headline="h"),
    ])
    only_semis = [THEMES["SEMIS"]]
    rows = refresh_theme_news(
        as_of=date(2026, 5, 25),
        themes=only_semis,
        http_get=lambda url: _RSS_FIXTURE,
        llm_client=_stub_client(batch),
    )
    assert rows and rows[0]["theme_key"] == "SEMIS"
    with sqlite3.connect(temp_db) as c:
        n = c.execute("SELECT COUNT(*) FROM theme_news WHERE theme_key='SEMIS'").fetchone()[0]
    assert n == 1

    # Re-run same day overwrites (upsert), does not duplicate.
    refresh_theme_news(as_of=date(2026, 5, 25), themes=only_semis,
                       http_get=lambda url: _RSS_FIXTURE,
                       llm_client=_stub_client(batch))
    with sqlite3.connect(temp_db) as c:
        n = c.execute("SELECT COUNT(*) FROM theme_news WHERE theme_key='SEMIS'").fetchone()[0]
    assert n == 1
