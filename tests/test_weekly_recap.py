"""Tests for src.weekly_recap — DB-driven gather + schema round-trip.

Does NOT call OpenAI. If you want to smoke-test generate_recap(), monkey-patch
src.weekly_recap._get_openai_client to return a stub.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from src.schemas import (
    Allocation, AllocationTilt, MacroNarrative, NewsletterConsensus,
    RegimeLabel, SectorRecap, WeeklyRecap,
)


# ---------------------------------------------------------------------------
# Helpers — seed a temp SQLite at the path config.settings.DB_PATH resolves to.
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file and re-init the schema.

    src.db imports DB_PATH at module load (``from config.settings import
    DB_PATH``), so patching only ``config.settings.DB_PATH`` doesn't reach
    the binding that ``_conn`` reads.  Patch both so the test is robust
    regardless of import order.
    """
    db_file = tmp_path / "test_sentiment.db"
    import config.settings as settings_mod
    import src.db as db_mod
    monkeypatch.setattr(settings_mod, "DB_PATH", db_file)
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    db_mod.init_db()
    yield db_file


def _insert_newsletter(db_path, author: str, pub_date: date,
                       summary: str, bias: str = "Neutral") -> int:
    with sqlite3.connect(db_path) as c:
        cur = c.execute(
            """INSERT INTO newsletters
               (content_hash, author, publication_date, overall_macro_bias, summary)
               VALUES (?, ?, ?, ?, ?)""",
            (f"hash-{author}-{pub_date.isoformat()}", author,
             pub_date.isoformat(), bias, summary),
        )
        return cur.lastrowid


def _insert_rating(db_path, newsletter_id: int, ticker: str,
                   score: int, reasoning: str) -> None:
    with sqlite3.connect(db_path) as c:
        c.execute(
            """INSERT INTO sector_ratings
               (newsletter_id, ticker, sentiment_score, reasoning)
               VALUES (?, ?, ?, ?)""",
            (newsletter_id, ticker, score, reasoning),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gather_context_empty_window(temp_db):
    """No newsletters in the last 7 days → n_newsletters=0, empty lists."""
    # Patch macro builders to no-op so we don't hit yfinance / FRED.
    with patch("src.weekly_recap._build_macro_snapshots", return_value=[]):
        from src.weekly_recap import gather_context
        ctx = gather_context(as_of=date(2026, 5, 23), lookback_days=7)

    assert ctx.n_newsletters == 0
    assert ctx.newsletters == []
    assert ctx.sector_rollups == []
    assert ctx.macro_snapshots == []
    assert ctx.as_of == date(2026, 5, 23)
    assert ctx.lookback_days == 7


def test_gather_context_rollup(temp_db):
    """Two newsletters across three sectors — per-sector mean and n correct."""
    as_of = date(2026, 5, 23)
    # Newsletter A (yesterday) covers XLK(+3) and XLF(-2).
    nid_a = _insert_newsletter(temp_db, "Author A",
                                as_of - timedelta(days=1),
                                "A's macro thesis.")
    _insert_rating(temp_db, nid_a, "XLK", 3, "AI capex is broadening.")
    _insert_rating(temp_db, nid_a, "XLF", -2, "Banks face NIM compression.")

    # Newsletter B (3 days ago) covers XLK(+1), XLF(+4), XLE(-1).
    nid_b = _insert_newsletter(temp_db, "Author B",
                                as_of - timedelta(days=3),
                                "B's thesis.", bias="Expansionary")
    _insert_rating(temp_db, nid_b, "XLK",  1, "Steady tech earnings.")
    _insert_rating(temp_db, nid_b, "XLF",  4, "Capital markets reopening.")
    _insert_rating(temp_db, nid_b, "XLE", -1, "Demand softens into recession.")

    # And one outside the window — must be ignored.
    nid_old = _insert_newsletter(temp_db, "Author C",
                                  as_of - timedelta(days=30),
                                  "Old thesis.")
    _insert_rating(temp_db, nid_old, "XLK", -5, "Should NOT appear.")

    with patch("src.weekly_recap._build_macro_snapshots", return_value=[]):
        from src.weekly_recap import gather_context
        ctx = gather_context(as_of=as_of, lookback_days=7)

    assert ctx.n_newsletters == 2

    rollups = {r.ticker: r for r in ctx.sector_rollups}
    assert set(rollups) == {"XLK", "XLF", "XLE"}, (
        f"unexpected tickers: {set(rollups)}"
    )

    # XLK: scores 3, 1 → mean 2.0, n=2
    assert rollups["XLK"].mean_sentiment == pytest.approx(2.0)
    assert rollups["XLK"].n_obs == 2
    # XLF: scores -2, 4 → mean 1.0, n=2
    assert rollups["XLF"].mean_sentiment == pytest.approx(1.0)
    assert rollups["XLF"].n_obs == 2
    # XLE: score -1 only, n=1
    assert rollups["XLE"].mean_sentiment == pytest.approx(-1.0)
    assert rollups["XLE"].n_obs == 1

    # XLK's top-excerpt should be the higher-magnitude one (Author A, +3)
    # ahead of Author B (+1) when sorted by absolute sentiment.
    assert "Author A" in rollups["XLK"].top_excerpts[0]


def test_schema_round_trip():
    """WeeklyRecap → dump → reload preserves equality and enum values."""
    recap = WeeklyRecap(
        generated_for_week_ending=date(2026, 5, 23),
        n_newsletters=4,
        macro=MacroNarrative(
            regime_label=RegimeLabel.LATE_CYCLE,
            summary="Equity vol calm, credit spreads quietly widening. "
                    "Curve still inverted. The setup looks late-cycle, with "
                    "growth slowing but no break yet.",
            dominant_themes=["AI capex cooling", "Credit watching"],
            contradictions=["Bulls cite earnings; bears cite spreads."],
        ),
        sectors=[
            SectorRecap(
                ticker="XLK",
                sector_name="Technology",
                plain_language_summary="Tech leadership intact but narrower. "
                                       "Most macro models say overweight; one "
                                       "newsletter flags AI capex digestion.",
                newsletter_consensus=NewsletterConsensus.BULLISH,
                macro_alignment="Real yields easing supports duration; DXY "
                                "weak; helpful for XLK.",
                key_risks=["AI capex normalisation",
                           "Mag-7 concentration risk"],
            ),
            SectorRecap(
                ticker="UFO",
                sector_name="Space",
                plain_language_summary="No coverage this week; leaning on "
                                       "macro. Risk-on regime is supportive "
                                       "for thematic growth baskets.",
                newsletter_consensus=NewsletterConsensus.NO_COVERAGE,
                macro_alignment="High beta to risk-on tape; constructive "
                                "only if HY OAS stays tight.",
                key_risks=["Thin float", "Speculative coverage"],
            ),
        ],
        allocation=[
            Allocation(
                ticker="XLK",
                suggested_tilt=AllocationTilt.OVERWEIGHT,
                rationale="Bullish consensus + supportive real yields.",
            ),
            Allocation(
                ticker="UFO",
                suggested_tilt=AllocationTilt.EQUAL_WEIGHT,
                rationale="No newsletter signal — hold tactical only.",
            ),
        ],
        caveats="Informational only. Not personalised investment advice.",
    )

    blob = recap.model_dump_json()
    reloaded = WeeklyRecap.model_validate_json(blob)
    assert reloaded == recap
    # Enum round-trip — values not stringly-typed.
    assert reloaded.macro.regime_label == RegimeLabel.LATE_CYCLE
    assert reloaded.sectors[0].newsletter_consensus == NewsletterConsensus.BULLISH
    assert reloaded.sectors[1].newsletter_consensus == NewsletterConsensus.NO_COVERAGE
    assert reloaded.allocation[0].suggested_tilt == AllocationTilt.OVERWEIGHT


def test_generate_recap_with_stub_client(temp_db):
    """Smoke-test the plumbing: gather_context → generate_recap → WeeklyRecap.

    Monkeypatches _get_openai_client to a stub that returns a fixed recap.
    Confirms the user-content markdown is built (received by the stub) and
    the returned object is the parsed WeeklyRecap.
    """
    from datetime import date
    from src.schemas import (
        Allocation, AllocationTilt, MacroNarrative, NewsletterConsensus,
        RegimeLabel, SectorRecap, WeeklyRecap,
    )

    # Seed one newsletter so gather_context returns non-empty.
    nid = _insert_newsletter(temp_db, "Stub Author", date(2026, 5, 22),
                              "Stub thesis.")
    _insert_rating(temp_db, nid, "XLK", 2, "Tech holding up.")

    fixed_recap = WeeklyRecap(
        generated_for_week_ending=date(2026, 5, 23),
        n_newsletters=1,
        macro=MacroNarrative(
            regime_label=RegimeLabel.MIXED,
            summary="Mixed signals. " * 4,
            dominant_themes=["Theme A"],
            contradictions=[],
        ),
        sectors=[
            SectorRecap(
                ticker="XLK", sector_name="Technology",
                plain_language_summary="Constructive on tech. " * 2,
                newsletter_consensus=NewsletterConsensus.BULLISH,
                macro_alignment="Real yields supportive.",
                key_risks=["AI capex"],
            ),
        ],
        allocation=[
            Allocation(ticker="XLK", suggested_tilt=AllocationTilt.OVERWEIGHT,
                       rationale="Bullish + supportive macro."),
        ],
        caveats="Not advice.",
    )

    received_user_content: list[str] = []

    class _StubMessage:
        def __init__(self, parsed):
            self.parsed = parsed
            self.refusal = None

    class _StubChoice:
        def __init__(self, parsed):
            self.message = _StubMessage(parsed)

    class _StubCompletion:
        def __init__(self, parsed):
            self.choices = [_StubChoice(parsed)]

    class _StubParseAPI:
        def parse(self, *, model, messages, response_format, temperature):
            # Capture the user content for assertions below.
            for m in messages:
                if m["role"] == "user":
                    received_user_content.append(m["content"])
            return _StubCompletion(fixed_recap)

    class _StubBetaCompletions:
        completions = _StubParseAPI()

    class _StubBetaChat:
        completions = _StubParseAPI()

    class _StubBeta:
        chat = _StubBetaChat()

    class _StubClient:
        beta = _StubBeta()

    with patch("src.weekly_recap._build_macro_snapshots", return_value=[]), \
         patch("src.weekly_recap._get_openai_client",
               return_value=_StubClient()):
        from src.weekly_recap import gather_context, generate_recap
        ctx = gather_context(as_of=date(2026, 5, 23), lookback_days=7)
        result = generate_recap(ctx)

    assert result == fixed_recap
    assert received_user_content, "stub never received a user message"
    # The markdown serialiser should at minimum mention the author and ticker.
    assert "Stub Author" in received_user_content[0]
    assert "XLK" in received_user_content[0]
