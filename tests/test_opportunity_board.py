"""Tests for src.opportunity_board.build_opportunity_board.

Pure-function style -- small synthetic signals/targets/breakout_rows
inputs, no network/DB. Mirrors this repo's existing convention (e.g.
tests/test_signals.py, tests/test_tiger_drift.py) of hand-built minimal
frames rather than the real pipeline.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.opportunity_board import (
    COLUMNS,
    MODEL_BREAKOUT,
    MODEL_SPDR,
    build_opportunity_board,
)


def _signals(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).T


BASE_SIGNALS = _signals({
    "XLK": {"state": "NEW_BUY", "name": "Technology", "state_reason": "fresh buy"},
    "XLF": {"state": "CHASE", "name": "Financials", "state_reason": "extended"},
    "XLE": {"state": "HOLD", "name": "Energy", "state_reason": "wait"},
})
FUNDED_TARGETS = pd.Series({"XLK": 0.475, "XLF": 0.05})

BREAKOUT_ROWS = [
    {"ticker": "GDX", "label": "Gold Miners", "signal": "BUY",
     "conviction": 3, "reasons": ["r1", "r2"]},
    {"ticker": "IBIT", "label": "Bitcoin Trust", "signal": "HIGH_CONVICTION_BUY",
     "conviction": 5, "reasons": ["r3"]},
    {"ticker": "SLV", "label": "Silver", "signal": "WATCH",
     "conviction": 2, "reasons": []},
    {"ticker": "COIN", "label": "Coinbase", "signal": "RISK_EXIT",
     "conviction": 0, "reasons": []},
    {"ticker": "SOL-USD", "label": "Solana", "signal": "NOT_ENOUGH_DATA",
     "conviction": 0, "reasons": []},
    {"ticker": "XME", "label": "Metals", "signal": "NO_SIGNAL",
     "conviction": 1, "reasons": []},
]


def test_empty_inputs_return_empty_frame_with_correct_columns():
    out = build_opportunity_board(pd.DataFrame(), pd.Series(dtype=float), [],
                                  chase_weight_fraction=0.25)
    assert list(out.columns) == COLUMNS
    assert out.empty


def test_new_buy_and_funded_chase_included_hold_excluded():
    out = build_opportunity_board(BASE_SIGNALS, FUNDED_TARGETS, [],
                                  chase_weight_fraction=0.25)
    assert set(out["ticker"]) == {"XLK", "XLF"}
    assert "XLE" not in set(out["ticker"])


def test_chase_excluded_entirely_when_fraction_is_zero():
    out = build_opportunity_board(BASE_SIGNALS, FUNDED_TARGETS, [],
                                  chase_weight_fraction=0.0)
    assert set(out["ticker"]) == {"XLK"}


def test_chase_excluded_when_not_funded_even_if_fraction_positive():
    unfunded_targets = pd.Series({"XLK": 0.95})  # XLF never made it into targets
    out = build_opportunity_board(BASE_SIGNALS, unfunded_targets, [],
                                  chase_weight_fraction=0.25)
    assert set(out["ticker"]) == {"XLK"}


def test_only_buy_and_high_conviction_buy_breakout_rows_included():
    out = build_opportunity_board(pd.DataFrame(), pd.Series(dtype=float),
                                  BREAKOUT_ROWS, chase_weight_fraction=0.25)
    assert set(out["ticker"]) == {"GDX", "IBIT"}


def test_model_labels_are_explicit_and_spdr_precedes_breakout():
    out = build_opportunity_board(BASE_SIGNALS, FUNDED_TARGETS, BREAKOUT_ROWS,
                                  chase_weight_fraction=0.25)
    assert set(out["model"]) == {MODEL_SPDR, MODEL_BREAKOUT}
    models = out["model"].tolist()
    first_breakout = models.index(MODEL_BREAKOUT)
    assert all(m == MODEL_SPDR for m in models[:first_breakout])
    assert all(m == MODEL_BREAKOUT for m in models[first_breakout:])


def test_spdr_group_ranked_by_supplied_expression_score_descending():
    scores = {"XLK": 40, "XLF": 90}
    out = build_opportunity_board(BASE_SIGNALS, FUNDED_TARGETS, [],
                                  expression_scores=scores,
                                  chase_weight_fraction=0.25)
    spdr = out[out["model"] == MODEL_SPDR].reset_index(drop=True)
    assert spdr.loc[0, ["ticker", "group_rank"]].tolist() == ["XLF", 1]
    assert spdr.loc[1, ["ticker", "group_rank"]].tolist() == ["XLK", 2]


def test_missing_expression_score_sorts_last_and_does_not_crash():
    signals = _signals({
        "XLK": {"state": "NEW_BUY", "name": "Technology", "state_reason": "fresh buy"},
        "XLY": {"state": "NEW_BUY", "name": "Discretionary", "state_reason": "fresh buy"},
    })
    targets = pd.Series({"XLK": 0.475, "XLY": 0.475})
    out = build_opportunity_board(signals, targets, [],
                                  expression_scores={"XLK": 55},  # XLY missing
                                  chase_weight_fraction=0.25)
    assert out.iloc[0]["ticker"] == "XLK"
    assert out.iloc[1]["ticker"] == "XLY"
    assert pd.isna(out.iloc[1]["score"])
    assert out.iloc[1]["score_display"] == "—"


def test_breakout_group_ranked_by_own_conviction_descending():
    out = build_opportunity_board(pd.DataFrame(), pd.Series(dtype=float),
                                  BREAKOUT_ROWS, chase_weight_fraction=0.25)
    bo = out.reset_index(drop=True)
    assert bo.loc[0, ["ticker", "group_rank"]].tolist() == ["IBIT", 1]
    assert bo.loc[1, ["ticker", "group_rank"]].tolist() == ["GDX", 2]


def test_score_display_strings_never_blend_the_two_scales():
    out = build_opportunity_board(BASE_SIGNALS, FUNDED_TARGETS, BREAKOUT_ROWS,
                                  expression_scores={"XLK": 80, "XLF": 40},
                                  chase_weight_fraction=0.25)
    spdr_displays = out.loc[out["model"] == MODEL_SPDR, "score_display"]
    breakout_displays = out.loc[out["model"] == MODEL_BREAKOUT, "score_display"]
    assert all("/100" in v for v in spdr_displays)
    assert all("/5" in v for v in breakout_displays)


def test_group_rank_is_1_based_within_each_model_not_a_global_rank():
    out = build_opportunity_board(BASE_SIGNALS, FUNDED_TARGETS, BREAKOUT_ROWS,
                                  expression_scores={"XLK": 80, "XLF": 40},
                                  chase_weight_fraction=0.25)
    for _, grp in out.groupby("model"):
        assert sorted(grp["group_rank"].tolist()) == list(range(1, len(grp) + 1))


def test_default_chase_weight_fraction_resolves_from_params():
    """No chase_weight_fraction passed -> falls back to
    config.settings.PARAMS.chase_weight_fraction (0.25 in this repo), so the
    funded CHASE row (XLF) is still included exactly as if 0.25 had been
    passed explicitly."""
    out = build_opportunity_board(BASE_SIGNALS, FUNDED_TARGETS, [])
    assert set(out["ticker"]) == {"XLK", "XLF"}