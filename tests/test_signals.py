"""Tests for src.signals — conviction score + sentiment-quality propagation.

Focuses on the new additive columns. The existing BUY/HOLD/SELL convergence
rules are not retested here (they remain governed by the project's hand-
verified scenarios).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import PARAMS
from src.signals import build_signals, refine_signals, target_weights


# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic frames that match what compute_sector_metrics
# would produce (only the columns refine_signals reads).
# ---------------------------------------------------------------------------

def _base_signal_row(
    *,
    signal: str = "BUY",
    reasons: str = "",
    rs3: float = 0.0,
    above_sma: bool = True,
    extension_pct: float = 0.02,
    sentiment_score: float = 0.0,
    n_obs: int = 1,
    score_stdev: float = 0.0,
    score_min: float = 0.0,
    score_max: float = 0.0,
) -> dict:
    return {
        "name": "Test",
        "price": 100.0,
        "sma200": 98.0,
        "above_sma": above_sma,
        "extension_pct": extension_pct,
        "return_3m": rs3 + 0.02,
        "spy_return_3m": 0.02,
        "relative_strength_3m": rs3,
        "rs_rank": 1,
        "sentiment_score": sentiment_score,
        "n_obs": n_obs,
        "score_stdev": score_stdev,
        "score_min": score_min,
        "score_max": score_max,
        "signal": signal,
        "reasons": reasons,
    }


def _frame(**kwargs) -> pd.DataFrame:
    """Build a one-row signals frame indexed by ticker 'XLK'."""
    row = _base_signal_row(**kwargs)
    return pd.DataFrame({"XLK": row}).T


def _macro(tailwinds: int = 1, headwinds: int = 0) -> pd.DataFrame:
    """Single-sector macro_alignment frame. Conviction/override key off the
    net (tailwinds - headwinds), so callers set those directly."""
    denom = tailwinds + headwinds
    ratio = (tailwinds / denom) if denom else 0.0
    return pd.DataFrame(
        {"tailwinds": [tailwinds], "headwinds": [headwinds], "neutral": [0],
         "ratio": [ratio], "detail": [[]]},
        index=pd.Index(["XLK"], name="sector"),
    )


# ---------------------------------------------------------------------------
# Conviction scoring — each marginal point in isolation.
# ---------------------------------------------------------------------------

def test_conviction_zero_baseline():
    """Nothing positive — conviction == 0."""
    df = _frame(rs3=-0.01, sentiment_score=0.0)
    out = refine_signals(df)
    assert out.loc["XLK", "conviction"] == 0


def test_conviction_point_for_positive_rs():
    """rs3 > 0 but below strong_rs_margin → +1 only."""
    df = _frame(rs3=0.01, sentiment_score=0.0)
    assert PARAMS.strong_rs_margin > 0.01
    out = refine_signals(df)
    assert out.loc["XLK", "conviction"] == 1


def test_conviction_point_for_strong_rs():
    """rs3 above strong_rs_margin → +1 (positive) +1 (strong) = 2."""
    df = _frame(rs3=PARAMS.strong_rs_margin + 0.01, sentiment_score=0.0)
    out = refine_signals(df)
    assert out.loc["XLK", "conviction"] == 2


def test_conviction_point_for_high_sentiment():
    """sentiment >= buy_threshold+1 → +1."""
    df = _frame(rs3=-0.01,
                sentiment_score=PARAMS.buy_sentiment_threshold + 1)
    out = refine_signals(df)
    assert out.loc["XLK", "conviction"] == 1


def test_conviction_point_for_consecutive_buy_weeks():
    """consecutive_buy_weeks >= 2 → +1. Achieved via a 2-row BUY history."""
    df = _frame(rs3=-0.01, sentiment_score=0.0, signal="BUY")
    history = pd.DataFrame(
        [{"XLK": "BUY"}, {"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-08", "2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    # Only the 2-consec-BUY component fires; ext/sma keep state on a buy-class
    # value but that's irrelevant to conviction.
    assert out.loc["XLK", "consecutive_buy_weeks"] == 2
    assert out.loc["XLK", "conviction"] == 1


def test_conviction_point_for_macro_tailwind():
    """net macro lean >= +1 → +1."""
    df = _frame(rs3=-0.01, sentiment_score=0.0)
    out = refine_signals(df, macro_alignment=_macro(tailwinds=1, headwinds=0))
    assert out.loc["XLK", "conviction"] == 1


def test_conviction_no_point_when_macro_net_zero():
    """Equal tailwinds/headwinds → net 0 → no macro point."""
    df = _frame(rs3=-0.01, sentiment_score=0.0)
    out = refine_signals(df, macro_alignment=_macro(tailwinds=2, headwinds=2))
    assert out.loc["XLK", "conviction"] == 0


def test_conviction_macro_headwind_is_symmetric_penalty():
    """net macro lean <= -1 subtracts a point (clamped at 0)."""
    # One positive component (rs>0) then a macro headwind nets it back to 0.
    df = _frame(rs3=0.01, sentiment_score=0.0)
    base = refine_signals(df)
    assert base.loc["XLK", "conviction"] == 1
    out = refine_signals(df, macro_alignment=_macro(tailwinds=0, headwinds=1))
    assert out.loc["XLK", "conviction"] == 0


def test_conviction_clamped_at_zero_floor():
    """A headwind on a zero-baseline sector cannot push conviction negative."""
    df = _frame(rs3=-0.01, sentiment_score=0.0)
    out = refine_signals(df, macro_alignment=_macro(tailwinds=0, headwinds=3))
    assert out.loc["XLK", "conviction"] == 0


def test_conviction_max_without_macro_is_four():
    """All four non-macro components active → 4 (max without macro frame)."""
    df = _frame(
        rs3=PARAMS.strong_rs_margin + 0.02,
        sentiment_score=PARAMS.buy_sentiment_threshold + 1.5,
        signal="BUY",
    )
    history = pd.DataFrame(
        [{"XLK": "BUY"}, {"XLK": "BUY"}, {"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-01", "2026-05-08", "2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    assert out.loc["XLK", "conviction"] == 4


def test_conviction_max_with_macro_is_five():
    df = _frame(
        rs3=PARAMS.strong_rs_margin + 0.02,
        sentiment_score=PARAMS.buy_sentiment_threshold + 1.5,
        signal="BUY",
    )
    history = pd.DataFrame(
        [{"XLK": "BUY"}, {"XLK": "BUY"}, {"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-01", "2026-05-08", "2026-05-15"]),
    )
    out = refine_signals(df, history=history,
                         macro_alignment=_macro(tailwinds=1, headwinds=0))
    assert out.loc["XLK", "conviction"] == 5


def test_refine_callable_with_no_macro_data():
    """No macro frame supplied → macro component is 0; function still runs."""
    df = _frame(rs3=0.05, sentiment_score=3.5)
    out = refine_signals(df)  # history=None, macro_alignment=None
    # conviction should be computable without macro_alignment.
    assert "conviction" in out.columns
    assert 0 <= out.loc["XLK", "conviction"] <= 4


# ---------------------------------------------------------------------------
# Sentiment-quality propagation through build_signals.
# ---------------------------------------------------------------------------

def test_build_signals_includes_quality_columns_when_available():
    metrics = pd.DataFrame({
        "XLK": {
            "name": "Tech", "price": 100.0, "sma200": 98.0,
            "above_sma": True, "extension_pct": 0.02,
            "return_3m": 0.05, "spy_return_3m": 0.02,
            "relative_strength_3m": 0.03, "rs_rank": 1,
        }
    }).T
    sentiment = pd.DataFrame({
        "score": [2.5], "n_obs": [4],
        "score_stdev": [0.5], "score_min": [2.0], "score_max": [3.0],
    }, index=pd.Index(["XLK"], name="ticker"))

    out = build_signals(metrics, sentiment)
    assert out.loc["XLK", "score_stdev"] == pytest.approx(0.5)
    assert out.loc["XLK", "score_min"] == pytest.approx(2.0)
    assert out.loc["XLK", "score_max"] == pytest.approx(3.0)


def test_build_signals_defaults_quality_columns_when_sentiment_empty():
    metrics = pd.DataFrame({
        "XLK": {
            "name": "Tech", "price": 100.0, "sma200": 98.0,
            "above_sma": True, "extension_pct": 0.02,
            "return_3m": 0.05, "spy_return_3m": 0.02,
            "relative_strength_3m": 0.03, "rs_rank": 1,
        }
    }).T
    out = build_signals(metrics, pd.DataFrame())
    assert out.loc["XLK", "score_stdev"] == pytest.approx(0.0)
    assert np.isnan(out.loc["XLK", "score_min"])
    assert np.isnan(out.loc["XLK", "score_max"])


# ---------------------------------------------------------------------------
# Macro veto / override pass — state transitions on a STRONG net lean.
# ---------------------------------------------------------------------------

def _strong_head() -> pd.DataFrame:
    return _macro(tailwinds=0, headwinds=PARAMS.macro_strong_count)


def _strong_tail() -> pd.DataFrame:
    return _macro(tailwinds=PARAMS.macro_strong_count, headwinds=0)


def test_macro_veto_downgrades_new_buy_to_hold():
    """A fresh NEW_BUY facing a strong macro headwind is vetoed to HOLD."""
    df = _frame(signal="BUY", rs3=0.05, extension_pct=0.02, above_sma=True)
    out = refine_signals(df, history=None, macro_alignment=_strong_head())
    assert out.loc["XLK", "state"] == "HOLD"
    assert "veto" in out.loc["XLK", "state_reason"].lower()


def test_macro_headwind_downgrades_stale_buy_to_reduce():
    """HOLD_IF_LONG (stale BUY) + strong headwind → REDUCE."""
    df = _frame(signal="BUY", rs3=0.05, extension_pct=0.02, above_sma=True)
    weeks = max(PARAMS.stale_buy_weeks, 1)
    history = pd.DataFrame(
        [{"XLK": "BUY"} for _ in range(weeks)],
        index=pd.to_datetime(
            [f"2026-0{(i % 8) + 1}-0{(i % 9) + 1}" for i in range(weeks)]
        ),
    )
    # Sanity: without macro this is HOLD_IF_LONG.
    base = refine_signals(df, history=history)
    assert base.loc["XLK", "state"] == "HOLD_IF_LONG"
    out = refine_signals(df, history=history, macro_alignment=_strong_head())
    assert out.loc["XLK", "state"] == "REDUCE"


def test_macro_tailwind_elevates_hold_to_watch():
    """A lagging HOLD that's above SMA200 + strong macro tailwind → WATCH."""
    df = _frame(signal="HOLD", rs3=-0.05, above_sma=True)
    out = refine_signals(df, history=None, macro_alignment=_strong_tail())
    assert out.loc["XLK", "state"] == "WATCH"


def test_macro_tailwind_does_not_rescue_below_sma():
    """WATCH requires above_sma — a broken chart stays HOLD."""
    df = _frame(signal="HOLD", rs3=-0.05, above_sma=False)
    out = refine_signals(df, history=None, macro_alignment=_strong_tail())
    assert out.loc["XLK", "state"] == "HOLD"


def test_balanced_macro_does_not_override_state():
    """net == 0 (tailwinds == headwinds) is below macro_strong_count, so the
    state is left untouched even though the sector has macro readings."""
    df = _frame(signal="BUY", rs3=0.05, extension_pct=0.02, above_sma=True)
    out = refine_signals(df, history=None,
                         macro_alignment=_macro(tailwinds=2, headwinds=2))
    assert out.loc["XLK", "state"] == "NEW_BUY"


def test_target_weights_excludes_watch_and_vetoed_hold():
    """WATCH and a macro-vetoed HOLD must not receive capital."""
    frame = pd.DataFrame({
        "BUYER": _base_signal_row(signal="BUY", rs3=0.05, above_sma=True,
                                  extension_pct=0.02),
        "VETOED": _base_signal_row(signal="BUY", rs3=0.05, above_sma=True,
                                   extension_pct=0.02),
        "WATCHER": _base_signal_row(signal="HOLD", rs3=-0.05, above_sma=True),
    }).T
    macro = pd.DataFrame({
        "tailwinds": [0, 0, PARAMS.macro_strong_count],
        "headwinds": [0, PARAMS.macro_strong_count, 0],
        "neutral": [0, 0, 0],
        "ratio": [0.0, 0.0, 1.0],
        "detail": [[], [], []],
    }, index=pd.Index(["BUYER", "VETOED", "WATCHER"], name="sector"))
    out = refine_signals(frame, history=None, macro_alignment=macro)
    assert out.loc["BUYER", "state"] == "NEW_BUY"
    assert out.loc["VETOED", "state"] == "HOLD"
    assert out.loc["WATCHER", "state"] == "WATCH"
    tw = target_weights(out)
    assert list(tw.index) == ["BUYER"]


# ---------------------------------------------------------------------------
# Partial-CHASE sleeve in target_weights — selected by walk-forward sweep
# with 6/6 fold consensus (mean OOS lift +5.35pp). See WALK_FORWARD_REPORT.md.
# ---------------------------------------------------------------------------

def _state_frame(states: dict[str, str]) -> pd.DataFrame:
    """Build a refined-signals frame with explicit per-ticker states.
    Other columns are filled with safe defaults so target_weights is the
    only thing under test.
    """
    rows = []
    for tkr, st in states.items():
        rows.append({
            "ticker": tkr, "state": st, "signal": "BUY" if st in ("NEW_BUY", "HOLD_IF_LONG", "CHASE") else "HOLD",
            "above_sma": True, "extension_pct": 0.05,
            "relative_strength_3m": 0.03, "rs_rank": 1,
            "sentiment_score": 2.0, "n_obs": 1, "conviction": 3,
        })
    return pd.DataFrame(rows).set_index("ticker")


def test_target_weights_excludes_chase_when_fraction_zero():
    """chase_weight_fraction=0 reverts to original behaviour (CHASE excluded)."""
    frame = _state_frame({"XLK": "CHASE", "XLY": "NEW_BUY"})
    tw = target_weights(frame, cash_buffer=0.05, chase_weight_fraction=0.0)
    assert "XLK" not in tw.index
    assert "XLY" in tw.index
    assert tw["XLY"] == pytest.approx(0.95)  # 1.0 - 0.05 cash, denom=1


def test_target_weights_includes_chase_capped_by_cash_buffer():
    """chase_weight_fraction=0.25 sleeves CHASE at up to 25% of confirmed
    per-name, capped by the cash buffer so total weight ≤ 1.0. With one
    CHASE name and a 5% buffer, the desired 0.11875 sleeve scales down
    to consume exactly the buffer (0.05) — no implicit leverage."""
    frame = _state_frame({"XLK": "CHASE", "XLY": "NEW_BUY", "XLF": "NEW_BUY"})
    tw = target_weights(frame, cash_buffer=0.05, chase_weight_fraction=0.25)
    # Two confirmed (XLY, XLF) split (1 - 0.05) / 2 = 0.475 each. Untouched.
    assert tw["XLY"] == pytest.approx(0.475)
    assert tw["XLF"] == pytest.approx(0.475)
    # CHASE allocation scaled down to fit the cash buffer.
    assert tw["XLK"] == pytest.approx(0.05)
    assert tw.sum() == pytest.approx(1.0)


def test_target_weights_chase_uncapped_when_buffer_is_generous():
    """When the buffer easily covers desired CHASE demand, CHASE is sized
    at exactly chase_weight_fraction × per_name (no scaling kicks in)."""
    states = {"A": "NEW_BUY", "B": "NEW_BUY", "C": "NEW_BUY",
              "D": "NEW_BUY", "E": "CHASE"}
    frame = _state_frame(states)
    tw = target_weights(frame, cash_buffer=0.20, chase_weight_fraction=0.25)
    per_name = 0.80 / 4
    assert tw["A"] == pytest.approx(per_name)
    assert tw["E"] == pytest.approx(per_name * 0.25)
    # Buffer not fully consumed — still some cash.
    assert tw.sum() < 1.0


def test_target_weights_chase_does_not_steal_from_confirmed():
    """Adding a CHASE sleeve must NOT reduce confirmed sizes — it comes out
    of cash (and is capped at the cash buffer)."""
    confirmed_only = _state_frame({"XLY": "NEW_BUY", "XLF": "NEW_BUY"})
    with_chase = _state_frame({"XLK": "CHASE", "XLY": "NEW_BUY",
                                "XLF": "NEW_BUY"})
    tw_a = target_weights(confirmed_only, cash_buffer=0.05,
                           chase_weight_fraction=0.25)
    tw_b = target_weights(with_chase, cash_buffer=0.05,
                           chase_weight_fraction=0.25)
    assert tw_b["XLY"] == pytest.approx(tw_a["XLY"])
    assert tw_b["XLF"] == pytest.approx(tw_a["XLF"])


def test_target_weights_defaults_to_params_chase_fraction():
    """When chase_weight_fraction=None the live PARAMS default is used."""
    frame = _state_frame({"XLK": "CHASE", "XLY": "NEW_BUY"})
    tw = target_weights(frame, cash_buffer=0.05)
    # Confirmed XLY at 0.95; CHASE XLK desired 0.95 * fraction, capped
    # at the 0.05 buffer.
    desired = 0.95 * float(PARAMS.chase_weight_fraction)
    expected = min(desired, 0.05) if PARAMS.chase_weight_fraction > 0 else 0.0
    if PARAMS.chase_weight_fraction > 0:
        assert tw["XLK"] == pytest.approx(expected)
    else:
        assert "XLK" not in tw.index


# ---------------------------------------------------------------------------
# Regime-aware promote_chase — CHASE folded into confirmed set at full weight.
# Used by the backtest's bull overlay (BacktestConfig.regime_aware).
# ---------------------------------------------------------------------------

def test_promote_chase_folds_chase_into_equal_weight():
    """promote_chase=True treats CHASE like NEW_BUY: full per-name weight,
    no partial sleeve. With buffer 0 the book is fully invested."""
    frame = _state_frame({"XLK": "CHASE", "XLY": "NEW_BUY", "XLF": "NEW_BUY"})
    tw = target_weights(frame, cash_buffer=0.0, promote_chase=True)
    # Three confirmed names split the whole book equally.
    assert tw["XLK"] == pytest.approx(1.0 / 3)
    assert tw["XLY"] == pytest.approx(1.0 / 3)
    assert tw["XLF"] == pytest.approx(1.0 / 3)
    assert tw.sum() == pytest.approx(1.0)


def test_promote_chase_gives_chase_more_than_partial_sleeve():
    """The whole point: in a bull, a promoted CHASE leader gets a real
    equal-weight slice, not the buffer-capped sleeve it gets when gated."""
    frame = _state_frame({"XLK": "CHASE", "XLY": "NEW_BUY"})
    gated = target_weights(frame, cash_buffer=0.05, chase_weight_fraction=0.25)
    promoted = target_weights(frame, cash_buffer=0.0, promote_chase=True)
    assert promoted["XLK"] > gated["XLK"]
    assert promoted["XLK"] == pytest.approx(0.5)  # split 1.0 with XLY


def test_promote_chase_false_is_unchanged():
    """promote_chase defaults False → identical to today's gated behaviour."""
    frame = _state_frame({"XLK": "CHASE", "XLY": "NEW_BUY"})
    default = target_weights(frame, cash_buffer=0.05, chase_weight_fraction=0.25)
    explicit = target_weights(frame, cash_buffer=0.05, chase_weight_fraction=0.25,
                              promote_chase=False)
    assert default.equals(explicit)


# ---------------------------------------------------------------------------
# State-transition hysteresis (1-week confirmation on entry, 1-week grace on
# exit). Motivated by a live case: a sector oscillating NEW_BUY <-> REDUCE
# four times in a month because RS and extension were both hovering right on
# their thresholds simultaneously. SELL is deliberately untouched by all of
# this -- a hard SELL gate still fires immediately (see test_signals.py's
# existing macro-veto tests above, which pass history=None and therefore
# exercise the pre-hysteresis fallback path unchanged).
# ---------------------------------------------------------------------------

def test_first_week_buy_with_history_is_watch_not_new_buy():
    """A fresh raw BUY, with real (if short) history available to check
    against, must wait one more week before it's a confirmed NEW_BUY."""
    df = _frame(signal="BUY", rs3=0.05, extension_pct=0.02, above_sma=True)
    history = pd.DataFrame(
        [{"XLK": "HOLD"}],
        index=pd.to_datetime(["2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    assert out.loc["XLK", "state"] == "WATCH"
    assert "first week" in out.loc["XLK", "state_reason"].lower()


def test_second_consecutive_buy_week_confirms_new_buy():
    """One prior BUY week (n_buy=1) plus today's BUY is the 2nd consecutive
    week -- now it's a confirmed, actionable NEW_BUY."""
    df = _frame(signal="BUY", rs3=0.05, extension_pct=0.02, above_sma=True)
    history = pd.DataFrame(
        [{"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    assert out.loc["XLK", "state"] == "NEW_BUY"


def test_no_history_still_gives_immediate_new_buy():
    """Documented fallback: with no history to check hysteresis against at
    all, a fresh raw BUY is still reported as NEW_BUY immediately -- the
    pre-existing, unconfirmed behaviour is unchanged when there's nothing
    to compare against."""
    df = _frame(signal="BUY", rs3=0.05, extension_pct=0.02, above_sma=True)
    out = refine_signals(df, history=None)
    assert out.loc["XLK", "state"] == "NEW_BUY"


def test_empty_history_frame_also_falls_back_to_immediate_new_buy():
    df = _frame(signal="BUY", rs3=0.05, extension_pct=0.02, above_sma=True)
    out = refine_signals(df, history=pd.DataFrame(columns=["XLK"]))
    assert out.loc["XLK", "state"] == "NEW_BUY"


def test_hold_right_after_a_buy_week_gets_one_week_grace():
    """Was BUY last week (n_buy=1), cooled to raw HOLD this week -- the
    XLV case: one soft snapshot shouldn't immediately trigger a trim."""
    df = _frame(signal="HOLD", rs3=-0.01, above_sma=True)
    history = pd.DataFrame(
        [{"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    assert out.loc["XLK", "state"] == "HOLD_IF_LONG"
    assert "cooled" in out.loc["XLK", "state_reason"].lower()


def test_hold_after_two_non_buy_weeks_is_reduce():
    """BUY, then HOLD, then HOLD again today -- the second consecutive
    non-BUY week confirms the demotion to REDUCE."""
    df = _frame(signal="HOLD", rs3=-0.01, above_sma=True)
    history = pd.DataFrame(
        [{"XLK": "BUY"}, {"XLK": "HOLD"}],
        index=pd.to_datetime(["2026-05-08", "2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    assert out.loc["XLK", "state"] == "REDUCE"


def test_grace_period_hold_if_long_still_gets_target_weight_capital():
    """The grace-period HOLD_IF_LONG must be treated identically to a
    genuinely-stale HOLD_IF_LONG by target_weights() -- same "hold if you
    own it" semantics, same sizing contract; no special-casing needed."""
    df = _frame(signal="HOLD", rs3=-0.01, above_sma=True)
    history = pd.DataFrame(
        [{"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    assert out.loc["XLK", "state"] == "HOLD_IF_LONG"
    tw = target_weights(out, cash_buffer=0.05)
    assert "XLK" in tw.index


def test_watch_from_first_week_buy_gets_no_target_weight_capital():
    """Matching WATCH's existing contract: a first-week, unconfirmed BUY
    must not receive capital, same as the pre-existing macro-override
    WATCH case."""
    df = _frame(signal="BUY", rs3=0.05, extension_pct=0.02, above_sma=True)
    history = pd.DataFrame(
        [{"XLK": "HOLD"}],
        index=pd.to_datetime(["2026-05-15"]),
    )
    out = refine_signals(df, history=history)
    assert out.loc["XLK", "state"] == "WATCH"
    tw = target_weights(out, cash_buffer=0.05)
    assert "XLK" not in tw.index


def test_strong_macro_headwind_still_overrides_a_grace_period_hold_if_long():
    """Macro can still cut risk through a grace period -- 'macro can cut
    risk freely but cannot add unconfirmed price risk' applies here too,
    same as it already does for a genuinely-stale HOLD_IF_LONG."""
    df = _frame(signal="HOLD", rs3=-0.01, above_sma=True)
    history = pd.DataFrame(
        [{"XLK": "BUY"}],
        index=pd.to_datetime(["2026-05-15"]),
    )
    base = refine_signals(df, history=history)
    assert base.loc["XLK", "state"] == "HOLD_IF_LONG"
    out = refine_signals(df, history=history, macro_alignment=_strong_head())
    assert out.loc["XLK", "state"] == "REDUCE"


def test_flip_flop_scenario_is_stable_across_a_boundary_hugging_sequence():
    """Reproduces the live XLI pattern end to end: raw signal oscillating
    BUY/HOLD/BUY/HOLD/BUY across five consecutive weeks, with real history
    available throughout (seeded with one prior week so hysteresis applies
    from the first observed week -- an entirely empty history correctly
    falls back to the unconfirmed behaviour per the documented contract,
    covered separately by test_no_history_still_gives_immediate_new_buy).

    Before this fix every BUY week reported NEW_BUY and every HOLD week
    reported REDUCE -- four hard flips in four transitions. After the fix,
    only a SUSTAINED (2-consecutive-week) read is ever reported as NEW_BUY
    or REDUCE; single-week wobbles show as WATCH or HOLD_IF_LONG instead.
    """
    sequence = ["BUY", "HOLD", "BUY", "HOLD", "BUY"]
    dates = pd.to_datetime(["2026-07-27", "2026-08-03", "2026-08-06",
                            "2026-08-13", "2026-08-21"])
    observed_states = []
    # Seed one prior (pre-sequence) week so has_history=True throughout.
    history = pd.DataFrame([{"XLK": "HOLD"}], index=pd.to_datetime(["2026-07-20"]))
    for sig, d in zip(sequence, dates):
        df = _frame(signal=sig, rs3=(0.01 if sig == "BUY" else -0.01),
                    extension_pct=0.06, above_sma=True)
        out = refine_signals(df, history=history)
        observed_states.append(out.loc["XLK", "state"])
        history = pd.concat([history, pd.DataFrame([{"XLK": sig}], index=[d])])

    # None of the hard, capital-moving labels (NEW_BUY / REDUCE) appear
    # until a read has actually been sustained for 2 consecutive weeks.
    assert observed_states[0] == "WATCH"          # week 1 BUY: unconfirmed
    assert observed_states[1] == "HOLD_IF_LONG"    # week 2 HOLD: one week's grace
    assert observed_states[2] == "WATCH"           # week 3 BUY: unconfirmed again
    assert observed_states[3] == "HOLD_IF_LONG"    # week 4 HOLD: grace again
    assert observed_states[4] == "WATCH"           # week 5 BUY: unconfirmed again
    # Zero hard flips into a capital-moving state across the whole noisy
    # run -- this is exactly the flip-flop the fix targets.
    assert "NEW_BUY" not in observed_states
    assert "REDUCE" not in observed_states