"""Tests for src.macro_alignment.compute_macro_alignment.

Pure-function tests. No network, no DB.
"""
from __future__ import annotations

import math

import pytest

from config.settings import SECTOR_ETFS
from src.macro_alignment import compute_macro_alignment


def _r(value: float) -> dict:
    """Build a minimal macro payload with just a `current` reading."""
    return {"current": value}


def test_all_tailwind_for_xlf():
    # T10Y2Y steep + HY OAS low.
    readings = {
        "T10Y2Y": _r(1.0),   # > 0.5 -> tailwind; not inverted, no headwind
        "HY_OAS": _r(3.0),   # < 4   -> tailwind; not > 5, no headwind
    }
    df = compute_macro_alignment(readings)
    assert df.loc["XLF", "tailwinds"] == 2
    assert df.loc["XLF", "headwinds"] == 0
    assert df.loc["XLF", "neutral"] == 0
    assert df.loc["XLF", "ratio"] == pytest.approx(1.0)
    # Detail trace lists both contributing rules.
    detail = df.loc["XLF", "detail"]
    assert len(detail) == 2
    assert all(v == "tailwind" for _, v in detail)


def test_all_headwind_for_xlf():
    readings = {
        "T10Y2Y": _r(-0.5),  # < 0 -> headwind, also not > 0.5
        "HY_OAS": _r(6.0),   # > 5 -> headwind
    }
    df = compute_macro_alignment(readings)
    assert df.loc["XLF", "tailwinds"] == 0
    assert df.loc["XLF", "headwinds"] == 2
    assert df.loc["XLF", "ratio"] == pytest.approx(0.0)


def test_mixed_tailwind_and_headwind_for_xlk():
    # Real 10Y low (tailwind) but DXY strong (headwind) and VIX high (headwind).
    readings = {
        "REAL_10Y": _r(0.5),    # < 1 -> tailwind; not > 2, no headwind
        "DXY":      _r(106.0),  # > 105 -> headwind
        "VIX":      _r(30.0),   # > 25 -> headwind
    }
    df = compute_macro_alignment(readings)
    assert df.loc["XLK", "tailwinds"] == 1
    assert df.loc["XLK", "headwinds"] == 2
    assert df.loc["XLK", "ratio"] == pytest.approx(1 / 3)


def test_sector_with_no_relevant_readings():
    """A sector with no rules firing reports zeros and ratio 0.0."""
    # Empty readings — nothing fires for any sector.
    df = compute_macro_alignment({})
    assert df.loc["XLF", "tailwinds"] == 0
    assert df.loc["XLF", "headwinds"] == 0
    assert df.loc["XLF", "neutral"] == 0
    assert df.loc["XLF", "ratio"] == pytest.approx(0.0)
    assert df.loc["XLF", "detail"] == []


def test_neutral_counts_separately_from_tailwind():
    # XLU has a neutral rule on T10Y2Y inversion.
    readings = {
        "T10Y2Y":   _r(-0.2),  # inverted -> neutral
        "REAL_10Y": _r(0.5),   # < 1 -> tailwind
    }
    df = compute_macro_alignment(readings)
    assert df.loc["XLU", "neutral"] == 1
    assert df.loc["XLU", "tailwinds"] == 1
    # ratio uses tailwind / (tailwind + headwind) — neutral excluded.
    assert df.loc["XLU", "ratio"] == pytest.approx(1.0)


def test_every_sector_is_indexed():
    df = compute_macro_alignment({})
    assert set(df.index) == set(SECTOR_ETFS.keys())


def test_missing_payload_keys_are_skipped():
    # A reading marked with NaN should be treated as absent.
    readings = {
        "T10Y2Y": {"current": float("nan"), "error": "FRED down"},
        "HY_OAS": _r(3.5),
    }
    df = compute_macro_alignment(readings)
    # T10Y2Y rules contribute nothing; only HY OAS < 4 fires.
    assert df.loc["XLF", "tailwinds"] == 1
    assert df.loc["XLF", "headwinds"] == 0


def test_ratio_when_only_headwinds():
    readings = {"VIX": _r(35.0), "HY_OAS": _r(6.0)}
    df = compute_macro_alignment(readings)
    # UFO has VIX>25 headwind + HY_OAS>5 headwind.
    assert df.loc["UFO", "tailwinds"] == 0
    assert df.loc["UFO", "headwinds"] == 2
    assert df.loc["UFO", "ratio"] == pytest.approx(0.0)


def test_detail_records_label_and_verdict():
    readings = {"HY_OAS": _r(3.0)}
    df = compute_macro_alignment(readings)
    detail = df.loc["XLF", "detail"]
    assert detail == [("HY OAS < 4% (credit benign)", "tailwind")]


def test_copper_gold_rule_reads_zscore_not_level():
    """Copper/Gold rules key off z_score_1y (the level ~0.0014 is uninformative).

    A high z fires the pro-growth tailwind for the cyclicals XLI and XLB; the
    raw level must NOT trip a level-based predicate.
    """
    readings = {"COPPER_GOLD": {"current": 0.0014, "z_score_1y": 0.9}}
    df = compute_macro_alignment(readings)
    assert df.loc["XLI", "tailwinds"] == 1
    assert df.loc["XLB", "tailwinds"] == 1
    assert ("Copper/Gold z > +0.5 (pro-growth)", "tailwind") in df.loc["XLI", "detail"]


def test_copper_gold_deflationary_zscore_is_headwind():
    readings = {"COPPER_GOLD": {"current": 0.0009, "z_score_1y": -1.2}}
    df = compute_macro_alignment(readings)
    assert df.loc["XLI", "headwinds"] == 1
    assert df.loc["XLB", "headwinds"] == 1


def test_missing_zscore_field_skips_rule():
    """A copper/gold payload with no z_score_1y contributes nothing."""
    readings = {"COPPER_GOLD": {"current": 0.0014}}  # no z_score_1y
    df = compute_macro_alignment(readings)
    assert df.loc["XLI", "tailwinds"] == 0
    assert df.loc["XLI", "headwinds"] == 0


def test_optional_field_defaults_to_current():
    """Rules without a 5th element still read `current`."""
    readings = {"UST10": _r(4.5)}  # XLF: 10Y > 4 -> tailwind
    df = compute_macro_alignment(readings)
    assert df.loc["XLF", "tailwinds"] == 1
