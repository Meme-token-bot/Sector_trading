"""Tests for src.regime_snapshot — synthetic series, no network/DB.

Mirrors the style of tests/test_regime_analysis.py and tests/test_macro_alignment.py:
pure-function, deterministic, no fixtures beyond hand-built frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime_snapshot import (
    DISPERSION_BANDS,
    compute_regime_and_breadth,
    dispersion_band,
)


def _bdays(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _metrics(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# compute_regime_and_breadth
# ---------------------------------------------------------------------------

def test_empty_inputs_are_handled_gracefully():
    out = compute_regime_and_breadth(pd.Series(dtype=float), pd.DataFrame())
    assert out["regime"] == "—"
    assert out["regime_days"] == 0
    assert out["n_core"] == 0
    assert out["pct_above_sma"] == 0.0
    assert np.isnan(out["rs_dispersion_pct"])


def test_flat_spy_is_bull_with_full_run_length():
    spy = pd.Series(100.0, index=_bdays("2024-01-01", 300))
    metrics = _metrics({
        "XLK": {"above_sma": True, "relative_strength_3m": 0.05},
        "XLF": {"above_sma": False, "relative_strength_3m": -0.02},
    })
    out = compute_regime_and_breadth(spy, metrics)
    assert out["regime"] == "BULL"
    assert out["regime_days"] == 300


def test_regime_run_length_counts_only_the_current_run():
    # 250 bull days, then a sharp drop that should register as CORRECTION or
    # BEAR for the trailing days only.
    idx = _bdays("2024-01-01", 260)
    vals = [100.0] * 250 + list(np.linspace(100.0, 78.0, 10))  # -22% tail
    spy = pd.Series(vals, index=idx)
    metrics = _metrics({"XLK": {"above_sma": True, "relative_strength_3m": 0.0}})
    out = compute_regime_and_breadth(spy, metrics)
    # The tail should have flipped the label, and the run length must be
    # shorter than the full 260-day window (i.e. it isn't counting the old
    # BULL days that preceded the flip).
    assert out["regime"] in ("CORRECTION", "BEAR")
    assert 0 < out["regime_days"] < 260


def test_breadth_excludes_supplementary_sectors():
    spy = pd.Series(100.0, index=_bdays("2024-01-01", 300))
    metrics = _metrics({
        "XLK": {"above_sma": True,  "relative_strength_3m": 0.05},
        "XLF": {"above_sma": True,  "relative_strength_3m": 0.03},
        "XLU": {"above_sma": False, "relative_strength_3m": -0.01},
        # UFO is supplementary — must not count toward n_core / n_above_sma,
        # matching the RS-rank convention in src/signals.py::build_signals.
        "UFO": {"above_sma": True,  "relative_strength_3m": 0.40},
    })
    out = compute_regime_and_breadth(spy, metrics)
    assert out["n_core"] == 3
    assert out["n_above_sma"] == 2
    assert out["pct_above_sma"] == pytest.approx(2 / 3)


def test_rs_dispersion_and_mean_computed_on_core_only():
    spy = pd.Series(100.0, index=_bdays("2024-01-01", 300))
    metrics = _metrics({
        "XLK": {"above_sma": True, "relative_strength_3m": 0.10},   # +10%
        "XLF": {"above_sma": True, "relative_strength_3m": -0.10},  # -10%
        "UFO": {"above_sma": True, "relative_strength_3m": 5.00},   # excluded
    })
    out = compute_regime_and_breadth(spy, metrics)
    # mean of +10%, -10% -> 0%; population stdev of [+10,-10] in pct points -> 10.0
    assert out["rs_mean_pct"] == pytest.approx(0.0, abs=1e-9)
    assert out["rs_dispersion_pct"] == pytest.approx(10.0, rel=1e-6)


def test_missing_columns_return_nan_dispersion_not_a_crash():
    spy = pd.Series(100.0, index=_bdays("2024-01-01", 300))
    metrics = _metrics({"XLK": {"above_sma": True}})  # no relative_strength_3m
    out = compute_regime_and_breadth(spy, metrics)
    assert np.isnan(out["rs_dispersion_pct"])
    assert np.isnan(out["rs_mean_pct"])
    assert out["n_above_sma"] == 1


# ---------------------------------------------------------------------------
# dispersion_band
# ---------------------------------------------------------------------------

def test_dispersion_band_low():
    emoji, label = dispersion_band(1.0)
    assert emoji == "🔴"
    assert "Low" in label


def test_dispersion_band_moderate():
    emoji, label = dispersion_band(3.5)
    assert emoji == "🟡"


def test_dispersion_band_high():
    emoji, label = dispersion_band(8.0)
    assert emoji == "🟢"


def test_dispersion_band_nan_is_neutral():
    emoji, label = dispersion_band(float("nan"))
    assert emoji == "⚪"
    assert label == "—"


def test_dispersion_bands_are_monotonically_increasing_thresholds():
    thresholds = [b[0] for b in DISPERSION_BANDS]
    assert thresholds == sorted(thresholds)
