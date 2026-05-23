"""Tests for src.tiger_client.compute_drift_by_sector enhancements.

Covers Agent DRIFT deliverables:
  * signals=  → state column joined
  * sma200_by_sector= → stop_at column joined
  * prices_by_sector= → current_price column joined
  * missing entries graceful (NaN, no crash)
  * urgency sort (applied in UI layer) produces SELL → REDUCE → BUY/HOLD order

The underlying utility keeps a stable sort-by-trade_value return; the
urgency re-sort lives in app.py. This test replicates that re-sort to
guard against regressions in the column data the UI consumes.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from config.settings import SECTOR_ETFS, SUPPLEMENTARY_SECTORS
from src.tiger_client import AccountSnapshot, compute_drift_by_sector


def _snapshot(positions: dict[str, float], nlv: float = 100_000.0) -> AccountSnapshot:
    """Build an AccountSnapshot from a {symbol: market_value} dict."""
    rows = [
        {"symbol": sym, "quantity": 1.0, "avg_cost": 1.0,
         "market_value": mv, "weight": mv / nlv}
        for sym, mv in positions.items()
    ]
    df = (pd.DataFrame(rows).set_index("symbol") if rows
          else pd.DataFrame(columns=["quantity", "avg_cost",
                                     "market_value", "weight"]).rename_axis("symbol"))
    return AccountSnapshot(net_liquidation=nlv, cash=nlv - sum(positions.values()),
                           positions=df)


def _main_sectors() -> list[str]:
    return [s for s in SECTOR_ETFS if s not in SUPPLEMENTARY_SECTORS]


def _equal_targets(active: list[str]) -> pd.Series:
    """Build an equal-weight target Series on the given sectors."""
    if not active:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(active), index=active, name="target_weight")


# ---------------------------------------------------------------------------
# 1. Baseline backwards-compat: omitting all new kwargs preserves columns.
# ---------------------------------------------------------------------------
def test_baseline_no_kwargs_keeps_old_columns():
    snap = _snapshot({"XLK": 10_000.0})
    targets = _equal_targets(["XLK"])
    df = compute_drift_by_sector(snap, targets)
    expected = {"target_weight", "current_weight", "drift",
                "target_value", "current_value", "trade_value"}
    assert expected.issubset(df.columns)
    # No state / stop_at / current_price unless requested.
    assert "state" not in df.columns
    assert "stop_at" not in df.columns
    assert "current_price" not in df.columns


# ---------------------------------------------------------------------------
# 2. signals= joins the state column keyed by sector ticker.
# ---------------------------------------------------------------------------
def test_signals_join_adds_state_column():
    sectors = _main_sectors()
    snap = _snapshot({s: 1000.0 for s in sectors})
    targets = _equal_targets(sectors)

    # Synthetic refined-signals frame: every sector gets a known state.
    states = ["SELL", "REDUCE", "NEW_BUY", "HOLD_IF_LONG", "CHASE", "HOLD"]
    state_col = [states[i % len(states)] for i in range(len(sectors))]
    sig = pd.DataFrame({"state": state_col}, index=sectors)

    df = compute_drift_by_sector(snap, targets, signals=sig)
    assert "state" in df.columns
    # Each sector should have its mapped state (we don't guarantee row order).
    for s in sectors:
        assert df.loc[s, "state"] == sig.loc[s, "state"]


def test_signals_join_handles_missing_sector_gracefully():
    """Sectors absent from `signals` should get a placeholder, not crash."""
    sectors = _main_sectors()
    snap = _snapshot({s: 1000.0 for s in sectors})
    targets = _equal_targets(sectors)
    # Only provide state for the first 2 sectors.
    partial = sectors[:2]
    sig = pd.DataFrame({"state": ["SELL", "NEW_BUY"]}, index=partial)
    df = compute_drift_by_sector(snap, targets, signals=sig)
    assert df.loc[partial[0], "state"] == "SELL"
    # Missing entries fall back to the placeholder "—".
    missing = [s for s in sectors if s not in partial][0]
    assert df.loc[missing, "state"] == "—"


# ---------------------------------------------------------------------------
# 3. sma200_by_sector= joins the stop_at column.
# ---------------------------------------------------------------------------
def test_sma200_join_adds_stop_at_column():
    snap = _snapshot({"XLK": 10_000.0, "XLF": 5_000.0})
    targets = _equal_targets(["XLK", "XLF"])
    sma = {"XLK": 200.0, "XLF": 50.0}
    df = compute_drift_by_sector(snap, targets, sma200_by_sector=sma)
    assert df.loc["XLK", "stop_at"] == pytest.approx(200.0)
    assert df.loc["XLF", "stop_at"] == pytest.approx(50.0)


def test_sma200_missing_entries_become_nan():
    sectors = _main_sectors()
    snap = _snapshot({s: 1000.0 for s in sectors})
    targets = _equal_targets(sectors)
    sma = {sectors[0]: 100.0}  # only one
    df = compute_drift_by_sector(snap, targets, sma200_by_sector=sma)
    assert df.loc[sectors[0], "stop_at"] == pytest.approx(100.0)
    # All others should be NaN — the UI renders these as "—".
    for s in sectors[1:]:
        assert math.isnan(df.loc[s, "stop_at"])


def test_prices_by_sector_join_adds_current_price():
    snap = _snapshot({"XLK": 10_000.0})
    targets = _equal_targets(["XLK"])
    df = compute_drift_by_sector(
        snap, targets,
        sma200_by_sector={"XLK": 200.0},
        prices_by_sector={"XLK": 220.0},
    )
    assert df.loc["XLK", "current_price"] == pytest.approx(220.0)
    assert df.loc["XLK", "stop_at"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# 4. Underlying frame keeps stable trade_value sort; urgency sort is UI-side.
# ---------------------------------------------------------------------------
def test_underlying_sort_is_trade_value_desc():
    """compute_drift_by_sector must preserve its sector-keyed stable
    return — urgency re-sort happens in the UI."""
    sectors = _main_sectors()
    # Hold positions only in the first two sectors so trade_value varies.
    snap = _snapshot({sectors[0]: 50_000.0, sectors[1]: 0.0})
    targets = _equal_targets(sectors)
    df = compute_drift_by_sector(snap, targets)
    # trade_value should be monotone non-increasing (descending).
    tv = df["trade_value"].tolist()
    assert tv == sorted(tv, reverse=True)


def test_urgency_resort_pattern():
    """Replicate the urgency re-sort the UI performs and verify ordering."""
    sectors = _main_sectors()
    snap = _snapshot({s: 1000.0 for s in sectors})
    targets = _equal_targets(sectors)
    # Mix of states; pick the first 4 sectors deterministically.
    states = {
        sectors[0]: "HOLD",
        sectors[1]: "SELL",
        sectors[2]: "REDUCE",
        sectors[3]: "NEW_BUY",
    }
    # Default the rest to HOLD.
    full = {s: states.get(s, "HOLD") for s in sectors}
    sig = pd.DataFrame({"state": list(full.values())}, index=list(full.keys()))
    df = compute_drift_by_sector(snap, targets, signals=sig)

    # Apply the same urgency sort the UI uses.
    urgency_rank = {"SELL": 0, "REDUCE": 1}
    df_sorted = df.assign(
        _u=df["state"].map(lambda s: urgency_rank.get(s, 2)),
        _a=df["trade_value"].abs(),
    ).sort_values(["_u", "_a"], ascending=[True, False])

    # The first row must be the SELL sector, second the REDUCE sector.
    ordered_states = df_sorted["state"].tolist()
    assert ordered_states[0] == "SELL"
    assert ordered_states[1] == "REDUCE"
    # Every remaining row must be neither SELL nor REDUCE.
    assert all(s not in ("SELL", "REDUCE") for s in ordered_states[2:])
