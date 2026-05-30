"""Regime-conditional backtest + drawdown attribution.

Usage:
    PYTHONPATH=. python3 scripts/run_regime_analysis.py
    PYTHONPATH=. python3 scripts/run_regime_analysis.py --min-dd 0.03

Procedure:
  1. Run the headline event-driven backtest over the full available history.
  2. Classify every trading day as BULL / CORRECTION / BEAR from SPY's
     drawdown-from-rolling-high.
  3. Compound the strategy AND SPY returns within each regime; report
     up-capture, down-capture, and per-regime MDD.
  4. Identify every SPY drawdown > min_dd_pct in the window. For each:
     show the strategy's drawdown over the same window, which sectors
     it was holding at the peak vs the trough, and what it rotated into
     during the drawdown.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.backtest import BacktestConfig, load_price_panel, run_backtest
from src.regime_analysis import (
    classify_regimes,
    drawdown_attribution,
    identify_drawdowns,
    regime_conditional_stats,
    regime_episodes,
)


def _pct(x: float, p: int = 2, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "—"
    fmt = f"{{:{'+' if signed else ''}.{p}f}}%"
    return fmt.format(x * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--min-dd", type=float, default=0.05,
                    help="Minimum SPY drawdown (decimal) to report.")
    ap.add_argument("--min-days", type=int, default=5,
                    help="Minimum peak-to-trough days to report.")
    args = ap.parse_args()

    print(f"Running headline backtest (cost={args.cost_bps:.1f}bps each side)…")
    closes, opens = load_price_panel()
    result = run_backtest(BacktestConfig(cost_bps=args.cost_bps),
                          closes=closes, opens=opens)
    s = result.stats
    print(f"  window: {s['window_start']} → {s['window_end']}")
    print(f"  strategy CAGR: {_pct(s['strategy']['cagr'])}  "
          f"SPY CAGR: {_pct(s['spy']['cagr'])}  "
          f"excess: {_pct(s['excess_cagr'])}")
    print()

    spy_close = closes["SPY"].dropna()
    regimes = classify_regimes(spy_close.reindex(result.equity.index, method="ffill"))
    print("=== Regime distribution ===")
    counts = regimes.value_counts().reindex(["BULL", "CORRECTION", "BEAR"],
                                              fill_value=0)
    total = int(counts.sum())
    for r, n in counts.items():
        pct = n / total * 100 if total else 0
        print(f"  {r:11s}  {n:4d} days  ({pct:5.1f}% of window)")
    print()

    print("=== Regime episodes (consecutive same-regime runs) ===")
    eps = regime_episodes(regimes)
    if not eps.empty:
        # Only show non-BULL episodes — bull is the steady state, the
        # interesting ones are CORRECTION/BEAR.
        non_bull = eps[eps["regime"] != "BULL"]
        if non_bull.empty:
            print("  (no CORRECTION or BEAR episodes in window)")
        else:
            for _, ep in non_bull.iterrows():
                print(f"  {ep['regime']:11s}  "
                      f"{ep['start'].date()} → {ep['end'].date()}  "
                      f"({ep['n_days']:3d} days)")
    print()

    print("=== Per-regime stats (strategy vs SPY) ===")
    stats_df = regime_conditional_stats(result.equity, result.benchmark_equity,
                                          regimes)
    if stats_df.empty:
        print("  (no regime data)")
    else:
        # Pretty-print the key columns.
        print(f"  {'regime':12s} {'days':>5s} {'strat_cum':>10s} "
              f"{'spy_cum':>10s} {'excess':>9s} {'cap_up':>8s} {'cap_dn':>8s} "
              f"{'strat_mdd':>10s} {'spy_mdd':>9s}")
        for r, row in stats_df.iterrows():
            print(f"  {r:12s} {int(row['n_days']):5d} "
                  f"{_pct(row['strategy_cum']):>10s} "
                  f"{_pct(row['spy_cum']):>10s} "
                  f"{_pct(row['excess_cum']):>9s} "
                  f"{row['capture_up']:>+8.2f} "
                  f"{row['capture_down']:>+8.2f} "
                  f"{_pct(row['strategy_mdd_in_regime']):>10s} "
                  f"{_pct(row['spy_mdd_in_regime']):>9s}")
    print()

    print(f"=== SPY drawdowns ≥ {args.min_dd*100:.0f}% in window ===")
    dd_rows = drawdown_attribution(result, result.benchmark_equity, spy_close,
                                     min_dd_pct=args.min_dd,
                                     min_days=args.min_days)
    if not dd_rows:
        print("  (no qualifying drawdowns)")
    else:
        for i, r in enumerate(dd_rows, 1):
            print(f"\n  Drawdown #{i}: {r['peak_date']} → {r['trough_date']}"
                  + (f" → recovered {r['recovery_date']}"
                     if r['recovery_date'] else "  (not yet recovered)"))
            print(f"    SPY:      {_pct(r['spy_drawdown'])}  "
                  f"({r['days_to_trough']} days peak→trough"
                  + (f", {r['days_to_recover']} days trough→recover)"
                     if r['days_to_recover'] is not None else ")"))
            print(f"    Strategy: {_pct(r['strategy_drawdown'])}  "
                  f"(excess vs SPY {_pct(r['excess_drawdown'])} — "
                  f"{'LOST LESS' if r['excess_drawdown'] > 0 else 'LOST MORE'})")
            print(f"    Held at peak:    "
                  f"{', '.join(r['held_at_peak']) or '(none)'}")
            print(f"    Held at trough:  "
                  f"{', '.join(r['held_at_trough']) or '(none)'}")
            if r['rotated_in_during_dd']:
                print(f"    Rotated INTO during DD:  "
                      f"{', '.join(r['rotated_in_during_dd'])}")
            if r['rotated_out_during_dd']:
                print(f"    Rotated OUT during DD:   "
                      f"{', '.join(r['rotated_out_during_dd'])}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
