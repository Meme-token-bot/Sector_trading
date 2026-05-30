"""Regenerate BACKTEST_REPORT.md from a fresh backtest run.

Usage:
    PYTHONPATH=. python3 scripts/run_backtest_report.py
    PYTHONPATH=. python3 scripts/run_backtest_report.py --cost-bps 3

Why this exists: the report must NEVER drift from what the backtest actually
produced. Every quantitative claim in BACKTEST_REPORT.md is interpolated
from the result of this run.
"""
from __future__ import annotations

import argparse
import sys

from src.backtest import save_equity_csv
from src.backtest_report import build_headline_report, write_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cost-bps", type=float, default=5.0,
                   help="Per-side cost in bps (default 5).")
    p.add_argument("--slippage-bps", type=float, default=0.0,
                   help="Per-side slippage in bps (default 0).")
    p.add_argument("--execution", choices=["next_open", "same_close"],
                   default="next_open", help="Fill convention.")
    p.add_argument("--branch", default="feat/history-expandable-and-signal-runner",
                   help="Branch name to stamp in the report header.")
    args = p.parse_args()

    print(f"Running backtest (cost={args.cost_bps:.1f}bps each side, "
          f"slippage={args.slippage_bps:.1f}bps, execution={args.execution})…")
    report = build_headline_report(
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        execution=args.execution,
    )

    s = report.ed_result.stats
    print(f"  window:        {s['window_start']} → {s['window_end']}")
    print(f"  strategy CAGR: {s['strategy']['cagr']*100:+.2f}%")
    print(f"  SPY CAGR:      {s['spy']['cagr']*100:+.2f}%")
    print(f"  excess CAGR:   {s['excess_cagr']*100:+.2f}%")
    print(f"  trades:        {s['n_trades']:,}")
    print(f"  ann turnover:  {s['annualised_turnover']:.2f}x")

    path = write_report(report, branch=args.branch)
    save_equity_csv(report.ed_result, "backtest_equity")
    print(f"\nWrote: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
