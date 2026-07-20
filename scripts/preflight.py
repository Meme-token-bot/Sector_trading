#!/usr/bin/env python
"""CLI: operational pre-flight check.

Run before each Monday's trading session (or whenever you want a one-shot
"is everything wired and ready?" answer). Verifies:

  1. Data freshness  — prices.db updated to last trading day, sentiment.db
                       ingestion current, signal_snapshots accumulating.
  2. Model state     — current per-sector states, conviction, regime.
  3. Tiger live link — connection works, NLV/cash readable, current
                       positions vs model targets (with rotation cash-need).
  4. Trades for Mon  — explicit BUY/SELL list with dollar amounts.

The actual check LOGIC lives in `src/preflight_checks.py` — pure functions,
no printing, no IO beyond the read-only queries each check needs. This
script is now just a terminal formatter over those same functions
(TRADING_EDGE_AUDIT.md item C1): the Dashboard's Pre-Trade Checklist panel
calls the identical functions, so the CLI and the app can't silently
answer "are we ready" differently.

Outputs human-readable text. Exit code 0 = ready; 1 = blocker found.

Usage:
    PYTHONPATH=. python3 scripts/preflight.py
    PYTHONPATH=. python3 scripts/preflight.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preflight_checks import (  # noqa: E402
    check_data_freshness, check_model_state, check_tiger,
    list_monday_orders, overall_verdict,
)

_GLYPH = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}
_COLOR = {"OK": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}
_RESET = "\033[0m"


def _print_rows(rows: list[dict]) -> None:
    for r in rows:
        glyph = _GLYPH.get(r["status"], "?")
        color = _COLOR.get(r["status"], "")
        print(f"  {color}{glyph}{_RESET} {r['label']:35s} {r['detail']}")


def _section(title: str) -> None:
    print(f"\n{title}")
    print("─" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON summary at the end.")
    args = ap.parse_args()

    print("=" * 70)
    print(f"PRE-FLIGHT CHECK   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_rows: list[dict] = []

    _section("1. Data freshness")
    freshness_rows = check_data_freshness()
    _print_rows(freshness_rows)
    all_rows += freshness_rows

    _section("2. Model state — current signals")
    model_rows, model_state = check_model_state()
    _print_rows(model_rows)
    signals = model_state.get("signals")
    if signals is not None and not signals.empty:
        print(f"\n  Per-sector states:")
        glyphs = {"NEW_BUY": "🟢", "HOLD_IF_LONG": "🟡", "CHASE": "🟠",
                 "REDUCE": "🟤", "WATCH": "🔭", "HOLD": "⚪", "SELL": "🔴"}
        for tkr, row in signals.iterrows():
            st = row.get("state", "")
            glyph = glyphs.get(st, "·")
            conv = int(row.get("conviction", 0)) if "conviction" in signals.columns else 0
            sent = row.get("sentiment_score", 0.0) if "sentiment_score" in signals.columns else 0.0
            rs = row.get("relative_strength_3m", 0.0) if "relative_strength_3m" in signals.columns else 0.0
            print(f"    {glyph} {tkr:5s} {st:12s}  "
                  f"conv={conv} sent={sent:+.1f} rs3m={(rs or 0)*100:+.1f}%")
    all_rows += model_rows

    _section("3. Tiger live link")
    tiger_rows = check_tiger(model_state)
    _print_rows(tiger_rows)
    all_rows += tiger_rows

    _section("4. Monday actions — what to enter in Tiger")
    order_summary, orders = list_monday_orders(model_state)
    _print_rows(order_summary)
    if orders:
        print()
        for o in orders:
            line = f"{o['emoji']} {o['action']:6s} {o['ticker']:5s}"
            if o["detail"]:
                line += f"  {o['detail']}"
            print(f"    {line}")
    all_rows += order_summary

    print()
    print("─" * 70)
    verdict = overall_verdict(all_rows)
    if verdict["verdict"] == "not_ready":
        line = f"\033[31m✗ NOT READY\033[0m — {verdict['n_fail']} blocker(s), {verdict['n_warn']} warning(s)"
        rc = 1
    elif verdict["verdict"] == "ready_with_warnings":
        line = f"\033[33m⚠ READY WITH WARNINGS\033[0m — {verdict['n_warn']} item(s) to review"
        rc = 0
    else:
        line = f"\033[32m✓ READY\033[0m — {verdict['n_ok']} checks passed"
        rc = 0
    print(f"  {line}")

    if args.json:
        print()
        print(json.dumps({
            "verdict": verdict["verdict"],
            "n_ok": verdict["n_ok"], "n_warn": verdict["n_warn"],
            "n_fail": verdict["n_fail"],
            "checks": all_rows,
            "orders": orders,
        }, indent=2))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
