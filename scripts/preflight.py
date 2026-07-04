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

Outputs human-readable text. Exit code 0 = ready; 1 = blocker found.

Usage:
    PYTHONPATH=. python3 scripts/preflight.py
    PYTHONPATH=. python3 scripts/preflight.py --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (  # noqa: E402
    BENCHMARK, DB_PATH, PARAMS, SECTOR_ETFS, SUPPLEMENTARY_SECTORS,
    gmail_configured, tiger_configured,
)
from src.price_store import PRICES_DB_PATH  # noqa: E402


# ---------------------------------------------------------------------------
# Status helpers — uniform OK / WARN / FAIL output
# ---------------------------------------------------------------------------

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
_GLYPH = {OK: "✓", WARN: "⚠", FAIL: "✗"}
_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
_RESET = "\033[0m"


def _line(status: str, label: str, detail: str = "") -> dict:
    glyph = _GLYPH.get(status, "?")
    color = _COLOR.get(status, "")
    print(f"  {color}{glyph}{_RESET} {label:35s} {detail}")
    return {"status": status, "label": label, "detail": detail}


def _section(title: str) -> None:
    print(f"\n{title}")
    print("─" * 70)


# ---------------------------------------------------------------------------
# Check 1: data freshness
# ---------------------------------------------------------------------------

def check_data_freshness() -> list[dict]:
    _section("1. Data freshness")
    rows: list[dict] = []

    today = date.today()
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    universe.append(BENCHMARK)

    con = sqlite3.connect(PRICES_DB_PATH)
    cur = con.cursor()
    last_bars = {}
    for t in universe:
        row = cur.execute(
            "SELECT MAX(bar_date) FROM ohlcv WHERE ticker=? AND timeframe='1d'",
            (t,)).fetchone()
        last_bars[t] = row[0] if row and row[0] else None
    con.close()

    if any(b is None for b in last_bars.values()):
        missing = [t for t, b in last_bars.items() if b is None]
        rows.append(_line(FAIL, "prices.db universe coverage",
                           f"missing: {', '.join(missing)}"))
    else:
        oldest_ticker, oldest_bar = min(last_bars.items(),
                                         key=lambda kv: kv[1])
        days_stale = (today - date.fromisoformat(oldest_bar)).days
        # Allow up to 3 calendar days (Friday → Monday + weekends).
        status = OK if days_stale <= 3 else (WARN if days_stale <= 7 else FAIL)
        rows.append(_line(status, "prices.db freshness",
                           f"oldest last bar: {oldest_ticker} @ {oldest_bar} "
                           f"({days_stale}d stale)"))

    # sentiment.db
    con = sqlite3.connect(DB_PATH)
    n_news = con.execute("SELECT COUNT(*) FROM newsletters").fetchone()[0]
    last_news = con.execute(
        "SELECT MAX(publication_date) FROM newsletters").fetchone()[0]
    n_snaps = con.execute(
        "SELECT COUNT(*) FROM signal_snapshots").fetchone()[0]
    last_snap = con.execute(
        "SELECT MAX(as_of) FROM signal_snapshots").fetchone()[0]
    con.close()

    if last_news:
        days_stale = (today - date.fromisoformat(last_news)).days
        status = OK if days_stale <= 3 else (WARN if days_stale <= 7 else FAIL)
        rows.append(_line(status, "sentiment.db newsletters",
                           f"{n_news} rows, last {last_news} ({days_stale}d stale)"))
    else:
        rows.append(_line(WARN, "sentiment.db newsletters", "(empty)"))

    if last_snap:
        days_stale = (today - date.fromisoformat(last_snap)).days
        status = OK if days_stale <= 7 else WARN
        rows.append(_line(status, "signal_snapshots",
                           f"{n_snaps} rows, last {last_snap} ({days_stale}d stale)"))
    else:
        rows.append(_line(WARN, "signal_snapshots",
                           "(empty — dashboard hasn't written one yet)"))

    return rows


# ---------------------------------------------------------------------------
# Check 2: model state — current regime, signals, target weights
# ---------------------------------------------------------------------------

def check_model_state() -> tuple[list[dict], dict]:
    _section("2. Model state — current signals")
    rows: list[dict] = []

    try:
        from src.backtest import load_price_panel
        from src.db import aggregate_sentiment
        from src.market_engine import compute_sector_metrics
        from src.regime_analysis import classify_regimes
        from src.signal_history import build_signal_history
        from src.signals import build_signals, refine_signals, target_weights
        closes, _ = load_price_panel(tickers=list(SECTOR_ETFS.keys()) + [BENCHMARK])
        spy_close = closes[BENCHMARK].dropna()
        regimes = classify_regimes(spy_close)
        current_regime = str(regimes.iloc[-1])

        metrics = compute_sector_metrics(closes)
        sentiment = aggregate_sentiment(as_of=date.today())
        raw = build_signals(metrics, sentiment)
        history = build_signal_history(closes, end=date.today())
        signals = refine_signals(raw, history)  # no macro (live macro fetch is slow)
        tgt = target_weights(signals)
    except Exception as e:  # noqa: BLE001
        rows.append(_line(FAIL, "model pipeline",
                           f"failed: {type(e).__name__}: {e}"))
        return rows, {"current_regime": "—", "signals": pd.DataFrame(),
                       "targets": pd.Series(dtype=float)}

    regime_color = {"BULL": OK, "CORRECTION": WARN, "BEAR": WARN}.get(
        current_regime, WARN)
    rows.append(_line(regime_color, "current regime",
                       f"{current_regime} (since "
                       f"{(regimes != current_regime)[::-1].idxmax().date()})"))

    state_counts = signals["state"].value_counts().to_dict()
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(state_counts.items()))
    rows.append(_line(OK, "state distribution", counts_str))
    print(f"\n  Per-sector states:")
    for tkr, row in signals.iterrows():
        st = row["state"]
        glyph = {"NEW_BUY": "🟢", "HOLD_IF_LONG": "🟡",
                 "CHASE": "🟠", "REDUCE": "🟤",
                 "WATCH": "🔭", "HOLD": "⚪",
                 "SELL": "🔴"}.get(st, "·")
        conv = int(row.get("conviction", 0))
        sent = row.get("sentiment_score", 0.0)
        rs = row.get("relative_strength_3m", 0.0) or 0.0
        print(f"    {glyph} {tkr:5s} {st:12s}  "
              f"conv={conv} sent={sent:+.1f} rs3m={rs*100:+.1f}%")

    return rows, {
        "current_regime": current_regime,
        "signals": signals,
        "targets": tgt,
    }


# ---------------------------------------------------------------------------
# Check 3: Tiger link + portfolio
# ---------------------------------------------------------------------------

def check_tiger(model_state: dict) -> list[dict]:
    _section("3. Tiger live link")
    rows: list[dict] = []

    if not tiger_configured():
        rows.append(_line(WARN, "Tiger configured",
                           "(.env missing TIGER_ID / TIGER_ACCOUNT / "
                           "TIGER_PRIVATE_KEY_PATH — manual execution only)"))
        return rows

    try:
        from src.tiger_client import (
            compute_drift_by_sector, fetch_account_snapshot,
        )
        snap = fetch_account_snapshot()
    except Exception as e:  # noqa: BLE001
        rows.append(_line(FAIL, "Tiger connection",
                           f"failed: {type(e).__name__}: {e}"))
        return rows

    nlv = float(snap.net_liquidation or 0)
    cash = float(snap.cash or 0)
    cash_pct = cash / nlv * 100 if nlv else 0

    rows.append(_line(OK, "Tiger connection", f"NLV ${nlv:,.0f}"))
    rows.append(_line(OK, "Cash on hand",
                       f"${cash:,.0f} ({cash_pct:.1f}% of NLV)"))

    targets = model_state.get("targets", pd.Series(dtype=float))
    if targets.empty:
        rows.append(_line(WARN, "Drift vs model targets",
                           "(no targets — model has nothing in BUY-class)"))
        return rows

    try:
        drift_df = compute_drift_by_sector(snap, targets,
                                            signals=model_state.get("signals"))
    except Exception as e:  # noqa: BLE001
        rows.append(_line(WARN, "Drift computation",
                           f"failed: {type(e).__name__}: {e}"))
        return rows

    # Cash needed to move from current → target — sum of positive trade_value.
    if "trade_value" in drift_df.columns:
        need_cash = float(drift_df.loc[drift_df["trade_value"] > 0,
                                         "trade_value"].sum())
        free_cash = float(drift_df.loc[drift_df["trade_value"] < 0,
                                         "trade_value"].sum() * -1)
        cash_ok = cash + free_cash >= need_cash * 0.95
        status = OK if cash_ok else WARN
        rows.append(_line(status, "Cash coverage for rotation",
                           f"need ${need_cash:,.0f}, free from SELLs "
                           f"${free_cash:,.0f}, on hand ${cash:,.0f}"))

    # Per-sector drift table
    print(f"\n  Drift by sector (top 8 by |trade_value|):")
    if "trade_value" in drift_df.columns:
        top = drift_df.reindex(drift_df["trade_value"].abs()
                                 .sort_values(ascending=False).index).head(8)
        for tkr, r in top.iterrows():
            tv = float(r.get("trade_value", 0))
            cur_w = float(r.get("current_weight", 0)) * 100
            tgt_w = float(r.get("target_weight", 0)) * 100
            arrow = "→" if tv > 0 else "←" if tv < 0 else "="
            action = ("BUY" if tv > 0 else "SELL" if tv < 0 else "—")
            print(f"    {tkr:5s} {action:4s} ${abs(tv):>8,.0f}  "
                  f"({cur_w:5.1f}% {arrow} {tgt_w:5.1f}%)")
    return rows


# ---------------------------------------------------------------------------
# Check 4: Monday orders (just the deltas that need action)
# ---------------------------------------------------------------------------

def list_monday_orders(model_state: dict) -> list[dict]:
    _section("4. Monday actions — what to enter in Tiger")
    rows: list[dict] = []
    signals = model_state.get("signals")
    targets = model_state.get("targets", pd.Series(dtype=float))

    if signals is None or signals.empty:
        rows.append(_line(WARN, "Orders", "(no model output)"))
        return rows

    orders: list[str] = []
    for tkr, row in signals.iterrows():
        state = row.get("state", "")
        if state == "NEW_BUY":
            w = float(targets.get(tkr, 0))
            orders.append(f"🟢 BUY  {tkr:5s}  target weight {w*100:.1f}%")
        elif state == "CHASE" and PARAMS.chase_weight_fraction > 0:
            w = float(targets.get(tkr, 0))
            orders.append(f"🟠 CHASE {tkr:5s} target weight {w*100:.1f}% "
                          f"({PARAMS.chase_weight_fraction*100:.0f}% partial)")
        elif state in ("SELL", "REDUCE"):
            orders.append(f"🔴 {state:6s} {tkr:5s}")

    if not orders:
        rows.append(_line(OK, "Orders", "no action — portfolio aligned with model"))
    else:
        rows.append(_line(OK, "Orders", f"{len(orders)} action(s)"))
        print()
        for o in orders:
            print(f"    {o}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    all_rows += check_data_freshness()
    model_rows, model_state = check_model_state()
    all_rows += model_rows
    all_rows += check_tiger(model_state)
    all_rows += list_monday_orders(model_state)

    # Summary footer.
    print()
    print("─" * 70)
    n_fail = sum(1 for r in all_rows if r["status"] == FAIL)
    n_warn = sum(1 for r in all_rows if r["status"] == WARN)
    n_ok = sum(1 for r in all_rows if r["status"] == OK)
    if n_fail:
        verdict = f"\033[31m✗ NOT READY\033[0m — {n_fail} blocker(s), {n_warn} warning(s)"
        rc = 1
    elif n_warn:
        verdict = f"\033[33m⚠ READY WITH WARNINGS\033[0m — {n_warn} item(s) to review"
        rc = 0
    else:
        verdict = f"\033[32m✓ READY\033[0m — {n_ok} checks passed"
        rc = 0
    print(f"  {verdict}")

    if args.json:
        print()
        print(json.dumps({
            "verdict": "ready" if rc == 0 else "not_ready",
            "n_ok": n_ok, "n_warn": n_warn, "n_fail": n_fail,
            "checks": all_rows,
        }, indent=2))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
