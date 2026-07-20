"""Pure data-producing checks for the pre-trade readiness panel.

Extracted from scripts/preflight.py (TRADING_EDGE_AUDIT.md item C1) so the
SAME checks run whether you invoke the CLI (`scripts/preflight.py`, still
supported, still prints exactly as before) or open the Dashboard — a
second, silently-drifting reimplementation inside app.py was exactly the
failure mode this extraction avoids. No print(), no ANSI color codes here;
`scripts/preflight.py` formats these for the terminal and `app.py` renders
them as a Streamlit panel.

Each check function returns a list of dicts:
  {"status": "OK"|"WARN"|"FAIL", "label": str, "detail": str}
"""
from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd

from config.settings import (
    BENCHMARK, DB_PATH, PARAMS, SECTOR_ETFS, SUPPLEMENTARY_SECTORS,
    tiger_configured,
)
from src.price_store import PRICES_DB_PATH

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def _row(status: str, label: str, detail: str = "") -> dict:
    return {"status": status, "label": label, "detail": detail}


# ---------------------------------------------------------------------------
# 1. Data freshness
# ---------------------------------------------------------------------------

def check_data_freshness() -> list[dict]:
    rows: list[dict] = []
    today = date.today()
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    universe.append(BENCHMARK)

    con = sqlite3.connect(PRICES_DB_PATH)
    cur = con.cursor()
    last_bars: dict[str, str | None] = {}
    for t in universe:
        row = cur.execute(
            "SELECT MAX(bar_date) FROM ohlcv WHERE ticker=? AND timeframe='1d'",
            (t,)).fetchone()
        last_bars[t] = row[0] if row and row[0] else None
    con.close()

    if any(b is None for b in last_bars.values()):
        missing = [t for t, b in last_bars.items() if b is None]
        rows.append(_row(FAIL, "prices.db universe coverage",
                         f"missing: {', '.join(missing)}"))
    else:
        oldest_ticker, oldest_bar = min(last_bars.items(), key=lambda kv: kv[1])
        days_stale = (today - date.fromisoformat(oldest_bar)).days
        status = OK if days_stale <= 3 else (WARN if days_stale <= 7 else FAIL)
        rows.append(_row(status, "prices.db freshness",
                         f"oldest last bar: {oldest_ticker} @ {oldest_bar} "
                         f"({days_stale}d stale)"))

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
        rows.append(_row(status, "sentiment.db newsletters",
                         f"{n_news} rows, last {last_news} ({days_stale}d stale)"))
    else:
        rows.append(_row(WARN, "sentiment.db newsletters", "(empty)"))

    if last_snap:
        days_stale = (today - date.fromisoformat(last_snap)).days
        status = OK if days_stale <= 7 else WARN
        rows.append(_row(status, "signal_snapshots",
                         f"{n_snaps} rows, last {last_snap} ({days_stale}d stale)"))
    else:
        rows.append(_row(WARN, "signal_snapshots",
                         "(empty — dashboard hasn't written one yet)"))

    return rows


# ---------------------------------------------------------------------------
# 2. Model state
# ---------------------------------------------------------------------------

def check_model_state() -> tuple[list[dict], dict]:
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
        signals = refine_signals(raw, history)
        tgt = target_weights(signals)
    except Exception as e:  # noqa: BLE001
        rows.append(_row(FAIL, "model pipeline",
                         f"failed: {type(e).__name__}: {e}"))
        return rows, {"current_regime": "—", "signals": pd.DataFrame(),
                      "targets": pd.Series(dtype=float)}

    regime_color = {"BULL": OK, "CORRECTION": WARN, "BEAR": WARN}.get(
        current_regime, WARN)
    since = (regimes != current_regime)[::-1].idxmax()
    rows.append(_row(regime_color, "current regime",
                     f"{current_regime} (since {since.date()})"))

    state_counts = signals["state"].value_counts().to_dict()
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(state_counts.items()))
    rows.append(_row(OK, "state distribution", counts_str))

    return rows, {"current_regime": current_regime, "signals": signals,
                  "targets": tgt}


# ---------------------------------------------------------------------------
# 3. Tiger live link
# ---------------------------------------------------------------------------

def check_tiger(model_state: dict, snapshot=None) -> list[dict]:
    """`snapshot`, if provided, is used as-is instead of fetching fresh —
    lets a caller that already has a recent AccountSnapshot (e.g. the
    Dashboard's own Tiger drift panel, via its `_cached_tiger_snapshot()`)
    avoid a second Tiger API round-trip for the same data. The CLI's
    behavior (always fetch fresh) is unchanged when this is omitted."""
    rows: list[dict] = []

    if not tiger_configured():
        rows.append(_row(WARN, "Tiger configured",
                         "(.env missing TIGER_ID / TIGER_ACCOUNT / "
                         "TIGER_PRIVATE_KEY_PATH — manual execution only)"))
        return rows

    if snapshot is not None:
        snap = snapshot
    else:
        try:
            from src.tiger_client import fetch_account_snapshot
            snap = fetch_account_snapshot()
        except Exception as e:  # noqa: BLE001
            rows.append(_row(FAIL, "Tiger connection",
                             f"failed: {type(e).__name__}: {e}"))
            return rows

    try:
        from src.tiger_client import compute_drift_by_sector
    except Exception as e:  # noqa: BLE001
        rows.append(_row(FAIL, "Tiger connection",
                         f"failed: {type(e).__name__}: {e}"))
        return rows

    nlv = float(snap.net_liquidation or 0)
    cash = float(snap.cash or 0)
    cash_pct = cash / nlv * 100 if nlv else 0

    rows.append(_row(OK, "Tiger connection", f"NLV ${nlv:,.0f}"))
    rows.append(_row(OK, "Cash on hand",
                     f"${cash:,.0f} ({cash_pct:.1f}% of NLV)"))

    targets = model_state.get("targets", pd.Series(dtype=float))
    if targets.empty:
        rows.append(_row(WARN, "Drift vs model targets",
                         "(no targets — model has nothing in BUY-class)"))
        return rows

    try:
        drift_df = compute_drift_by_sector(snap, targets,
                                           signals=model_state.get("signals"))
    except Exception as e:  # noqa: BLE001
        rows.append(_row(WARN, "Drift computation",
                         f"failed: {type(e).__name__}: {e}"))
        return rows

    if "trade_value" in drift_df.columns:
        need_cash = float(drift_df.loc[drift_df["trade_value"] > 0,
                                       "trade_value"].sum())
        free_cash = float(drift_df.loc[drift_df["trade_value"] < 0,
                                       "trade_value"].sum() * -1)
        cash_ok = cash + free_cash >= need_cash * 0.95
        status = OK if cash_ok else WARN
        rows.append(_row(status, "Cash coverage for rotation",
                         f"need ${need_cash:,.0f}, free from SELLs "
                         f"${free_cash:,.0f}, on hand ${cash:,.0f}"))

    return rows


# ---------------------------------------------------------------------------
# 4. Monday orders
# ---------------------------------------------------------------------------

def list_monday_orders(model_state: dict) -> tuple[list[dict], list[dict]]:
    """Returns (summary_rows, orders). `orders` is the structured per-ticker
    list — separated from `summary_rows` so a UI can render both a status
    line AND the actual order list, which the original CLI-only version
    only ever printed, never returned structurally."""
    summary: list[dict] = []
    signals = model_state.get("signals")
    targets = model_state.get("targets", pd.Series(dtype=float))

    if signals is None or signals.empty:
        summary.append(_row(WARN, "Orders", "(no model output)"))
        return summary, []

    orders: list[dict] = []
    for tkr, row in signals.iterrows():
        state = row.get("state", "")
        if state == "NEW_BUY":
            w = float(targets.get(tkr, 0))
            orders.append({"emoji": "🟢", "action": "BUY", "ticker": tkr,
                          "detail": f"target weight {w*100:.1f}%"})
        elif state == "CHASE" and PARAMS.chase_weight_fraction > 0:
            w = float(targets.get(tkr, 0))
            orders.append({"emoji": "🟠", "action": "CHASE", "ticker": tkr,
                          "detail": f"target weight {w*100:.1f}% "
                                    f"({PARAMS.chase_weight_fraction*100:.0f}% partial)"})
        elif state in ("SELL", "REDUCE"):
            orders.append({"emoji": "🔴", "action": state, "ticker": tkr,
                          "detail": ""})

    if not orders:
        summary.append(_row(OK, "Orders", "no action — portfolio aligned with model"))
    else:
        summary.append(_row(OK, "Orders", f"{len(orders)} action(s)"))
    return summary, orders


# ---------------------------------------------------------------------------
# Overall verdict — shared by the CLI's exit code and the app's readiness badge
# ---------------------------------------------------------------------------

def overall_verdict(all_rows: list[dict]) -> dict:
    """Roll every check row up into one readiness verdict.

    Returns {"verdict": "ready"|"ready_with_warnings"|"not_ready",
             "n_ok": int, "n_warn": int, "n_fail": int}."""
    n_fail = sum(1 for r in all_rows if r["status"] == FAIL)
    n_warn = sum(1 for r in all_rows if r["status"] == WARN)
    n_ok = sum(1 for r in all_rows if r["status"] == OK)
    if n_fail:
        verdict = "not_ready"
    elif n_warn:
        verdict = "ready_with_warnings"
    else:
        verdict = "ready"
    return {"verdict": verdict, "n_ok": n_ok, "n_warn": n_warn, "n_fail": n_fail}
