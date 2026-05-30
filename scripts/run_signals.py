"""Ad-hoc: replicate the Dashboard signal pipeline outside Streamlit.

Mirrors app.py's tab_dashboard wiring: fetch_prices -> compute_sector_metrics
-> build_signals -> build_signal_history -> refine_signals -> target_weights.

Prices: tries the live yfinance fetch (what the dashboard uses); on any
failure falls back to the local prices.db cache so we still produce signals
offline. Macro overlay (FRED/yfinance) degrades to None if unreachable.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd

from config.settings import BENCHMARK, SECTOR_ETFS
from src.db import aggregate_sentiment, save_signal_snapshot
from src.market_engine import compute_sector_metrics, fetch_prices
from src.price_store import load_ohlcv
from src.signal_history import build_signal_history
from src.signals import build_signals, refine_signals, target_weights

TICKERS = list(SECTOR_ETFS.keys()) + [BENCHMARK]


def get_prices() -> tuple[pd.DataFrame, str]:
    try:
        df = fetch_prices(TICKERS)
        if not df.empty and BENCHMARK in df.columns and len(df) > 60:
            return df, "live yfinance"
        raise RuntimeError("live fetch returned too little data")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] live fetch failed ({e!r}); using prices.db cache", file=sys.stderr)
        start = date.today() - timedelta(days=400)
        cols = {}
        for t in TICKERS:
            s = load_ohlcv(t, "1d", start=start)
            if not s.empty:
                cols[t] = s["close"]
        return pd.DataFrame(cols), "local cache (prices.db)"


def get_macro():
    try:
        from src.macro_alignment import compute_macro_alignment
        from src.market_engine import (
            copper_gold_ratio, dxy_level, fetch_fred_indicators,
            fetch_macro_prices, gold_oil_ratio, vix_level, yield_curve_spread,
        )
        mp = fetch_macro_prices()
        fred = fetch_fred_indicators()
        bundle = {
            "T10Y2Y": yield_curve_spread(), "HY_OAS": fred.get("HY_OAS", {}),
            "UST10": fred.get("UST10", {}), "REAL_10Y": fred.get("REAL_10Y", {}),
            "BREAKEVEN_5Y5Y": fred.get("BREAKEVEN_5Y5Y", {}),
            "DXY": dxy_level(mp), "VIX": vix_level(mp),
            "GOLD_OIL": gold_oil_ratio(mp), "COPPER_GOLD": copper_gold_ratio(mp),
        }
        return compute_macro_alignment(bundle), "live"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] macro overlay unavailable ({e!r}); refining without it", file=sys.stderr)
        return None, "unavailable"


def main() -> None:
    today = date.today()
    prices, psrc = get_prices()
    last_bar = prices.index.max()
    print(f"# price source: {psrc} | last bar: {last_bar.date()} | tickers: {len(prices.columns)}")

    metrics = compute_sector_metrics(prices)
    sentiment = aggregate_sentiment(as_of=today)
    raw = build_signals(metrics, sentiment)
    history = build_signal_history(prices, end=today)
    macro, msrc = get_macro()
    print(f"# macro overlay: {msrc} | sentiment rows: {len(sentiment)}")
    signals = refine_signals(raw, history, macro_alignment=macro)
    targets = target_weights(signals)
    # Persist the snapshot so forward-perf tracking can read the exact
    # refined state we emitted today (rather than re-replaying the model).
    try:
        n = save_signal_snapshot(today, signals, macro_alignment=macro)
        print(f"# wrote {n} rows to signal_snapshots for {today.isoformat()}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] signal_snapshots write failed: {e!r}", file=sys.stderr)

    cols = ["name", "state", "signal", "above_sma", "relative_strength_3m",
            "rs_rank", "extension_pct", "sentiment_score", "n_obs",
            "consecutive_buy_weeks", "conviction"]
    cols = [c for c in cols if c in signals.columns]
    view = signals[cols].copy()

    # Macro pill ("tw/hw" or "—"): the alignment frame lives separately
    # from `signals`, so refine_signals leaves these counts off the main
    # frame. Surface them here so HOLD/SELL/CHASE rows aren't silent about
    # macro the way they are in the human reasons string.
    if macro is not None and not macro.empty:
        def _pill(tkr: str) -> str:
            if tkr not in macro.index:
                return "—"
            tw = int(macro.loc[tkr, "tailwinds"] or 0)
            hw = int(macro.loc[tkr, "headwinds"] or 0)
            return f"{tw}/{hw}" if (tw + hw) else "—"
        view.insert(2, "macro_t/h", [_pill(t) for t in view.index])

    order = {"NEW_BUY": 0, "HOLD_IF_LONG": 1, "CHASE": 2, "WATCH": 3,
             "HOLD": 4, "REDUCE": 5, "SELL": 6}
    if "state" in view.columns:
        view = view.sort_values("state", key=lambda s: s.map(order).fillna(9))

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== PER-SECTOR SIGNALS ===")
    print(view.to_string())

    print("\n=== STATE COUNTS ===")
    if "state" in signals.columns:
        print(signals["state"].value_counts().to_string())

    print("\n=== TARGET WEIGHTS (equal-weight across BUY-class) ===")
    print(targets.to_string() if len(targets) else "(none — no NEW_BUY/HOLD_IF_LONG sectors)")

    print("\n=== REASONS ===")
    rc = "state_reason" if "state_reason" in signals.columns else "reasons"
    for tkr, row in signals.iterrows():
        st = row.get("state", row.get("signal"))
        print(f"  {tkr:5} [{st}] {row.get(rc, '')}")

    if macro is not None and not macro.empty:
        print("\n=== MACRO ALIGNMENT (which rules fired per sector) ===")
        for tkr, row in macro.iterrows():
            detail = row.get("detail") or []
            if not detail:
                print(f"  {tkr:5}  (no rules fired)")
                continue
            tag = {"tailwind": "+", "headwind": "-", "neutral": "·"}
            print(f"  {tkr:5}  T{int(row['tailwinds'])}/H{int(row['headwinds'])}"
                  + (f"  N{int(row['neutral'])}" if int(row['neutral']) else ""))
            for label, verdict in detail:
                print(f"         {tag.get(verdict, '?')} {label}")


if __name__ == "__main__":
    main()
