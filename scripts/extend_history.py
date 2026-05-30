#!/usr/bin/env python
"""CLI: extend `prices.db` history backward by wiping and re-fetching tickers.

The default `update_ticker` flow only fetches forward — it never extends
older history once a cold-start has happened. To get deeper-bear coverage
(2018 Q4, 2020 COVID), we have to wipe and re-pull with an explicit start
date.

Usage:
    PYTHONPATH=. python3 scripts/extend_history.py
    PYTHONPATH=. python3 scripts/extend_history.py --from 2018-01-01
    PYTHONPATH=. python3 scripts/extend_history.py --tickers SPY,XLK,XLE

Defaults to the backtest universe (SPY + 11 SPDR sectors). Thematic ETFs
are NOT extended by default — they're noise for the regime-evidence
question this script supports, and extending all ~80 cached tickers
multiplies runtime for no benefit.

WARNING: destructive on success — wipes 1d AND 1wk for each target ticker
before re-fetching. Existing rows for non-targeted tickers are untouched.
Side-effect: if Streamlit is running it'll briefly serve empty frames for
the affected tickers between WIPE and the first UPSERT — restart it after
the script finishes to be safe.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Allow direct invocation from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import BENCHMARK, SECTOR_ETFS, SUPPLEMENTARY_SECTORS  # noqa: E402
from src.market_engine import fetch_ohlcv_yf  # noqa: E402
from src.price_store import (  # noqa: E402
    _df_to_rows,
    last_bar_date,
    upsert_ohlcv,
    wipe_ticker,
)


def _default_universe() -> list[str]:
    """SPY + the 11 SPDR sectors (UFO + thematics excluded — the backtest
    doesn't trade them and extending them is wasted bandwidth)."""
    sectors = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    return [BENCHMARK] + sectors


def extend_ticker(ticker: str, start: date,
                  timeframes: tuple[str, ...] = ("1d", "1wk")) -> dict:
    """Wipe + re-fetch one ticker from `start` to today.

    Returns a per-timeframe result dict with row counts and the new range.
    """
    today = date.today()
    out: dict[str, dict] = {}
    for tf in timeframes:
        prior_min = last_bar_date(ticker, tf)  # for the diff report
        wiped = wipe_ticker(ticker, tf)
        try:
            df = fetch_ohlcv_yf([ticker], tf, start=start, end=today)
        except Exception as e:  # noqa: BLE001
            out[tf] = {"status": "fetch_error", "error": str(e),
                       "wiped_rows": wiped, "new_rows": 0}
            continue
        if df.empty:
            out[tf] = {"status": "empty_response",
                       "wiped_rows": wiped, "new_rows": 0}
            continue
        rows = _df_to_rows(df, ticker, tf)
        n = upsert_ohlcv(rows)
        new_min = min((r["bar_date"] for r in rows), default=None)
        new_max = max((r["bar_date"] for r in rows), default=None)
        out[tf] = {
            "status": "ok",
            "wiped_rows": wiped,
            "new_rows": n,
            "prior_min_date": str(prior_min) if prior_min else None,
            "new_min_date": str(new_min) if new_min else None,
            "new_max_date": str(new_max) if new_max else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--from", dest="start", default="2018-01-01",
        help="Cold-start date (YYYY-MM-DD). Default 2018-01-01 — captures "
             "the 2018-Q4 correction, COVID 2020 crash, and 2022 rate shock.",
    )
    ap.add_argument(
        "--tickers", default=None,
        help="Comma-separated ticker list. Default: SPY + 11 SPDR sectors.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit; do not wipe or refetch.",
    )
    args = ap.parse_args()

    try:
        start = datetime.fromisoformat(args.start).date()
    except ValueError:
        print(f"error: invalid --from date {args.start!r} (use YYYY-MM-DD)",
              file=sys.stderr)
        return 2

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _default_universe()

    print("=" * 70)
    print(f"Extending prices.db history for {len(tickers)} tickers")
    print(f"  Date range: {start} → {date.today()} "
          f"(~{(date.today() - start).days // 365} years)")
    print(f"  Tickers:    {', '.join(tickers)}")
    print(f"  Timeframes: 1d + 1wk")
    if args.dry_run:
        print("\n(dry-run — no changes)")
        return 0
    print("=" * 70)

    summary = {"ok": 0, "fetch_error": 0, "empty_response": 0}
    total_wiped = total_new = 0
    for t in tickers:
        res = extend_ticker(t, start)
        for tf, r in res.items():
            status = r["status"]
            summary[status] = summary.get(status, 0) + 1
            total_wiped += r.get("wiped_rows", 0)
            total_new += r.get("new_rows", 0)
            if status == "ok":
                print(f"  {t:5s} {tf:4s}  wiped={r['wiped_rows']:5d}  "
                      f"new={r['new_rows']:5d}  "
                      f"{r['prior_min_date']} → {r['new_min_date']}  "
                      f"(now ends {r['new_max_date']})")
            else:
                err = r.get("error", "(no rows returned)")
                print(f"  {t:5s} {tf:4s}  FAILED ({status}): {err}")

    print("-" * 70)
    print(f"Summary: {summary['ok']} ok, "
          f"{summary.get('fetch_error', 0)} fetch errors, "
          f"{summary.get('empty_response', 0)} empty responses")
    print(f"Total: wiped {total_wiped:,} rows, wrote {total_new:,} new rows")

    if summary.get("fetch_error", 0) + summary.get("empty_response", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
