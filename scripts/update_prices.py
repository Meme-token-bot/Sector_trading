#!/usr/bin/env python
"""CLI: update the OHLCV cache for the signal + expression universe.

Idempotent. First run does a 5-year cold-start pull for each (ticker,
timeframe) pair; subsequent runs only refetch the trailing overlap window
and append new bars (plus split detection — see `price_store.update_ticker`).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/update_prices.py` from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.expressions import all_expression_tickers  # noqa: E402
from config.settings import BENCHMARK, SECTOR_ETFS  # noqa: E402
from src.price_store import update_all  # noqa: E402


def main() -> int:
    signal_tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    expression_tickers = all_expression_tickers()
    # dict.fromkeys: dedupe while preserving insertion order so the signal
    # universe stays at the front of the list.
    tickers = list(dict.fromkeys([*SECTOR_ETFS, BENCHMARK, *expression_tickers]))
    n_extra = len(tickers) - len(signal_tickers)

    print("=" * 70)
    print(f"Universe: {len(signal_tickers)} signal + {n_extra} expression = "
          f"total {len(tickers)} unique")
    print(f"Updating OHLCV cache for {len(tickers)} tickers × 2 timeframes")
    print("=" * 70)

    def _progress(ticker: str, timeframe: str, status: str) -> None:
        # status is replaced by the row count line printed below — keep this
        # callback as a no-op hook for the Streamlit progress bar.
        return None

    results = update_all(tickers=tickers, progress=_progress)

    for r in results:
        print(f"  {r['ticker']:<5} {r['timeframe']:<4} "
              f"{r['status']:<16} ({r['rows_written']} rows)  "
              f"{r['notes']}")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_split = sum(1 for r in results if r["status"] == "split_detected")
    n_err = sum(1 for r in results if r["status"] == "error")
    print("-" * 70)
    print(f"Summary: {n_ok} ok, {n_split} split-replaced, {n_err} errors")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
