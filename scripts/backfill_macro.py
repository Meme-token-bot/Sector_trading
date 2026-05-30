"""Backfill FULL-HISTORY FRED macro series into data/macro_history.csv.

The live model's `_fetch_fred_series` clips to a 400-day lookback (it only
needs current readings). To backtest a macro-aware sector allocator we need
the full history. This script pulls every series in config.settings.FRED_SERIES
from its inception, forward-fills to a daily business-day grid (FRED series
publish on mixed calendars), and caches the result.

Honest note: this enables HISTORICAL macro backtesting — the one signal leg
(besides technical) that actually has a usable past. Newsletter sentiment still
has only ~3 months and cannot be backtested.

Reproduce:
    PYTHONPATH=. python3 scripts/backfill_macro.py
    PYTHONPATH=. python3 scripts/backfill_macro.py --out data/macro_history.csv
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config.settings import FRED_SERIES  # noqa: E402


def fetch_full(series_id: str) -> pd.Series:
    """Full-history FRED series (no lookback clip). Robust to '.' sentinels
    and DATE/observation_date header casing."""
    import requests
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, headers={"User-Agent": "sector-rotation/1.0"},
                        timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
    ).dropna()
    s.name = series_id
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "macro_history.csv")
    args = ap.parse_args()

    cols = {}
    log = []
    for logical, series_id in FRED_SERIES.items():
        try:
            s = fetch_full(series_id)
            cols[logical] = s
            log.append(f"  {logical:<16} {series_id:<14} {len(s):>6} rows  "
                       f"{s.index[0].date()} -> {s.index[-1].date()}")
        except Exception as e:  # noqa: BLE001
            log.append(f"  {logical:<16} {series_id:<14} FAILED: {e!r}")

    if not cols:
        Path("/tmp/macro_backfill.log").write_text("no series fetched\n")
        print("FAILED: no series fetched")
        return 1

    # Union daily business-day index, forward-fill (macro levels persist between
    # prints — that's economically correct and causal: today's value is the last
    # published reading).
    raw = pd.DataFrame(cols).sort_index()
    bidx = pd.bdate_range(raw.index.min(), raw.index.max())
    macro = raw.reindex(bidx).ffill()
    macro.index.name = "date"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(args.out)

    summary = (f"Wrote {args.out} : {macro.shape[0]} rows x {macro.shape[1]} cols\n"
               f"Range: {macro.index[0].date()} -> {macro.index[-1].date()}\n"
               + "\n".join(log) + "\n")
    Path("/tmp/macro_backfill.log").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
