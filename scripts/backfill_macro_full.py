"""Backfill the FULL macro feature set the sector allocator needs.

Your macro->sector rules (src/macro_alignment.py) key off NINE indicators:
  FRED:      HY_OAS, REAL_10Y, T10Y2Y, UST10, BREAKEVEN_5Y5Y
  yfinance:  VIX (^VIX), DXY (DX-Y.NYB), and the COPPER_GOLD / GOLD_OIL ratios
             (from HG=F, GC=F, CL=F)

Plus a credit proxy: HY_OAS only returns ~2023+ from FRED's CSV endpoint, so
we also pull BAA10Y (Moody's Baa - 10Y, 1986+) and store it as HY_OAS_PROXY.
The allocator can fall back to the proxy (rescaled) before 2023.

Output: data/macro_history_full.csv — one business-day row, columns above,
forward-filled (a macro level persists until the next print; that's causal —
today's reading is the last published value).

Run it yourself (network is reliable from your shell, flaky from the agent):
    ! PYTHONPATH=. python3 scripts/backfill_macro_full.py

It prints a coverage table and writes /tmp/macro_full.log with the same.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

FRED = {
    "HY_OAS": "BAMLH0A0HYM2",
    "REAL_10Y": "DFII10",
    "T10Y2Y": "T10Y2Y",
    "UST10": "DGS10",
    "BREAKEVEN_5Y5Y": "T5YIFR",
    "HY_OAS_PROXY": "BAA10Y",   # 1986+ credit-stress proxy for pre-2023 HY_OAS
}

# yfinance tickers -> logical name (ratios computed after).
YF = {
    "^VIX": "VIX",
    "DX-Y.NYB": "DXY",
    "GC=F": "GOLD",
    "CL=F": "OIL",
    "HG=F": "COPPER",
}


def fetch_fred(series_id: str) -> pd.Series:
    import requests
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, headers={"User-Agent": "sector-rotation/1.0"},
                        timeout=45)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    s = pd.Series(
        pd.to_numeric(df[df.columns[1]], errors="coerce").values,
        index=pd.to_datetime(df[df.columns[0]]),
    ).dropna()
    return s


def fetch_yf(ticker: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("empty")
    close = df["Close"]
    if hasattr(close, "columns"):       # MultiIndex single-ticker frame
        close = close.iloc[:, 0]
    s = close.dropna()
    s.index = pd.to_datetime(s.index)
    return s


def main() -> int:
    cols: dict[str, pd.Series] = {}
    log: list[str] = []

    for name, sid in FRED.items():
        try:
            s = fetch_fred(sid)
            cols[name] = s
            log.append(f"  FRED {name:<16} {sid:<14} {len(s):>6} rows  "
                       f"{s.index[0].date()} -> {s.index[-1].date()}")
        except Exception as e:  # noqa: BLE001
            log.append(f"  FRED {name:<16} {sid:<14} FAILED: {e!r}")

    for tkr, name in YF.items():
        try:
            s = fetch_yf(tkr)
            cols[name] = s
            log.append(f"  YF   {name:<16} {tkr:<14} {len(s):>6} rows  "
                       f"{s.index[0].date()} -> {s.index[-1].date()}")
        except Exception as e:  # noqa: BLE001
            log.append(f"  YF   {name:<16} {tkr:<14} FAILED: {e!r}")

    if not cols:
        print("FAILED: nothing fetched")
        return 1

    raw = pd.DataFrame(cols).sort_index()
    bidx = pd.bdate_range(raw.index.min(), raw.index.max())
    macro = raw.reindex(bidx).ffill()

    # Derived ratios the rules use (computed post-ffill so both legs are aligned).
    if {"COPPER", "GOLD"}.issubset(macro.columns):
        macro["COPPER_GOLD"] = macro["COPPER"] / macro["GOLD"]
    if {"GOLD", "OIL"}.issubset(macro.columns):
        macro["GOLD_OIL"] = macro["GOLD"] / macro["OIL"]
    macro.index.name = "date"

    out = ROOT / "data" / "macro_history_full.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(out)

    # Coverage at the backtest start, so we can see which legs are usable.
    probe = macro.loc[:"2019-01-18"]
    cov = []
    if len(probe):
        row = probe.iloc[-1]
        for c in macro.columns:
            v = row[c]
            cov.append(f"    {c:<16} {'NA' if pd.isna(v) else f'{float(v):.3f}'}")

    summary = (f"Wrote {out}: {macro.shape[0]} rows x {macro.shape[1]} cols\n"
               f"Range: {macro.index[0].date()} -> {macro.index[-1].date()}\n"
               + "\n".join(log)
               + "\n  --- values at 2019-01-18 (allocator start) ---\n"
               + "\n".join(cov) + "\n")
    Path("/tmp/macro_full.log").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
