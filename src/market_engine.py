"""Module B: market data + quantitative variables."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import BENCHMARK, MACRO_TICKERS, PARAMS, SECTOR_ETFS


def fetch_prices(tickers: list[str], lookback_days: int = 400) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        tickers, start=start, end=end + timedelta(days=1),
        auto_adjust=True, progress=False, group_by="column",
    )
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            df = df.xs("Close", axis=1, level=0)
    else:
        df = df[["Close"]].rename(columns={"Close": tickers[0]})
    return df.dropna(how="all").dropna(axis=1, how="all")


def compute_sector_metrics(prices: pd.DataFrame,
                           as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute per-sector metrics. If `as_of` is given, only price data
    on or before that date is used — for historical signal replay.
    """
    if BENCHMARK not in prices.columns:
        raise ValueError(f"price frame must include benchmark {BENCHMARK}")

    if as_of is not None:
        prices = prices.loc[:pd.Timestamp(as_of)]

    sma_window = PARAMS.sma_window
    mom_window = PARAMS.momentum_window

    if len(prices) < mom_window + 1:
        return pd.DataFrame()

    spy_ret = prices[BENCHMARK].iloc[-1] / prices[BENCHMARK].iloc[-mom_window - 1] - 1

    rows: list[dict] = []
    for tkr in SECTOR_ETFS:
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        if len(s) < sma_window + 1:
            continue
        price = float(s.iloc[-1])
        sma = float(s.rolling(sma_window).mean().iloc[-1])
        ret_3m = float(s.iloc[-1] / s.iloc[-mom_window - 1] - 1)
        rows.append({
            "ticker": tkr,
            "name": SECTOR_ETFS[tkr],
            "price": price,
            "sma200": sma,
            "above_sma": price > sma,
            "extension_pct": (price - sma) / sma if sma else 0.0,
            "return_3m": ret_3m,
            "spy_return_3m": float(spy_ret),
            "relative_strength_3m": ret_3m - float(spy_ret),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")
    df["rs_rank"] = df["relative_strength_3m"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("relative_strength_3m", ascending=False)


def gold_oil_ratio(macro_prices: pd.DataFrame) -> dict:
    gold = macro_prices[MACRO_TICKERS["GOLD"]].dropna()
    oil = macro_prices[MACRO_TICKERS["OIL"]].dropna()
    common = gold.index.intersection(oil.index)
    if common.empty:
        return {"current": np.nan, "z_score_1y": np.nan}
    ratio = gold.loc[common] / oil.loc[common]
    z = (ratio.iloc[-1] - ratio.mean()) / ratio.std(ddof=0)
    return {"current": float(ratio.iloc[-1]),
            "z_score_1y": float(z),
            "series": ratio}


def yield_curve_spread() -> dict:
    """Fetch FRED T10Y2Y directly via CSV (avoids pandas-datareader compatibility breakage)."""
    try:
        import io
        import requests
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
        resp = requests.get(url, headers={"User-Agent": "sector-rotation/1.0"}, timeout=30)
        resp.raise_for_status()
        s = pd.read_csv(io.StringIO(resp.text), parse_dates=["observation_date"])
        s = s.rename(columns={"observation_date": "date", "T10Y2Y": "value"})
        s["value"] = pd.to_numeric(s["value"], errors="coerce")
        s = s.dropna().set_index("date")["value"]
        cutoff = pd.Timestamp(date.today() - timedelta(days=120))
        s = s[s.index >= cutoff]
        if s.empty:
            return {"current": np.nan, "slope_30d": np.nan}
        slope = (s.iloc[-1] - s.iloc[-30]) / 30 if len(s) >= 30 else np.nan
        return {"current": float(s.iloc[-1]),
                "slope_30d": float(slope),
                "series": s}
    except Exception as e:
        return {"current": np.nan, "slope_30d": np.nan, "error": str(e)}


def fetch_macro_prices() -> pd.DataFrame:
    return fetch_prices(list(MACRO_TICKERS.values()), lookback_days=400)
