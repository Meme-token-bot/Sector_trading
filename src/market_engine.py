"""Module B: market data + quantitative variables."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import BENCHMARK, FRED_SERIES, MACRO_TICKERS, PARAMS, SECTOR_ETFS


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


def copper_gold_ratio(macro_prices: pd.DataFrame) -> dict:
    copper = macro_prices[MACRO_TICKERS["COPPER"]].dropna()
    gold = macro_prices[MACRO_TICKERS["GOLD"]].dropna()
    common = copper.index.intersection(gold.index)
    if common.empty:
        return {"current": np.nan, "z_score_1y": np.nan}
    ratio = copper.loc[common] / gold.loc[common]
    window = ratio.tail(252)
    std = window.std(ddof=0)
    z = (ratio.iloc[-1] - window.mean()) / std if std else np.nan
    return {"current": float(ratio.iloc[-1]),
            "z_score_1y": float(z),
            "series": ratio}


def dxy_level(macro_prices: pd.DataFrame) -> dict:
    s = macro_prices[MACRO_TICKERS["DXY"]].dropna()
    if s.empty:
        return {"current": np.nan, "z_score_1y": np.nan}
    window = s.tail(252)
    std = window.std(ddof=0)
    z = (s.iloc[-1] - window.mean()) / std if std else np.nan
    return {"current": float(s.iloc[-1]),
            "z_score_1y": float(z),
            "series": s}


def vix_level(macro_prices: pd.DataFrame) -> dict:
    s = macro_prices[MACRO_TICKERS["VIX"]].dropna()
    if s.empty:
        return {"current": np.nan, "z_score_1y": np.nan}
    window = s.tail(252)
    std = window.std(ddof=0)
    z = (s.iloc[-1] - window.mean()) / std if std else np.nan
    return {"current": float(s.iloc[-1]),
            "z_score_1y": float(z),
            "series": s}


def _fetch_fred_series(series_id: str, lookback_days: int = 400) -> pd.Series:
    """Pull a FRED series via the public CSV endpoint.

    Returns a date-indexed float Series (NaNs dropped) trimmed to the trailing
    `lookback_days` calendar days. Raises on HTTP / parse failure — callers
    decide whether to surface or swallow.
    """
    import io
    import requests
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, headers={"User-Agent": "sector-rotation/1.0"}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", series_id: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"]
    cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))
    s = s[s.index >= cutoff]
    s.name = series_id
    return s


def yield_curve_spread() -> dict:
    """FRED T10Y2Y via direct CSV (avoids pandas-datareader compatibility breakage)."""
    try:
        s = _fetch_fred_series(FRED_SERIES["T10Y2Y"], lookback_days=120)
        if s.empty:
            return {"current": np.nan, "slope_30d": np.nan}
        slope = (s.iloc[-1] - s.iloc[-30]) / 30 if len(s) >= 30 else np.nan
        return {"current": float(s.iloc[-1]),
                "slope_30d": float(slope),
                "series": s}
    except Exception as e:
        return {"current": np.nan, "slope_30d": np.nan, "error": str(e)}


def fetch_fred_indicators() -> dict[str, dict]:
    """Pull the panel's non-curve FRED series. One bad series surfaces as
    {"current": nan, "error": ...} but does not abort the others."""
    # Each entry lists which derived fields are meaningful for that series.
    # HY OAS / 5Y5Y breakeven are level indicators where the 1y z-score is the
    # regime signal; nominal & real 10Y move on direction, so 30d slope matters.
    spec = {
        "HY_OAS":         ("z_score_1y",),
        "UST10":          ("slope_30d",),
        "REAL_10Y":       ("slope_30d",),
        "BREAKEVEN_5Y5Y": ("z_score_1y",),
    }
    out: dict[str, dict] = {}
    for key, fields in spec.items():
        try:
            s = _fetch_fred_series(FRED_SERIES[key], lookback_days=400)
            if s.empty:
                out[key] = {"current": np.nan, "error": "empty series"}
                continue
            entry: dict = {"current": float(s.iloc[-1]), "series": s}
            if "z_score_1y" in fields:
                window = s.tail(252)
                std = window.std(ddof=0)
                entry["z_score_1y"] = (float((s.iloc[-1] - window.mean()) / std)
                                       if std else np.nan)
            if "slope_30d" in fields:
                entry["slope_30d"] = (float((s.iloc[-1] - s.iloc[-30]) / 30)
                                      if len(s) >= 30 else np.nan)
            out[key] = entry
        except Exception as e:
            out[key] = {"current": np.nan, "error": str(e)}
    return out


def fetch_macro_prices() -> pd.DataFrame:
    return fetch_prices(list(MACRO_TICKERS.values()), lookback_days=400)


def fetch_ohlcv_yf(tickers: list[str], timeframe: str,
                   start: date, end: date | None = None) -> pd.DataFrame:
    """Pull OHLCV via yfinance and normalize to flat row format.

    Returns a DataFrame with columns:
        ticker, bar_date, open, high, low, close, volume

    `timeframe` maps directly onto yfinance's `interval` argument ('1d' or
    '1wk'). For weekly bars yfinance returns the week-START Monday as the
    index; we normalize to the trading-week's end (Friday) so the stored
    `bar_date` is the *as-of* date of the bar rather than its opening day.

    Uses `auto_adjust=True` and `group_by='ticker'` so the returned shape is
    predictable across the single-vs-multi-ticker boundary. yfinance returns
    flat columns when one ticker is passed and a MultiIndex when many are;
    both shapes are normalized here. Rows with all-NaN OHLC are dropped.
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "bar_date", "open",
                                     "high", "low", "close", "volume"])
    if timeframe not in ("1d", "1wk"):
        raise ValueError(f"unsupported timeframe: {timeframe!r}")

    end_date = end or date.today()
    raw = yf.download(
        tickers,
        start=start,
        end=end_date + timedelta(days=1),
        interval=timeframe,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["ticker", "bar_date", "open",
                                     "high", "low", "close", "volume"])

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        # group_by='ticker' => level 0 = ticker, level 1 = field.
        # Some yfinance versions flip the order; detect by inspecting level 0.
        lvl0 = set(raw.columns.get_level_values(0))
        ticker_first = bool(lvl0 & set(tickers))
        for tkr in tickers:
            try:
                sub = raw[tkr] if ticker_first else raw.xs(tkr, axis=1, level=1)
            except KeyError:
                continue
            frames.append(_normalize_single_ohlcv(sub, tkr))
    else:
        # Single-ticker flat columns.
        frames.append(_normalize_single_ohlcv(raw, tickers[0]))

    if not frames:
        return pd.DataFrame(columns=["ticker", "bar_date", "open",
                                     "high", "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    if timeframe == "1wk":
        # yfinance puts the week start in the index — shift to Friday so the
        # stored bar_date represents the week-ending session.
        out["bar_date"] = out["bar_date"] + pd.Timedelta(days=4)
    return out


def _normalize_single_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Turn a single-ticker yfinance frame into the flat row format."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "bar_date", "open",
                                     "high", "low", "close", "volume"])
    cols = {c.lower(): c for c in df.columns}
    needed = ["open", "high", "low", "close", "volume"]
    if not all(k in cols for k in needed):
        return pd.DataFrame(columns=["ticker", "bar_date", "open",
                                     "high", "low", "close", "volume"])
    sub = df[[cols[k] for k in needed]].copy()
    sub.columns = needed
    sub = sub.dropna(subset=["open", "high", "low", "close"], how="all")
    if sub.empty:
        return pd.DataFrame(columns=["ticker", "bar_date", "open",
                                     "high", "low", "close", "volume"])
    sub = sub.reset_index().rename(columns={sub.index.name or "index": "bar_date",
                                            "Date": "bar_date",
                                            "Datetime": "bar_date"})
    # Ensure bar_date column exists after reset_index regardless of original index name.
    if "bar_date" not in sub.columns:
        first_col = sub.columns[0]
        sub = sub.rename(columns={first_col: "bar_date"})
    sub["bar_date"] = pd.to_datetime(sub["bar_date"]).dt.tz_localize(None).dt.normalize()
    sub.insert(0, "ticker", ticker)
    return sub[["ticker", "bar_date", "open", "high", "low", "close", "volume"]]
