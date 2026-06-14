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


def _yf_close(tickers: list[str] | str, lookback_days: int) -> pd.DataFrame:
    """Download Close prices from yfinance, always returning a plain DataFrame
    with ticker names as columns (no MultiIndex), NaN rows dropped, tz-naive index.
    """
    cutoff = date.today() - timedelta(days=lookback_days)
    raw = yf.download(
        tickers, start=cutoff, end=date.today() + timedelta(days=1),
        auto_adjust=True, progress=False,
    )
    if raw is None or raw.empty:
        cols = [tickers] if isinstance(tickers, str) else list(tickers)
        return pd.DataFrame(columns=cols)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"].copy()
    else:
        tkr = tickers if isinstance(tickers, str) else tickers[0]
        df = raw[["Close"]].rename(columns={"Close": tkr})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.dropna(how="all")


def _fetch_tnx_series(lookback_days: int = 400) -> pd.Series:
    """Pull 10Y nominal Treasury yield via yfinance (^TNX).

    yfinance stores ^TNX directly in percentage points (e.g. 4.35 = 4.35%).
    No scaling is applied.
    """
    df = _yf_close("^TNX", lookback_days)
    if df.empty or "^TNX" not in df.columns:
        return pd.Series(dtype=float, name="^TNX")
    return df["^TNX"].dropna().rename("^TNX")


def _fetch_yield_curve_yf(lookback_days: int = 400) -> pd.Series:
    """10Y–3M Treasury spread from yfinance (^TNX minus ^IRX).

    ^TNX = 10Y yield in %.  ^IRX = 13-week T-bill yield in %.
    The 10Y–3M spread is not identical to FRED's T10Y2Y (10Y–2Y) but carries
    the same recession-signal interpretation and is the best proxy available
    via yfinance.
    """
    df = _yf_close(["^TNX", "^IRX"], lookback_days)
    if df.empty or "^TNX" not in df.columns or "^IRX" not in df.columns:
        return pd.Series(dtype=float, name="T10Y3M_yf")
    spread = (df["^TNX"] - df["^IRX"]).dropna()
    spread.name = "T10Y3M_yf"
    return spread


def yield_curve_spread() -> dict:
    """10Y–2Y Treasury spread, preferring FRED T10Y2Y with yfinance (10Y–3M) fallback.

    FRED's T10Y2Y series is the canonical source. If it times out (which happens
    intermittently on DGS* / T10Y* series from some networks), we fall back to
    computing the spread from yfinance ^TNX and ^IRX. The 10Y–3M spread is not
    identical to 10Y–2Y but carries the same recession-signal interpretation.
    The source used is recorded in the returned dict as 'source'.
    """
    def _compute(s: pd.Series) -> dict:
        slope = (s.iloc[-1] - s.iloc[-30]) / 30 if len(s) >= 30 else np.nan
        return {"current": float(s.iloc[-1]), "slope_30d": float(slope), "series": s}

    # Primary: FRED T10Y2Y
    try:
        s = _fetch_fred_series(FRED_SERIES["T10Y2Y"], lookback_days=120)
        if not s.empty:
            return {**_compute(s), "source": "FRED T10Y2Y"}
    except Exception:
        pass

    # Fallback: yfinance 10Y–3M
    try:
        s = _fetch_yield_curve_yf(lookback_days=120)
        if not s.empty:
            return {**_compute(s), "source": "yfinance 10Y–3M"}
    except Exception as e:
        pass

    return {"current": np.nan, "slope_30d": np.nan, "error": "all sources failed"}


def fetch_fred_indicators() -> dict[str, dict]:
    """Pull the panel's non-curve FRED series. One bad series surfaces as
    {"current": nan, "error": ...} but does not abort the others.

    UST10 is sourced from yfinance ^TNX (FRED DGS10 times out on this network).
    HY_OAS, REAL_10Y, and BREAKEVEN_5Y5Y use FRED sequentially — parallel FRED
    connections triggered rate-limiting that broke DFII10 (REAL_10Y).
    BREAKEVEN_5Y5Y has no yfinance equivalent; it shows blank if FRED times out.
    """
    spec = {
        "HY_OAS":         ("z_score_1y",),
        "UST10":          ("slope_30d",),
        "REAL_10Y":       ("slope_30d",),
        "BREAKEVEN_5Y5Y": ("z_score_1y",),
    }

    def _compute_fields(s: pd.Series, fields: tuple) -> dict:
        entry: dict = {"current": float(s.iloc[-1]), "series": s}
        if "z_score_1y" in fields:
            window = s.tail(252)
            std = window.std(ddof=0)
            entry["z_score_1y"] = (float((s.iloc[-1] - window.mean()) / std)
                                   if std else np.nan)
        if "slope_30d" in fields:
            entry["slope_30d"] = (float((s.iloc[-1] - s.iloc[-30]) / 30)
                                  if len(s) >= 30 else np.nan)
        return entry

    def _from_fred(key: str, fields: tuple) -> tuple[str, dict]:
        """Fetch a single key from FRED CSV. Called sequentially for FRED series
        to avoid triggering rate-limits from parallel connections."""
        try:
            s = _fetch_fred_series(FRED_SERIES[key], lookback_days=400)
            if s.empty:
                return key, {"current": np.nan, "error": "empty series"}
            return key, _compute_fields(s, fields)
        except Exception as e:
            return key, {"current": np.nan, "error": str(e)}

    def _ust10_from_yf(fields: tuple) -> tuple[str, dict]:
        """UST10 via yfinance ^TNX (primary), FRED DGS10 (fallback)."""
        try:
            s = _fetch_tnx_series(lookback_days=400)
            if not s.empty:
                return "UST10", {**_compute_fields(s, fields), "source": "yfinance ^TNX"}
        except Exception:
            pass
        return _from_fred("UST10", fields)

    out: dict[str, dict] = {}

    # UST10 resolved via yfinance (fast, no FRED timeout risk).
    out["UST10"] = _ust10_from_yf(spec["UST10"])[1]

    # Remaining three series all use FRED. Fetch sequentially to avoid
    # hammering FRED with parallel connections, which caused DFII10 (REAL_10Y)
    # to regress from working to timing-out when parallelism was added.
    for key in ("HY_OAS", "REAL_10Y", "BREAKEVEN_5Y5Y"):
        _, result = _from_fred(key, spec[key])
        out[key] = result

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
