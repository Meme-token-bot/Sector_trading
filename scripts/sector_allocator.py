"""Cross-sectional sector allocator — weight ALL sectors continuously.

Instead of discrete BUY/HOLD/SELL rotation, this scores every SPDR sector each
rebalance and distributes 100% of the book across them by a blended score. Two
blend modes (run both, compare):

  MODE A  "tech-anchored + macro tilt":
      base_i = softmax(z_technical_i / temp)        # price drives the book
      w_i    = base_i * (1 + k * macro_net_i)       # macro nudges each name
      renormalize. Macro adjusts, never dominates (mirrors the live overlay).

  MODE B  "equal z-score blend":
      score_i = w_t * z_technical_i + w_m * z_macro_i
      w_i     = softmax(score_i / temp)             # macro & tech co-equal

Technical score per sector (causal, data <= t): z-scored avg of
  trend = price/SMA200 - 1, rs_3m = 3m return - SPY 3m, mom_6m = 6m return.

Macro score: reuses src.macro_alignment.compute_macro_alignment (YOUR sector
rules) on readings reconstructed as-of t from data/macro_merged.csv
(current + z_score_1y per indicator). HY_OAS is absent pre-2023 so its rules
simply don't fire then (documented gap; VIX + yield curve proxy most of it).

Long-only, fully invested across the 11 sectors, monthly rebalance, 5bps/side.
No look-ahead. Writes results to /tmp/allocator_results.txt and stdout.
Run:  PYTHONPATH=. python3 scripts/sector_allocator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import BENCHMARK, SECTOR_ETFS, SUPPLEMENTARY_SECTORS  # noqa: E402
from src.price_store import load_ohlcv_multi  # noqa: E402
from src.macro_alignment import compute_macro_alignment  # noqa: E402

INITIAL = 10_000.0
COST_BPS = 5.0
START = pd.Timestamp("2019-01-18")
MACRO_CSV = ROOT / "data" / "macro_merged.csv"


def load_data():
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    wide = load_ohlcv_multi(universe + [BENCHMARK], "1d")
    close = wide.xs("close", axis=1, level=1).sort_index()
    close.index = pd.to_datetime(close.index)
    macro = pd.read_csv(MACRO_CSV, parse_dates=["date"], index_col="date")
    return close, universe, macro


def _z(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 1e-12 else x * 0.0


def technical_z(close, universe, t) -> pd.Series:
    hist = close.loc[:t]
    if len(hist) < 200:
        return pd.Series(0.0, index=universe)
    trend = (hist.iloc[-1] / hist.tail(200).mean() - 1.0)[universe]
    if len(hist) > 63:
        spy3 = hist[BENCHMARK].iloc[-1] / hist[BENCHMARK].iloc[-63] - 1.0
        rs3 = pd.Series({s: (hist[s].iloc[-1] / hist[s].iloc[-63] - 1.0) - spy3
                         for s in universe})
    else:
        rs3 = pd.Series(0.0, index=universe)
    if len(hist) > 126:
        mom6 = pd.Series({s: hist[s].iloc[-1] / hist[s].iloc[-126] - 1.0
                          for s in universe})
    else:
        mom6 = pd.Series(0.0, index=universe)
    return (_z(trend) + _z(rs3) + _z(mom6)) / 3.0


def _readings_asof(macro, t):
    h = macro.loc[:t]
    if h.empty:
        return {}
    out = {}
    for col in macro.columns:
        s = h[col].dropna()
        if s.empty:
            continue
        cur = float(s.iloc[-1])
        win = s.tail(252)
        sd = win.std(ddof=0)
        z = float((cur - win.mean()) / sd) if sd > 1e-12 else float("nan")
        out[col] = {"current": cur, "z_score_1y": z}
    return out


def macro_net(macro, universe, t) -> pd.Series:
    readings = _readings_asof(macro, t)
    if not readings:
        return pd.Series(0.0, index=universe)
    align = compute_macro_alignment(readings)
    return (align["tailwinds"] - align["headwinds"]).reindex(universe).fillna(0.0).astype(float)


def softmax(s: pd.Series, temp: float) -> pd.Series:
    x = (s / temp).clip(-50, 50)
    e = np.exp(x - x.max())
    return e / e.sum()


def weights_at(close, universe, macro, t, cfg) -> pd.Series:
    mode, wt, wm, k, temp = cfg
    if mode == "EW":
        return pd.Series(1.0 / len(universe), index=universe)
    tz = technical_z(close, universe, t)
    if mode == "A":
        base = softmax(tz, temp)
        w = (base * (1.0 + k * macro_net(macro, universe, t))).clip(lower=0.0)
        return w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(universe), index=universe)
    if mode == "B":
        mz = _z(macro_net(macro, universe, t))
        return softmax(wt * tz + wm * mz, temp)
    raise ValueError(mode)


def _reb_dates(idx, freq="M"):
    s = pd.Series(idx, index=idx)
    return list(pd.DatetimeIndex(s.groupby(s.dt.to_period(freq)).last().values))


def run(close, universe, macro, cfg, start, end, freq="M") -> dict:
    win = close.loc[start:end]
    if len(win) < 30:
        return {"final": float("nan"), "cagr": float("nan"), "mdd": float("nan"), "turn": float("nan")}
    idx = win.index
    daily = close[universe].pct_change()
    rebs = [d for d in _reb_dates(idx, freq) if idx[0] <= d <= idx[-1]]
    equity = INITIAL
    curve = {}
    cur_w = weights_at(close, universe, macro, idx[0], cfg)
    ni = 0
    turn_tot = 0.0
    for d in idx[1:]:
        equity *= (1.0 + float((cur_w * daily.loc[d].reindex(universe).fillna(0)).sum()))
        if ni < len(rebs) and d == rebs[ni]:
            nw = weights_at(close, universe, macro, d, cfg)
            turn = float((nw - cur_w).abs().sum()) / 2.0
            equity *= (1.0 - turn * COST_BPS / 10_000.0)
            turn_tot += turn
            cur_w = nw
            ni += 1
        curve[d] = equity
    cv = pd.Series(curve)
    yrs = (cv.index[-1] - cv.index[0]).days / 365.25
    cagr = (cv.iloc[-1] / INITIAL) ** (1 / yrs) - 1.0 if yrs > 0 else float("nan")
    mdd = float((cv / cv.cummax() - 1.0).min())
    return {"final": float(cv.iloc[-1]), "cagr": float(cagr), "mdd": mdd,
            "turn": turn_tot / yrs if yrs > 0 else float("nan"), "curve": cv}


def spy_stats(close, start, end) -> dict:
    s = close[BENCHMARK].loc[start:end].dropna()
    cv = s / s.iloc[0] * INITIAL
    yrs = (cv.index[-1] - cv.index[0]).days / 365.25
    return {"final": float(cv.iloc[-1]), "cagr": float((cv.iloc[-1] / INITIAL) ** (1 / yrs) - 1.0),
            "mdd": float((cv / cv.cummax() - 1.0).min()), "turn": 0.0}


CONFIGS = {
    "equal-weight 11":     ("EW", 0, 0, 0, 1.0),
    "A tech-only":         ("A", 1, 0, 0.0, 0.5),
    "A tech+macro k0.5":   ("A", 1, 1, 0.5, 0.5),
    "A tech+macro k1.0":   ("A", 1, 1, 1.0, 0.5),
    "B 70/30 tech/macro":  ("B", 0.7, 0.3, 0, 0.5),
    "B 50/50 tech/macro":  ("B", 0.5, 0.5, 0, 0.5),
}


def main() -> int:
    close, universe, macro = load_data()
    end = close.index.max()
    spy = spy_stats(close, START, end)
    L = [f"Window {START.date()} -> {end.date()}  (${INITIAL:,.0f}, {COST_BPS:.0f}bps/side, monthly)", "",
         f"{'Strategy':<22}{'Final $':>11}{'CAGR':>8}{'MaxDD':>8}{'Turn/yr':>9}{'vs SPY':>10}",
         "-" * 68,
         f"{'100% SPY':<22}{spy['final']:>11,.0f}{spy['cagr']*100:>7.1f}%{spy['mdd']*100:>7.1f}%{0:>8.1f}x{0:>10,.0f}"]
    for name, cfg in CONFIGS.items():
        r = run(close, universe, macro, cfg, START, end, "M")
        L.append(f"{name:<22}{r['final']:>11,.0f}{r['cagr']*100:>7.1f}%"
                 f"{r['mdd']*100:>7.1f}%{r['turn']:>8.1f}x{r['final']-spy['final']:>10,.0f}")
    out = "\n".join(L) + "\n"
    Path("/tmp/allocator_results.txt").write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
