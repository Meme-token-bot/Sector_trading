"""Walk-forward validation of the 'tilt, don't rotate' strategy.

The full-window tilt backtest found ONE config that beat SPY (50% core / top-2
sectors / 252d momentum / quarterly, +$2,695 on $10k). That could easily be a
hindsight artifact — the config that happened to lean into tech hardest. This
script tests it honestly:

  For each walk-forward fold:
    1. On the TRAIN window, score every candidate tilt config by CAGR.
    2. Pick the train-window winner.
    3. Score THAT config OUT OF SAMPLE on the test window.
    4. Compare its OOS return to SPY over the same test window.

If the train-picked config reliably beats SPY out-of-sample, the edge is real.
If it's a coin-flip (or the "robust" config is just always-the-same tech tilt),
we say so.

Reuses the fold schedule from src.walk_forward so it matches the rotation
strategy's validation exactly. No look-ahead: momentum is causal, and train and
test windows never overlap.

Reproduce:
    PYTHONPATH=. python3 scripts/tilt_walk_forward.py
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
from src.walk_forward import WalkForwardConfig, build_folds  # noqa: E402

COST_BPS = 5.0

# Candidate grid (core_weight, top_n, lookback_days, freq).
CANDIDATES = [
    (0.50, 2, 252, "Q"),
    (0.50, 3, 126, "Q"),
    (0.60, 3, 126, "Q"),
    (0.60, 2, 126, "Q"),
    (0.70, 3, 126, "Q"),
    (0.70, 3, 126, "M"),
    (0.50, 3, 126, "A"),
    (0.00, 1, 126, "Q"),
    (0.00, 3, 126, "Q"),
]


def load_closes() -> tuple[pd.DataFrame, list[str]]:
    universe = [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]
    wide = load_ohlcv_multi(universe + [BENCHMARK], "1d")
    close = wide.xs("close", axis=1, level=1).sort_index()
    close.index = pd.to_datetime(close.index)
    return close, universe


def _momentum(close, universe, t, lookback) -> pd.Series:
    hist = close[universe].loc[:t]
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    window = hist.iloc[-(lookback + 1):]
    return window.iloc[-1] / window.iloc[0] - 1.0


def _reb_dates(idx, freq):
    s = pd.Series(idx, index=idx)
    per = {"M": "M", "Q": "Q", "A": "Y"}[freq]
    return list(pd.DatetimeIndex(s.groupby(s.dt.to_period(per)).last().values))


def _weights(close, universe, t, core_weight, top_n, lookback) -> pd.Series:
    w = pd.Series(0.0, index=close.columns)
    w[BENCHMARK] = core_weight
    mom = _momentum(close, universe, t, lookback).dropna()
    tilt = 1.0 - core_weight
    if mom.empty:
        w[BENCHMARK] = 1.0
        return w
    top = mom.sort_values(ascending=False).head(top_n)
    top = top[top > 0]
    if top.empty:
        w[BENCHMARK] += tilt
        return w
    for tkr, wt in (top / top.sum() * tilt).items():
        w[tkr] += wt
    return w


def run_window(close, universe, cfg, start, end) -> float:
    """CAGR of one tilt config over [start, end]. Full mark-to-market with
    rebalances + costs; weights chosen causally from data <= each reb date."""
    cw, tn, lb, fr = cfg
    # Need lookback history BEFORE start, so feed the full close frame but only
    # accrue equity within the window.
    win = close.loc[start:end]
    if len(win) < 30:
        return float("nan")
    idx = win.index
    daily = close.pct_change()
    rebs = [d for d in _reb_dates(idx, fr) if idx[0] <= d <= idx[-1]]
    equity = 1.0
    cur_w = _weights(close, universe, idx[0], cw, tn, lb)  # set at window open
    next_i = 0
    for d in idx[1:]:
        equity *= (1.0 + float((cur_w * daily.loc[d]).sum()))
        if next_i < len(rebs) and d == rebs[next_i]:
            nw = _weights(close, universe, d, cw, tn, lb)
            turn = float((nw - cur_w).abs().sum()) / 2.0
            equity *= (1.0 - turn * COST_BPS / 10_000.0)
            cur_w = nw
            next_i += 1
    years = (idx[-1] - idx[0]).days / 365.25
    return equity ** (1 / years) - 1.0 if years > 0 else float("nan")


def spy_cagr(close, start, end) -> float:
    s = close[BENCHMARK].loc[start:end].dropna()
    if len(s) < 2:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1.0 if years > 0 else float("nan")


def main() -> int:
    close, universe = load_closes()
    folds = build_folds(close, WalkForwardConfig())
    print(f"Loaded {len(close)} bars. {len(folds)} folds. {len(CANDIDATES)} candidates.\n")

    rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        tr_s, tr_e = pd.Timestamp(tr_s), pd.Timestamp(tr_e)
        te_s, te_e = pd.Timestamp(te_s), pd.Timestamp(te_e)
        # 1) pick train winner by CAGR
        train_scores = {c: run_window(close, universe, c, tr_s, tr_e) for c in CANDIDATES}
        winner = max(train_scores, key=lambda c: (train_scores[c]
                     if not np.isnan(train_scores[c]) else -9))
        # 2) score winner OOS, plus SPY OOS
        oos = run_window(close, universe, winner, te_s, te_e)
        spy = spy_cagr(close, te_s, te_e)
        rows.append({"fold": i, "test": f"{te_s.date()} → {te_e.date()}",
                     "winner": winner, "oos": oos, "spy": spy,
                     "delta": oos - spy})

    print(f"{'Fold':<5}{'Test window':<26}{'Train pick':<22}{'OOS':>8}{'SPY':>8}{'Δ':>8}")
    print("-" * 77)
    for r in rows:
        cw, tn, lb, fr = r["winner"]
        pick = f"c{int(cw*100)}/top{tn}/{lb}d/{fr}"
        print(f"{r['fold']:<5}{r['test']:<26}{pick:<22}"
              f"{r['oos']*100:>7.1f}%{r['spy']*100:>7.1f}%{r['delta']*100:>7.1f}%")

    d = np.array([r["delta"] for r in rows])
    wins = int((d > 1e-6).sum()); losses = int((d < -1e-6).sum())
    # which configs got picked
    from collections import Counter
    picks = Counter((f"c{int(c[0]*100)}/top{c[1]}/{c[2]}d/{c[3]}") for c in (r["winner"] for r in rows))
    print("-" * 77)
    print(f"Mean OOS Δ vs SPY: {d.mean()*100:+.2f}%/yr   "
          f"({wins} folds beat SPY, {losses} lost, median Δ {np.median(d)*100:+.2f}%)")
    print(f"Train picked: {dict(picks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
