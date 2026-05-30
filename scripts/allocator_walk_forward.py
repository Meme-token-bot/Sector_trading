"""Walk-forward validation of the cross-sectional sector allocator.

The full-window run had every scored config beating SPY, with macro adding
value. That's exactly the setup most likely to be overfit, so test it honestly:
on each fold, pick the best allocator config by CAGR on the TRAIN window, then
score THAT config out-of-sample on the next window vs SPY. Same fold schedule
as every other walk-forward in this repo (src.walk_forward.build_folds).

A second question this answers: does the macro leg survive OOS, or does the
train process just always pick tech-only? We log which config wins each fold.

Run:  PYTHONPATH=. python3 scripts/allocator_walk_forward.py
Writes /tmp/allocator_wf.txt + stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.walk_forward import WalkForwardConfig, build_folds  # noqa: E402
import scripts.sector_allocator as A  # noqa: E402


def main() -> int:
    close, universe, macro = A.load_data()
    folds = build_folds(close, WalkForwardConfig())

    rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        tr_s, tr_e = pd.Timestamp(tr_s), pd.Timestamp(tr_e)
        te_s, te_e = pd.Timestamp(te_s), pd.Timestamp(te_e)
        # 1) pick train winner by CAGR (skip the EW baseline as a "pick")
        train = {}
        for name, cfg in A.CONFIGS.items():
            if name == "equal-weight 11":
                continue
            r = A.run(close, universe, macro, cfg, tr_s, tr_e, "M")
            train[name] = r["cagr"]
        winner = max(train, key=lambda n: (train[n] if not np.isnan(train[n]) else -9))
        # 2) score winner OOS vs SPY
        oos = A.run(close, universe, macro, A.CONFIGS[winner], te_s, te_e, "M")
        spy = A.spy_stats(close, te_s, te_e)
        rows.append({"fold": i, "test": f"{te_s.date()} -> {te_e.date()}",
                     "winner": winner, "oos": oos["cagr"], "spy": spy["cagr"],
                     "delta": oos["cagr"] - spy["cagr"]})

    L = [f"{'Fold':<5}{'Test window':<26}{'Train pick':<22}{'OOS':>8}{'SPY':>8}{'D':>8}",
         "-" * 77]
    for r in rows:
        L.append(f"{r['fold']:<5}{r['test']:<26}{r['winner']:<22}"
                 f"{r['oos']*100:>7.1f}%{r['spy']*100:>7.1f}%{r['delta']*100:>7.1f}%")
    d = np.array([r["delta"] for r in rows])
    wins = int((d > 1e-6).sum()); losses = int((d < -1e-6).sum())
    mean = float(d.mean()); med = float(np.median(d)); sd = float(d.std(ddof=1))
    tstat = mean / (sd / np.sqrt(len(d))) if sd > 0 else float("nan")
    from collections import Counter
    picks = Counter(r["winner"] for r in rows)
    L.append("-" * 77)
    L.append(f"Mean OOS D vs SPY: {mean*100:+.2f}%/yr  median {med*100:+.2f}%  "
             f"({wins} beat SPY, {losses} lost)  t-stat {tstat:.2f}")
    L.append(f"Best fold {d.max()*100:+.1f}%  worst {d.min()*100:+.1f}%")
    L.append(f"Train picked: {dict(picks)}")
    out = "\n".join(L) + "\n"
    Path("/tmp/allocator_wf.txt").write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
