"""Walk-forward out-of-sample test of the regime-aware bull overlay.

Unlike the 1D parameter sweep (`run_walk_forward.py`), the overlay is not a
single tunable value — it's a behavioural switch (promote CHASE + drop buffer
in confirmed BULL regimes only). So we test it directly: on each fold's
out-of-sample test window, run the defensive baseline vs `regime_aware=True`
and compare. There is nothing to "fit" on the train window — the regime
classifier is rule-based — so each test window is a clean OOS read.

Reports per-fold and mean OOS excess_cagr AND cagr (both, since the overlay
targets absolute bull capture which excess_cagr partly masks).

Reproduce:
    PYTHONPATH=. python3 scripts/walk_forward_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.backtest import BacktestConfig, load_price_panel, run_backtest  # noqa: E402
from src.walk_forward import WalkForwardConfig, build_folds  # noqa: E402


def main() -> int:
    wfc = WalkForwardConfig()
    closes, opens = load_price_panel()
    folds = build_folds(closes, wfc)
    print(f"Loaded {len(closes)} bars. {len(folds)} OOS test windows.\n")

    rows = []
    for i, (_tr_s, _tr_e, te_s, te_e) in enumerate(folds, 1):
        base = run_backtest(BacktestConfig(start=te_s, end=te_e),
                            closes=closes, opens=opens)
        aware = run_backtest(BacktestConfig(start=te_s, end=te_e, regime_aware=True),
                             closes=closes, opens=opens)
        rows.append({
            "fold": i, "test": f"{te_s} → {te_e}",
            "base_excess": base.stats["excess_cagr"],
            "new_excess": aware.stats["excess_cagr"],
            "base_cagr": base.stats["strategy"]["cagr"],
            "new_cagr": aware.stats["strategy"]["cagr"],
        })
        print(f"  fold {i:>2}/{len(folds)}: {te_s} → {te_e}  "
              f"excess {base.stats['excess_cagr']*100:+6.2f}% → "
              f"{aware.stats['excess_cagr']*100:+6.2f}%")

    def pct(x):
        return f"{x*100:+7.2f}%"

    print("\n| Fold | Test window | Base excess | New excess | Δ excess | Base CAGR | New CAGR | Δ CAGR |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['fold']} | {r['test']} | {pct(r['base_excess'])} | "
              f"{pct(r['new_excess'])} | {pct(r['new_excess']-r['base_excess'])} | "
              f"{pct(r['base_cagr'])} | {pct(r['new_cagr'])} | "
              f"{pct(r['new_cagr']-r['base_cagr'])} |")

    be = np.array([r["base_excess"] for r in rows])
    ne = np.array([r["new_excess"] for r in rows])
    bc = np.array([r["base_cagr"] for r in rows])
    nc = np.array([r["new_cagr"] for r in rows])
    wins = int((ne > be + 1e-9).sum())
    losses = int((ne < be - 1e-9).sum())
    print(f"\nMean OOS excess_cagr — base: {pct(be.mean())}, new: {pct(ne.mean())}, "
          f"delta: {pct((ne-be).mean())}")
    print(f"Mean OOS cagr        — base: {pct(bc.mean())}, new: {pct(nc.mean())}, "
          f"delta: {pct((nc-bc).mean())}")
    print(f"Per-fold: {wins} better, {losses} worse, {len(rows)-wins-losses} unchanged "
          f"(by excess_cagr)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
