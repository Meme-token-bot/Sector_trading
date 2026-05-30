"""A/B the regime-aware bull overlay against the defensive baseline.

Runs the full backtest twice on the real price panel — once with
`regime_aware=False` (today's model) and once `True` (promote CHASE to full
weight + drop the cash buffer in confirmed BULL regimes only) — and prints a
side-by-side of the metrics that actually matter for the bull-capture question:

  * headline CAGR / excess vs SPY / max drawdown
  * per-regime up-capture & down-capture (BULL row is the target)
  * drawdown-attribution win count (must NOT degrade)

Read-only: writes nothing. Reproduce:
    PYTHONPATH=. python3 scripts/compare_regime_aware.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import BENCHMARK  # noqa: E402
from src.backtest import BacktestConfig, load_price_panel, run_backtest  # noqa: E402
from src.regime_analysis import (  # noqa: E402
    classify_regimes,
    drawdown_attribution,
    regime_conditional_stats,
)


def _dd_wins(result, spy_close) -> tuple[int, int, float]:
    rows = drawdown_attribution(result, result.benchmark_equity, spy_close,
                                min_dd_pct=0.05, min_days=5)
    if not rows:
        return 0, 0, 0.0
    wins = sum(1 for r in rows if r["excess_drawdown"] > 0)
    mean_excess = sum(r["excess_drawdown"] for r in rows) / len(rows)
    return wins, len(rows), mean_excess


def main() -> int:
    closes, opens = load_price_panel()
    spy_close = closes[BENCHMARK].dropna()

    base = run_backtest(BacktestConfig(), closes=closes, opens=opens)
    aware = run_backtest(BacktestConfig(regime_aware=True), closes=closes, opens=opens)

    def regstats(res):
        regimes = classify_regimes(spy_close.reindex(res.equity.index, method="ffill"))
        return regime_conditional_stats(res.equity, res.benchmark_equity, regimes)

    rs_b, rs_a = regstats(base), regstats(aware)

    bs, as_ = base.stats, aware.stats
    print(f"Window: {bs['window_start']} → {bs['window_end']}\n")
    print(f"{'Metric':<22}{'Baseline':>12}{'Regime-aware':>14}")
    print("-" * 48)
    print(f"{'CAGR':<22}{bs['strategy']['cagr']*100:>11.2f}%{as_['strategy']['cagr']*100:>13.2f}%")
    print(f"{'SPY CAGR':<22}{bs['spy']['cagr']*100:>11.2f}%{as_['spy']['cagr']*100:>13.2f}%")
    print(f"{'Excess CAGR':<22}{bs['excess_cagr']*100:>11.2f}%{as_['excess_cagr']*100:>13.2f}%")
    print(f"{'Max drawdown':<22}{bs['strategy']['max_drawdown']*100:>11.2f}%{as_['strategy']['max_drawdown']*100:>13.2f}%")
    print(f"{'Sharpe':<22}{bs['strategy']['sharpe']:>12.2f}{as_['strategy']['sharpe']:>14.2f}")
    print(f"{'Trades':<22}{bs['n_trades']:>12,}{as_['n_trades']:>14,}")
    print(f"{'Ann turnover':<22}{bs['annualised_turnover']:>11.2f}x{as_['annualised_turnover']:>13.2f}x")

    print(f"\nPer-regime up / down capture (BULL is the target):")
    print(f"{'Regime':<14}{'Base up':>9}{'Aware up':>10}{'Base dn':>9}{'Aware dn':>10}"
          f"{'Base exc':>10}{'Aware exc':>11}")
    print("-" * 73)
    for reg in rs_b.index:
        bu = rs_b.loc[reg, "capture_up"]; au = rs_a.loc[reg, "capture_up"]
        bd = rs_b.loc[reg, "capture_down"]; ad = rs_a.loc[reg, "capture_down"]
        be = rs_b.loc[reg, "excess_cum"]; ae = rs_a.loc[reg, "excess_cum"]
        print(f"{reg:<14}{bu:>9.2f}{au:>10.2f}{bd:>9.2f}{ad:>10.2f}"
              f"{be*100:>9.1f}%{ae*100:>10.1f}%")

    wb, nb, eb = _dd_wins(base, spy_close)
    wa, na, ea = _dd_wins(aware, spy_close)
    print(f"\nDrawdown attribution (protection must NOT degrade):")
    print(f"  Baseline:     {wb}/{nb} wins, mean excess {eb*100:+.2f}%")
    print(f"  Regime-aware: {wa}/{na} wins, mean excess {ea*100:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
