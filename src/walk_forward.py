"""Walk-forward parameter sweep for the mechanical core.

Why walk-forward (and not "just sweep on the full window")
----------------------------------------------------------
Picking the best parameter on the full 4-year history is **curve-fitting**:
you learn what worked, then claim it'll keep working. The strategy looks
great on data it was fit on and dies out of sample.

Walk-forward instead:
  1. Split the timeline into rolling (train, test) folds.
  2. On each TRAIN window, find the candidate value that maxes the metric.
  3. Score that *chosen* value on the TEST window — out-of-sample, untouched.
  4. Aggregate across folds. A parameter is "robust" iff its OOS score is
     consistently good across folds — not just on average.

This module deliberately keeps the sweep ONE-DIMENSIONAL per parameter
(holds all others at defaults while varying one). Full joint grids are
combinatorial and the dataset is too short to support more than a few df.
The 1D result is the floor of what we can claim honestly; a small joint
verification at the end checks the winners aren't mutually destructive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, load_price_panel, run_backtest

DEFAULT_GRIDS: dict[str, list[Any]] = {
    "extension_pct_cutoff":     [0.08, 0.12, 0.18, 0.25, 0.40],
    "weak_rs_rank_cutoff":      [0,    1,    2,    3,    4],
    "stale_buy_weeks":          [3,    4,    6,    8,    12],
    "momentum_window":          [21,   42,   63,   126,  200],
    "chase_weight_fraction":    [0.0,  0.25, 0.5,  0.75, 1.0],
    "cash_buffer":              [0.00, 0.02, 0.05, 0.10],
}

_CFG_FIELDS = {"chase_weight_fraction", "cash_buffer"}


@dataclass
class WalkForwardConfig:
    train_years: float = 2.0
    test_months: float = 6.0
    step_months: float = 6.0
    metric: str = "excess_cagr"
    min_train_rebalances: int = 30


@dataclass
class FoldResult:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    param: str
    candidate_train_scores: dict[Any, float]
    winner: Any
    winner_test_score: float
    default_test_score: float
    delta_vs_default: float


@dataclass
class ParamSweepResult:
    param: str
    candidates: list[Any]
    default_value: Any
    folds: list[FoldResult]
    winners_by_fold: list[Any]
    mean_winner_oos: float
    mean_default_oos: float
    mean_delta: float
    n_folds: int


def build_folds(closes: pd.DataFrame, wfc: WalkForwardConfig
                ) -> list[tuple[date, date, date, date]]:
    if closes.empty:
        return []
    first = closes.dropna(how="all").index[0].date()
    last = closes.dropna(how="all").index[-1].date()
    train_days = int(round(wfc.train_years * 365.25))
    test_days = int(round(wfc.test_months * 30.4375))
    step_days = int(round(wfc.step_months * 30.4375))

    folds: list[tuple[date, date, date, date]] = []
    train_start = first
    while True:
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days)
        if test_end > last:
            break
        folds.append((train_start, train_end, test_start, test_end))
        train_start = train_start + timedelta(days=step_days)
    return folds


def _make_config(param: str, value: Any, start: date, end: date,
                 base: BacktestConfig | None = None) -> BacktestConfig:
    base = base or BacktestConfig()
    cfg = BacktestConfig(
        start=start, end=end,
        execution=base.execution, cost_bps=base.cost_bps,
        slippage_bps=base.slippage_bps,
        cash_buffer=base.cash_buffer,
        initial_capital=base.initial_capital,
        trade_policy=base.trade_policy,
        sentiment_gate=base.sentiment_gate,
        param_overrides=dict(base.param_overrides or {}),
        chase_weight_fraction=base.chase_weight_fraction,
    )
    if param in _CFG_FIELDS:
        setattr(cfg, param, value)
    else:
        cfg.param_overrides = dict(cfg.param_overrides or {})
        cfg.param_overrides[param] = value
    return cfg


def _score(stats: dict, metric: str) -> float:
    s = stats["strategy"]
    if metric == "excess_cagr":
        return float(stats["excess_cagr"])
    if metric == "cagr":
        return float(s["cagr"])
    if metric == "sharpe":
        return float(s["sharpe"])
    raise ValueError(f"unknown metric {metric!r}")


def _default_for(param: str) -> Any:
    from config.settings import PARAMS
    if param == "cash_buffer":
        return 0.05
    return getattr(PARAMS, param)


def sweep_1d(param: str,
             candidates: list[Any] | None = None,
             wfc: WalkForwardConfig | None = None,
             base: BacktestConfig | None = None,
             closes: pd.DataFrame | None = None,
             opens: pd.DataFrame | None = None,
             on_progress: Callable[[str], None] | None = None,
             ) -> ParamSweepResult:
    wfc = wfc or WalkForwardConfig()
    candidates = list(candidates if candidates is not None
                       else DEFAULT_GRIDS[param])
    default_value = _default_for(param)
    if default_value not in candidates:
        candidates = candidates + [default_value]
    if closes is None or opens is None:
        closes, opens = load_price_panel()
    folds_schedule = build_folds(closes, wfc)
    if not folds_schedule:
        raise RuntimeError("not enough price history for the requested fold schedule")

    fold_results: list[FoldResult] = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds_schedule):
        if on_progress:
            on_progress(f"[{param}] fold {i+1}/{len(folds_schedule)}: "
                        f"train {tr_s}→{tr_e}, test {te_s}→{te_e}")
        train_scores: dict[Any, float] = {}
        for cand in candidates:
            cfg = _make_config(param, cand, tr_s, tr_e, base=base)
            try:
                res = run_backtest(cfg, closes=closes, opens=opens)
            except Exception:
                continue
            train_scores[cand] = _score(res.stats, wfc.metric)
        if not train_scores:
            continue
        winner = max(train_scores, key=train_scores.get)

        cfg_w = _make_config(param, winner, te_s, te_e, base=base)
        cfg_d = _make_config(param, default_value, te_s, te_e, base=base)
        try:
            w_oos = _score(run_backtest(cfg_w, closes=closes, opens=opens).stats,
                           wfc.metric)
        except Exception:
            w_oos = float("nan")
        try:
            d_oos = _score(run_backtest(cfg_d, closes=closes, opens=opens).stats,
                           wfc.metric)
        except Exception:
            d_oos = float("nan")
        fold_results.append(FoldResult(
            fold=i, train_start=tr_s, train_end=tr_e,
            test_start=te_s, test_end=te_e,
            param=param, candidate_train_scores=train_scores,
            winner=winner, winner_test_score=w_oos,
            default_test_score=d_oos,
            delta_vs_default=(w_oos - d_oos)
                             if not (np.isnan(w_oos) or np.isnan(d_oos))
                             else float("nan"),
        ))

    if not fold_results:
        raise RuntimeError(f"no successful folds for param {param!r}")

    winners = [f.winner for f in fold_results]
    w_oos = np.array([f.winner_test_score for f in fold_results
                      if not np.isnan(f.winner_test_score)])
    d_oos = np.array([f.default_test_score for f in fold_results
                      if not np.isnan(f.default_test_score)])
    deltas = np.array([f.delta_vs_default for f in fold_results
                       if not np.isnan(f.delta_vs_default)])
    return ParamSweepResult(
        param=param, candidates=candidates, default_value=default_value,
        folds=fold_results, winners_by_fold=winners,
        mean_winner_oos=float(w_oos.mean()) if len(w_oos) else float("nan"),
        mean_default_oos=float(d_oos.mean()) if len(d_oos) else float("nan"),
        mean_delta=float(deltas.mean()) if len(deltas) else float("nan"),
        n_folds=len(fold_results),
    )


def pick_robust_winner(result: ParamSweepResult) -> tuple[Any, str]:
    from collections import Counter
    counts = Counter(result.winners_by_fold)
    if not counts:
        return result.default_value, "no folds — keep default"

    if np.isnan(result.mean_delta) or result.mean_delta <= 0.005:
        sign = "negative" if result.mean_delta < 0 else "≤0.5pp"
        return result.default_value, (
            f"train picks lifted OOS by {result.mean_delta*100:+.2f}pp "
            f"({sign}) — keep default")

    max_n = max(counts.values())
    modes = [v for v, n in counts.items() if n == max_n]
    if len(modes) > 1:
        mean_by_cand: dict[Any, float] = {}
        for cand in modes:
            scores = [f.candidate_train_scores.get(cand, float("nan"))
                      for f in result.folds]
            scores = [s for s in scores if not np.isnan(s)]
            mean_by_cand[cand] = float(np.mean(scores)) if scores else float("-inf")
        winner = max(mean_by_cand, key=mean_by_cand.get)
    else:
        winner = modes[0]

    if winner == result.default_value:
        return winner, (
            f"modal pick == default ({max_n}/{result.n_folds} folds); "
            f"mean OOS lift {result.mean_delta*100:+.2f}pp — no change")
    return winner, (
        f"modal pick across {result.n_folds} folds ({max_n}/{result.n_folds}); "
        f"mean OOS lift {result.mean_delta*100:+.2f}pp")


# ---------------------------------------------------------------------------
# Status payload for the Dashboard's walk-forward trust badge
# (TRADING_EDGE_AUDIT.md item B3 — write side. Read side is
# app.py::_cached_walk_forward_verdict, which reads the JSON this produces.)
# ---------------------------------------------------------------------------

def build_status_payload(
    results: dict[str, ParamSweepResult],
    winners: dict[str, Any],
    reasons: dict[str, str],
    wfc: WalkForwardConfig,
    joint: dict | None = None,
    generated_at: date | None = None,
) -> dict:
    """Assemble the small JSON status object the Dashboard reads.

    Pure function — deliberately separated from `scripts/run_walk_forward.py`
    so the payload shape is unit-testable without running a real sweep
    (network/price-DB access). The CLI script's only remaining job is to
    call this and write the result to `data/walk_forward_status.json`.

    Schema (kept intentionally small — the Dashboard only ever renders a
    one-line trust badge from this, not the full sweep table; the full
    per-fold detail stays in the generated WALK_FORWARD_REPORT.md):

    {
      "generated_at": "YYYY-MM-DD",
      "schedule": "train Xy / test Ymo / step Zmo, metric=...",
      "verdicts": {
        param: {
          "winner": ..., "default": ..., "why": str,
          "mean_delta": float, "changed": bool,
        }, ...
      },
      "joint": {"mean_default_excess": ..., "mean_new_excess": ...,
                "mean_delta_excess": ...} | None,
    }
    """
    generated_at = generated_at or date.today()
    verdicts: dict[str, dict] = {}
    for param, result in results.items():
        winner = winners.get(param, result.default_value)
        verdicts[param] = {
            "winner": winner,
            "default": result.default_value,
            "why": reasons.get(param, ""),
            "mean_delta": (None if np.isnan(result.mean_delta)
                          else round(float(result.mean_delta), 5)),
            "changed": bool(winner != result.default_value),
        }
    payload = {
        "generated_at": generated_at.isoformat(),
        "schedule": (f"train {wfc.train_years}y / test {wfc.test_months}mo / "
                    f"step {wfc.step_months}mo, metric={wfc.metric}"),
        "verdicts": verdicts,
    }
    if joint:
        payload["joint"] = {
            "mean_default_excess": joint.get("mean_default_excess"),
            "mean_new_excess": joint.get("mean_new_excess"),
            "mean_delta_excess": joint.get("mean_delta_excess"),
        }
    return payload
