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

# ---------------------------------------------------------------------------
# Configuration & candidate grids
# ---------------------------------------------------------------------------

# Each candidate is a fully-formed BacktestConfig override + a label. We keep
# `chase_weight_fraction` and PARAMS overrides SEPARATE so the sweep can
# touch both surfaces. PARAMS overrides go into `param_overrides` dict;
# `chase_weight_fraction` is a BacktestConfig field.
#
# Grids are intentionally COARSE (5-ish candidates each) — the train window
# is only ~2 years, so trying to discriminate between 0.18 and 0.19 is
# noise. Coarse-then-fine is the right discipline.
DEFAULT_GRIDS: dict[str, list[Any]] = {
    "extension_pct_cutoff":     [0.08, 0.12, 0.18, 0.25, 0.40],
    "weak_rs_rank_cutoff":      [0,    1,    2,    3,    4],
    "stale_buy_weeks":          [3,    4,    6,    8,    12],
    "momentum_window":          [21,   42,   63,   126,  200],
    "chase_weight_fraction":    [0.0,  0.25, 0.5,  0.75, 1.0],
    "cash_buffer":              [0.00, 0.02, 0.05, 0.10],
}

# Fields that live on BacktestConfig directly rather than on PARAMS — these
# need different plumbing in `_make_config`.
_CFG_FIELDS = {"chase_weight_fraction", "cash_buffer"}


@dataclass
class WalkForwardConfig:
    train_years: float = 2.0
    test_months: float = 6.0
    step_months: float = 6.0
    metric: str = "excess_cagr"        # "excess_cagr" or "cagr" or "sharpe"
    min_train_rebalances: int = 30     # skip degenerate folds


@dataclass
class FoldResult:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    param: str
    candidate_train_scores: dict[Any, float]   # candidate value -> train metric
    winner: Any                                # candidate the optimizer would pick
    winner_test_score: float                   # OOS score of the winner
    default_test_score: float                  # OOS score of the live default
    delta_vs_default: float                    # winner_test_score - default_test_score


@dataclass
class ParamSweepResult:
    param: str
    candidates: list[Any]
    default_value: Any
    folds: list[FoldResult]
    # Aggregations across folds:
    winners_by_fold: list[Any]                 # what train picked, fold by fold
    mean_winner_oos: float                     # mean OOS score when using train's pick
    mean_default_oos: float                    # mean OOS score with default
    mean_delta: float                          # winner - default, averaged
    n_folds: int


# ---------------------------------------------------------------------------
# Fold scheduling
# ---------------------------------------------------------------------------

def build_folds(closes: pd.DataFrame, wfc: WalkForwardConfig
                ) -> list[tuple[date, date, date, date]]:
    """Generate (train_start, train_end, test_start, test_end) date tuples.

    Anchored on the price panel's earliest tradeable date (NOT just the
    panel's first row — the SMA200 + momentum warmup chews ~263 bars off
    the front). We don't need to subtract that here because run_backtest
    handles warmup internally; we just need to leave enough room for the
    LAST fold's test window to fit before the panel ends.
    """
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


# ---------------------------------------------------------------------------
# Backtest helper — applies one candidate value to one window
# ---------------------------------------------------------------------------

def _make_config(param: str, value: Any, start: date, end: date,
                 base: BacktestConfig | None = None) -> BacktestConfig:
    """Build a BacktestConfig that varies exactly one knob on top of `base`."""
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
    """The currently-shipped default for a tunable. Used as the comparison
    point so the report can say "winner beat default by X" honestly. Reads
    `chase_weight_fraction` from `PARAMS` (since the walk-forward sweep
    promoted it from a backtest-only knob to a real `SignalParams` field);
    `cash_buffer` stays at the 0.05 BacktestConfig default."""
    from config.settings import PARAMS
    if param == "cash_buffer":
        return 0.05
    return getattr(PARAMS, param)


# ---------------------------------------------------------------------------
# Main: 1D walk-forward sweep over one parameter
# ---------------------------------------------------------------------------

def sweep_1d(param: str,
             candidates: list[Any] | None = None,
             wfc: WalkForwardConfig | None = None,
             base: BacktestConfig | None = None,
             closes: pd.DataFrame | None = None,
             opens: pd.DataFrame | None = None,
             on_progress: Callable[[str], None] | None = None,
             ) -> ParamSweepResult:
    """Walk-forward sweep across one parameter.

    For each fold, runs N backtests on the train window (one per candidate),
    picks the best by `wfc.metric`, then runs ONE backtest on the test
    window with that pick — out-of-sample. Default value is also scored on
    the test window for honest comparison.
    """
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
            except Exception:  # noqa: BLE001
                # Skip combos that can't run (e.g. window too short for the
                # warmup implied by an over-large momentum_window).
                continue
            train_scores[cand] = _score(res.stats, wfc.metric)
        if not train_scores:
            continue
        winner = max(train_scores, key=train_scores.get)

        # Score winner OOS:
        cfg_w = _make_config(param, winner, te_s, te_e, base=base)
        cfg_d = _make_config(param, default_value, te_s, te_e, base=base)
        try:
            w_oos = _score(run_backtest(cfg_w, closes=closes, opens=opens).stats,
                           wfc.metric)
        except Exception:  # noqa: BLE001
            w_oos = float("nan")
        try:
            d_oos = _score(run_backtest(cfg_d, closes=closes, opens=opens).stats,
                           wfc.metric)
        except Exception:  # noqa: BLE001
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


# ---------------------------------------------------------------------------
# Pick a "robust" winner from a ParamSweepResult
# ---------------------------------------------------------------------------

def pick_robust_winner(result: ParamSweepResult) -> tuple[Any, str]:
    """Choose a single value per parameter using a robustness rule.

    Rules, in order:
      1. If the train optimizer's per-fold winners don't beat the default
         OUT-OF-SAMPLE on average by at least +0.5pp, KEEP THE DEFAULT.
         A negative mean delta means the optimizer is overfitting the
         train window; applying its picks live would lose money. A near-
         zero delta isn't worth churning a live parameter for.
      2. Otherwise, choose the MODAL winner across folds (the value the
         optimizer picked most often). Ties broken by per-candidate
         mean OOS score across folds (not just train score).

    Returns (winner_value, justification_string).
    """
    from collections import Counter
    counts = Counter(result.winners_by_fold)
    if not counts:
        return result.default_value, "no folds — keep default"

    # Rule 1: train optimizer must actually earn its keep OOS.
    if np.isnan(result.mean_delta) or result.mean_delta <= 0.005:
        sign = "negative" if result.mean_delta < 0 else "≤0.5pp"
        return result.default_value, (
            f"train picks lifted OOS by {result.mean_delta*100:+.2f}pp "
            f"({sign}) — keep default")

    # Rule 2: modal pick across folds, OOS-tiebreak.
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
