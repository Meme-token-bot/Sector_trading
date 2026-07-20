"""Walk-forward parameter sweep + joint validation + report.

Usage:
    PYTHONPATH=. python3 scripts/run_walk_forward.py
    PYTHONPATH=. python3 scripts/run_walk_forward.py --train-years 1.5 --test-months 4

Procedure:
  1. For each tunable parameter, run a 1D walk-forward sweep — for every fold,
     pick the train-window winner, then score it OUT OF SAMPLE on the next
     6 months.
  2. Pick a robust winner per parameter (modal across folds; ignore <0.5pp
     lifts to avoid parameter churn).
  3. Build a JOINT config from those winners and score it walk-forward
     against the current defaults. This is the honest test that the
     1D-chosen values don't conflict with each other.
  4. Write `WALK_FORWARD_REPORT.md` with the per-param tables and the joint
     verdict. Print the suggested PARAMS diff to stdout.
  5. Write `data/walk_forward_status.json` — a small machine-readable verdict
     summary the Dashboard's Decision Cockpit reads to render a one-line
     "walk-forward validated as of DATE, N/M defaults kept OOS" trust badge
     (TRADING_EDGE_AUDIT.md item B3). This is deliberately NOT the full
     report — just enough for the badge. The full per-fold detail stays in
     WALK_FORWARD_REPORT.md, which this script also still writes.

Nothing here mutates `config/settings.py`. Apply the diff manually if you
agree with it.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from src.backtest import BacktestConfig, load_price_panel, run_backtest
from src.walk_forward import (
    DEFAULT_GRIDS,
    ParamSweepResult,
    WalkForwardConfig,
    _CFG_FIELDS,
    build_folds,
    build_status_payload,
    pick_robust_winner,
    sweep_1d,
)


PARAMS_TO_SWEEP = [
    "extension_pct_cutoff",
    "weak_rs_rank_cutoff",
    "stale_buy_weeks",
    "momentum_window",
    "chase_weight_fraction",
    "cash_buffer",
]


def _pct(x: float, p: int = 2, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    fmt = f"{{:{'+' if signed else ''}.{p}f}}%"
    return fmt.format(x * 100.0)


def _render_param_section(param: str, result: ParamSweepResult,
                          winner: Any, why: str) -> str:
    lines: list[str] = []
    lines.append(f"### `{param}`  (default {result.default_value})")
    lines.append("")
    lines.append(f"- **Robust winner:** `{winner}`  *(reason: {why})*")
    lines.append(f"- Folds: {result.n_folds}. "
                 f"Mean OOS with winner: {_pct(result.mean_winner_oos)}. "
                 f"Mean OOS with default: {_pct(result.mean_default_oos)}. "
                 f"Mean lift: **{_pct(result.mean_delta)}**.")
    lines.append("")
    lines.append("Per-fold winner picks (train-best) and OOS scores vs default:")
    lines.append("")
    lines.append("| Fold | Train window | Test window | Train winner | OOS (winner) | OOS (default) | Δ |")
    lines.append("|---:|---|---|:---:|---:|---:|---:|")
    for f in result.folds:
        lines.append(
            f"| {f.fold+1} | {f.train_start} → {f.train_end} | "
            f"{f.test_start} → {f.test_end} | `{f.winner}` | "
            f"{_pct(f.winner_test_score)} | {_pct(f.default_test_score)} | "
            f"{_pct(f.delta_vs_default)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _joint_validation(winners: dict[str, Any],
                      wfc: WalkForwardConfig,
                      closes, opens) -> dict:
    folds = build_folds(closes, wfc)
    param_overrides = {k: v for k, v in winners.items() if k not in _CFG_FIELDS}
    cfg_field_overrides = {k: v for k, v in winners.items() if k in _CFG_FIELDS}
    rows: list[dict] = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        cfg_def = BacktestConfig(start=te_s, end=te_e)
        cfg_new = BacktestConfig(
            start=te_s, end=te_e,
            param_overrides=param_overrides,
            **cfg_field_overrides,
        )
        try:
            r_def = run_backtest(cfg_def, closes=closes, opens=opens)
        except Exception:
            continue
        try:
            r_new = run_backtest(cfg_new, closes=closes, opens=opens)
        except Exception:
            continue
        rows.append({
            "fold": i + 1,
            "test_window": f"{te_s} → {te_e}",
            "default_cagr": float(r_def.stats["strategy"]["cagr"]),
            "new_cagr": float(r_new.stats["strategy"]["cagr"]),
            "spy_cagr": float(r_def.stats["spy"]["cagr"]),
            "default_excess": float(r_def.stats["excess_cagr"]),
            "new_excess": float(r_new.stats["excess_cagr"]),
        })
    if not rows:
        return {"folds": [], "mean_default_excess": float("nan"),
                "mean_new_excess": float("nan"), "mean_delta_excess": float("nan")}
    d_ex = np.array([r["default_excess"] for r in rows])
    n_ex = np.array([r["new_excess"] for r in rows])
    return {
        "folds": rows,
        "mean_default_excess": float(d_ex.mean()),
        "mean_new_excess": float(n_ex.mean()),
        "mean_delta_excess": float((n_ex - d_ex).mean()),
        "n_folds": len(rows),
    }


def _render_joint(joint: dict, winners: dict[str, Any]) -> str:
    lines = ["## Joint validation\n"]
    lines.append("Apply ALL the per-parameter winners simultaneously and "
                 "walk-forward-score the joint config against the current "
                 "defaults on the same fold schedule. This catches "
                 "destructive interactions between independently-chosen "
                 "winners.")
    lines.append("")
    lines.append("**Suggested parameter set:**\n")
    for k, v in winners.items():
        lines.append(f"- `{k}` = `{v}`")
    lines.append("")
    if joint.get("folds"):
        lines.append(f"**Mean OOS excess CAGR — default: "
                     f"{_pct(joint['mean_default_excess'])}, "
                     f"new: {_pct(joint['mean_new_excess'])}, "
                     f"delta: {_pct(joint['mean_delta_excess'])}** "
                     f"(over {joint['n_folds']} folds)")
        lines.append("")
        lines.append("| Fold | Test window | Default excess vs SPY | New excess vs SPY | Δ |")
        lines.append("|---:|---|---:|---:|---:|")
        for r in joint["folds"]:
            lines.append(
                f"| {r['fold']} | {r['test_window']} | "
                f"{_pct(r['default_excess'])} | {_pct(r['new_excess'])} | "
                f"{_pct(r['new_excess'] - r['default_excess'])} |"
            )
    else:
        lines.append("(joint validation produced no usable folds)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-years", type=float, default=2.0)
    ap.add_argument("--test-months", type=float, default=6.0)
    ap.add_argument("--step-months", type=float, default=6.0)
    ap.add_argument("--metric", default="excess_cagr",
                    choices=["excess_cagr", "cagr", "sharpe"])
    ap.add_argument("--report", default="WALK_FORWARD_REPORT.md")
    ap.add_argument("--status-json", default="data/walk_forward_status.json",
                    help="Where to write the Dashboard's trust-badge status "
                         "file (relative paths are resolved against ROOT).")
    args = ap.parse_args()

    wfc = WalkForwardConfig(
        train_years=args.train_years,
        test_months=args.test_months,
        step_months=args.step_months,
        metric=args.metric,
    )
    closes, opens = load_price_panel()
    folds = build_folds(closes, wfc)
    print(f"Loaded {len(closes)} bars. Built {len(folds)} fold(s):")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        print(f"  Fold {i}: train {tr_s}→{tr_e}, test {te_s}→{te_e}")
    print()

    results: dict[str, ParamSweepResult] = {}
    winners: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for param in PARAMS_TO_SWEEP:
        print(f"=== sweeping {param} (candidates: {DEFAULT_GRIDS[param]}) ===")
        res = sweep_1d(param, wfc=wfc, closes=closes, opens=opens,
                       on_progress=lambda s: print("  " + s))
        w, why = pick_robust_winner(res)
        results[param] = res
        winners[param] = w
        reasons[param] = why
        print(f"  → winner: {w!r}  ({why})")
        print(f"  → mean OOS lift vs default: {_pct(res.mean_delta)}")
        print()

    print("=== joint validation ===")
    joint = _joint_validation(winners, wfc, closes, opens)
    print(f"  mean OOS excess — default: {_pct(joint['mean_default_excess'])}, "
          f"new: {_pct(joint['mean_new_excess'])}, "
          f"delta: {_pct(joint['mean_delta_excess'])}")
    print()

    # ---- render report ----
    sections = ["# Walk-Forward Parameter Sweep\n",
                f"**Generated:** {date.today().isoformat()}",
                f"**Schedule:** train {args.train_years} years, "
                f"test {args.test_months} months, step {args.step_months} months. "
                f"Metric: `{args.metric}`. {len(folds)} folds.",
                f"**Reproduce:** `PYTHONPATH=. python3 scripts/run_walk_forward.py`\n",
                "## Per-parameter 1D sweeps\n"]
    for param in PARAMS_TO_SWEEP:
        sections.append(_render_param_section(param, results[param],
                                               winners[param], reasons[param]))
    sections.append(_render_joint(joint, winners))
    sections.append("\n## Suggested PARAMS diff\n")
    sections.append("```python")
    sections.append("# config/settings.SignalParams — current → suggested:")
    from config.settings import PARAMS
    for k, v in winners.items():
        cur = (getattr(PARAMS, k) if k not in _CFG_FIELDS
               else ("0.0 (chase_weight_fraction)" if k == "chase_weight_fraction"
                     else "0.05 (cash_buffer)"))
        marker = "  # NO CHANGE" if str(cur) == str(v) else "  # CHANGE"
        sections.append(f"  {k}: {cur} → {v}{marker}")
    sections.append("```")
    sections.append("\n## How to apply\n")
    sections.append(
        "1. Update the corresponding fields on `SignalParams` in "
        "`config/settings.py` to the values above (only the `# CHANGE` rows). "
        "`chase_weight_fraction` and `cash_buffer` live on `BacktestConfig` / "
        "`target_weights()` instead — wire them through the live model where "
        "they matter.\n"
        "2. Re-run `scripts/run_backtest_report.py` to regenerate the "
        "headline backtest with the new defaults.\n"
        "3. Watch the dashboard for at least a quarter before declaring "
        "victory; walk-forward is a hypothesis, not a guarantee."
    )

    report_path = Path(args.report)
    report_path.write_text("\n".join(sections))
    print(f"Wrote: {report_path}")

    # ---- write the Dashboard trust-badge status file ----
    status_payload = build_status_payload(
        results=results, winners=winners, reasons=reasons,
        wfc=wfc, joint=joint,
    )
    status_path = Path(args.status_json)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    status_path.write_text(json.dumps(status_payload, indent=2))
    n_changed = sum(1 for v in status_payload["verdicts"].values() if v["changed"])
    n_total = len(status_payload["verdicts"])
    print(f"Wrote: {status_path}  "
          f"({n_total - n_changed}/{n_total} defaults kept OOS)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
