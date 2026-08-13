"""Unified, labeled cross-asset opportunity board.

Pulls today's actionable rows from the two existing, deliberately SEPARATE
signal pipelines into one table for scanning:

  * the sentiment-gated SPDR/expression model (src/signals.py) -- NEW_BUY
    sectors, plus any CHASE sector currently funded by the partial-CHASE
    sleeve (chase_weight_fraction > 0 and the ticker actually landed in
    `targets`, mirroring the exact condition app.py's "This Week's Orders"
    panel already uses to decide whether a CHASE row is actionable);
  * the un-gated breakout/consolidation model (src/breakout_signals.py, via
    whatever shape app.py's `_cached_breakout_universe` already produces)
    -- BUY / HIGH_CONVICTION_BUY rows over the expanded universe (metals,
    crypto, plain sector proxies scanned independently of the sentiment
    gate; see that module's docstring for why it is NOT sentiment-gated).

The two pipelines are NEVER merged into one vocabulary or one score. Every
row keeps its own native state string (NEW_BUY / CHASE vs BUY /
HIGH_CONVICTION_BUY) and is tagged with an explicit `model` label. Ranking
happens WITHIN each model group only -- there is deliberately no cross-model
score. SPDR rows are ranked using `src.expression_signals.
expression_strength_score` (0-100) of the actual vehicle the caller would
buy for that sector (the caller wires this in via `expression_scores`, since
computing it requires price lookups this module intentionally does not do).
Breakout rows are ranked using their own native `conviction` (0-5). These
two numbers are never compared to each other -- see `score_display`, which
always spells out the scale so a reader can't mistake a "72" for being
better or worse than a "4" without noticing they're different systems.

Pure function: no IO, no Streamlit, no caching -- same convention as
regime_analysis.py / macro_alignment.py / risk_metrics.py in this codebase.
The caller (app.py) is responsible for sourcing `signals`, `targets`,
`breakout_rows`, and (optionally) `expression_scores`.
"""
from __future__ import annotations

import pandas as pd

MODEL_SPDR = "Sentiment-gated (SPDR)"
MODEL_BREAKOUT = "Price/flow only (Breakout)"

_MODEL_ORDER = {MODEL_SPDR: 0, MODEL_BREAKOUT: 1}

# Column order for the returned DataFrame. Additive-only convention -- if a
# future caller wants more columns, append; don't reorder these.
COLUMNS = [
    "model", "ticker", "name", "state", "score", "score_display",
    "target_weight", "why", "group_rank",
]

_BREAKOUT_BUY_STATES = ("BUY", "HIGH_CONVICTION_BUY")


def _spdr_rows(
    signals: pd.DataFrame | None,
    targets: pd.Series,
    expression_scores: dict[str, int] | None,
    chase_weight_fraction: float,
) -> list[dict]:
    if signals is None or signals.empty:
        return []
    expression_scores = expression_scores or {}
    rows: list[dict] = []
    for tkr, row in signals.iterrows():
        state = row.get("state", "")
        is_new_buy = state == "NEW_BUY"
        is_funded_chase = (
            state == "CHASE"
            and chase_weight_fraction > 0
            and tkr in targets.index
        )
        if not (is_new_buy or is_funded_chase):
            continue
        score = expression_scores.get(tkr)
        score_display = f"{score}/100 (vehicle)" if score is not None else "—"
        weight = float(targets.get(tkr, 0.0)) if tkr in targets.index else 0.0
        rows.append({
            "model": MODEL_SPDR,
            "ticker": tkr,
            "name": row.get("name", tkr),
            "state": state,
            "score": score,
            "score_display": score_display,
            "target_weight": weight,
            "why": row.get("state_reason", ""),
        })
    return rows


def _breakout_rows(breakout_rows: list[dict] | None) -> list[dict]:
    if not breakout_rows:
        return []
    rows: list[dict] = []
    for r in breakout_rows:
        sig = r.get("signal")
        if sig not in _BREAKOUT_BUY_STATES:
            continue
        conv = int(r.get("conviction", 0) or 0)
        rows.append({
            "model": MODEL_BREAKOUT,
            "ticker": r.get("ticker", ""),
            "name": r.get("label", r.get("ticker", "")),
            "state": sig,
            "score": conv,
            "score_display": f"{conv}/5 conviction",
            "target_weight": float("nan"),
            "why": " · ".join((r.get("reasons") or [])[:3]),
        })
    return rows


def build_opportunity_board(
    signals: pd.DataFrame,
    targets: pd.Series,
    breakout_rows: list[dict] | None,
    *,
    expression_scores: dict[str, int] | None = None,
    chase_weight_fraction: float | None = None,
) -> pd.DataFrame:
    """Merge today's actionable SPDR + breakout rows into one labeled table.

    Parameters
    ----------
    signals : refined-signals frame from `src.signals.refine_signals` (must
        carry a `state` column; `state_reason` / `name` used when present).
    targets : target-weight Series from `src.signals.target_weights`.
    breakout_rows : the list[dict] shape app.py's `_cached_breakout_universe`
        already produces (one dict per expanded-universe ticker).
    expression_scores : optional `{sector_ticker: score}` map -- the
        `src.expression_signals.expression_strength_score` of that sector's
        top-ranked expression (i.e. the same vehicle the orders panel's
        `_cached_top_vehicle` would surface). Reused here, unmodified, as
        the within-group ranking score for SPDR rows. When omitted (or a
        ticker is missing from it), that row's score is left as `None` /
        "—" and it sorts to the bottom of the SPDR group -- it never falls
        back to a rescaled `conviction` or any other stand-in that would
        look like a second scoring system.
    chase_weight_fraction : defaults to `config.settings.PARAMS.
        chase_weight_fraction` when omitted, matching
        `src.signals.target_weights`'s own default.

    Returns
    -------
    DataFrame, one row per actionable opportunity, columns per `COLUMNS`.
    Sorted by `model` (SPDR group first -- this repo already treats the
    sentiment-gated model as primary and the breakout scanner as a
    supplementary read, see README.md's weekly workflow), then by that
    row's OWN model-native score descending within the group, then by
    ticker for a deterministic tie-break. `group_rank` is the 1-based rank
    *within the row's own model group* -- deliberately not a global rank,
    since a global rank would imply the two scores sit on one shared scale,
    which they do not.
    """
    if chase_weight_fraction is None:
        from config.settings import PARAMS
        chase_weight_fraction = float(PARAMS.chase_weight_fraction)

    rows = _spdr_rows(signals, targets, expression_scores, chase_weight_fraction)
    rows += _breakout_rows(breakout_rows)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows)

    # Missing SPDR scores (None) must sort AFTER real scores within their
    # group. -1 is a safe sentinel: real scores are 0-100 (SPDR) or 0-5
    # (breakout, never missing), so -1 is strictly below any real value.
    df["_model_rank"] = df["model"].map(_MODEL_ORDER).fillna(99)
    df["_score_sort"] = df["score"].fillna(-1)
    df = df.sort_values(
        ["_model_rank", "_score_sort", "ticker"],
        ascending=[True, False, True],
    )
    df["group_rank"] = df.groupby("model").cumcount() + 1
    df = df.drop(columns=["_model_rank", "_score_sort"]).reset_index(drop=True)
    return df[COLUMNS]