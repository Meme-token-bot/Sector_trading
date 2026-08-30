"""Industry Rotation Grid — 4-state classifier, rotation flow, sector health.

Pure functions, no IO — same convention as src/regime_snapshot.py,
src/macro_alignment.py, src/regime_analysis.py, src/risk_metrics.py. This
module never imports sqlite3, requests, or yfinance, and never touches
src/price_store.py or src/db.py directly. Callers (app.py) are responsible
for loading prices (src.price_store.load_ohlcv_multi) and for persisting /
reading back snapshots (src.db.save_industry_rotation_snapshot and
friends) — exactly the same split src/regime_snapshot.py already uses via
app.py's `_cached_regime_and_breadth`.

Deliberately independent of, and never modifies, src/signals.py,
refine_signals, target_weights, or signal_snapshots. This is a discovery /
context tool: it never sizes a position (see config/industries.py's module
docstring for the full non-goals list).

The four states
---------------
Classified per industry from price alone (no sentiment gate — same
reasoning src/breakout_signals.py already gives for being un-gated):

  CLIMBING  — above its own trend MA AND relative strength vs SPY is
              clearly improving over the trailing window.
  BASE      — above its own trend MA but RS is roughly flat / not clearly
              improving.
  TIRED     — below its own trend MA but RS is still positive, or the
              slide is too recent/mild to call a clean breakdown.
  DOWNHILL  — below its own trend MA AND RS is clearly deteriorating.
  NOT_ENOUGH_DATA — fewer bars than the classifier needs (see
              IndustryRotationParams and `_min_bars_needed` below).

See config.settings.IndustryRotationParams for the exact thresholds and
why the trend MA here is shorter than the sector model's SMA200.
"""
from __future__ import annotations

import pandas as pd

from config.industries import INDUSTRIES, IndustryProxy
from config.settings import INDUSTRY_ROTATION, IndustryRotationParams

STATES = ("CLIMBING", "BASE", "TIRED", "DOWNHILL")
FAVORABLE_STATES = frozenset({"CLIMBING", "BASE"})
UNFAVORABLE_STATES = frozenset({"TIRED", "DOWNHILL"})
NOT_ENOUGH_DATA = "NOT_ENOUGH_DATA"

_EMPTY_RESULT: dict = {
    "state": NOT_ENOUGH_DATA, "above_ma": None, "ma_value": None,
    "rs_now": None, "rs_prior": None, "rs_slope": None, "price": None,
}

_STATE_COLUMNS = ["industry", "label", "parent_sector", "ticker", "state",
                  "above_ma", "ma_value", "rs_now", "rs_prior", "rs_slope",
                  "price"]


def _min_bars_needed(params: IndustryRotationParams) -> int:
    """Bars required before RS-slope is computable at all — the binding
    constraint (RS itself needs `rs_window` bars of return, and the slope
    compares that to the SAME read `rs_slope_window` bars earlier). The
    trend MA's own `trend_ma_window` requirement is folded in via max()."""
    rs_chain = params.rs_window + params.rs_slope_window + 1
    return max(params.trend_ma_window, rs_chain)


def classify_industry_state(
    close: pd.Series,
    spy_close: pd.Series,
    params: IndustryRotationParams = INDUSTRY_ROTATION,
) -> dict:
    """Classify ONE industry's rotation state as of the LAST bar in `close`.

    `close` and `spy_close` should already be sliced to `as_of` by the
    caller for historical replay — same look-ahead discipline as
    src/market_engine.py::compute_sector_metrics(prices, as_of=...). This
    function itself does no slicing; it only ever looks at the tail of
    whatever it's handed.

    Returns a dict — never raises, never returns NaN silently mislabeled
    as a real reading:
      {
        "state": "CLIMBING"|"BASE"|"TIRED"|"DOWNHILL"|"NOT_ENOUGH_DATA",
        "above_ma": bool | None,
        "ma_value": float | None,
        "rs_now": float | None,      # industry return - SPY return, rs_window bars
        "rs_prior": float | None,    # same read, rs_slope_window bars earlier
        "rs_slope": float | None,    # rs_now - rs_prior
        "price": float | None,
      }
    """
    if close is None or spy_close is None or close.empty or spy_close.empty:
        return dict(_EMPTY_RESULT)

    close = close.dropna()
    if close.empty:
        return dict(_EMPTY_RESULT)

    # Align SPY onto the industry's own index (ffill, then drop any leading
    # dates SPY has no reading for yet) so every downstream positional
    # offset is guaranteed to compare same-date bars — mirrors
    # src/charts.py::compute_chart_overlays's own SPY-alignment pattern.
    spy_aligned = spy_close.reindex(close.index).ffill()
    common = spy_aligned.dropna().index
    close = close.loc[common]
    spy_aligned = spy_aligned.loc[common]

    needed = _min_bars_needed(params)
    if len(close) < needed:
        return dict(_EMPTY_RESULT)

    price = float(close.iloc[-1])
    ma_value = float(close.tail(params.trend_ma_window).mean())
    above_ma = price > ma_value

    def _rs_as_of(bars_back: int) -> float | None:
        end_i = len(close) - 1 - bars_back
        start_i = end_i - params.rs_window
        if start_i < 0 or end_i < 0:
            return None
        c_start, c_end = close.iloc[start_i], close.iloc[end_i]
        s_start, s_end = spy_aligned.iloc[start_i], spy_aligned.iloc[end_i]
        if c_start == 0 or s_start == 0:
            return None
        c_ret = c_end / c_start - 1.0
        s_ret = s_end / s_start - 1.0
        return float(c_ret - s_ret)

    rs_now = _rs_as_of(0)
    rs_prior = _rs_as_of(params.rs_slope_window)
    rs_slope = (rs_now - rs_prior) if (rs_now is not None and rs_prior is not None) else None

    if rs_now is None:
        # Length gate passed but a $0 print or similar corrupted a bar —
        # still report price/MA honestly rather than discarding everything.
        return {**_EMPTY_RESULT, "state": NOT_ENOUGH_DATA,
                "price": price, "above_ma": above_ma, "ma_value": ma_value}

    if above_ma:
        state = "CLIMBING" if (rs_slope is not None and rs_slope > params.rs_slope_threshold) else "BASE"
    else:
        clean_breakdown = (rs_slope is not None
                           and rs_slope <= -params.rs_slope_threshold
                           and rs_now <= 0.0)
        state = "DOWNHILL" if clean_breakdown else "TIRED"

    return {"state": state, "above_ma": above_ma, "ma_value": ma_value,
            "rs_now": rs_now, "rs_prior": rs_prior, "rs_slope": rs_slope,
            "price": price}


def compute_all_industry_states(
    prices: pd.DataFrame,
    industries: dict[str, IndustryProxy] | None = None,
    as_of: pd.Timestamp | None = None,
    params: IndustryRotationParams = INDUSTRY_ROTATION,
) -> pd.DataFrame:
    """One row per industry with its rotation state + diagnostics.

    `prices` is a wide close-price frame (one column per ticker, DatetimeIndex)
    and must include a column named "SPY" (config.settings.BENCHMARK).
    Industries whose ticker isn't present in `prices.columns` are SKIPPED —
    not fabricated as NOT_ENOUGH_DATA — so a caller can diff the returned
    frame's index against `industries` to see exactly what's missing,
    matching the honesty principle config/industries.py itself documents.

    Returns a DataFrame indexed by industry key with columns:
      label, parent_sector, ticker, state, above_ma, ma_value,
      rs_now, rs_prior, rs_slope, price
    Empty (but correctly-columned) frame if SPY isn't present or nothing
    in `industries` has a matching price column.
    """
    from config.settings import BENCHMARK  # local import: config, not IO

    industries = industries if industries is not None else INDUSTRIES
    if prices is None or prices.empty or BENCHMARK not in prices.columns:
        return pd.DataFrame(columns=_STATE_COLUMNS).set_index("industry")

    if as_of is not None:
        prices = prices.loc[:pd.Timestamp(as_of)]

    spy = prices[BENCHMARK]
    rows: list[dict] = []
    for key, ind in industries.items():
        if ind.ticker not in prices.columns:
            continue
        result = classify_industry_state(prices[ind.ticker], spy, params)
        rows.append({"industry": key, "label": ind.label,
                     "parent_sector": ind.parent_sector, "ticker": ind.ticker,
                     **result})

    if not rows:
        return pd.DataFrame(columns=_STATE_COLUMNS).set_index("industry")
    return pd.DataFrame(rows, columns=_STATE_COLUMNS).set_index("industry")


def sector_health(states: pd.Series) -> dict:
    """{'favorable': n, 'unfavorable': n, 'neutral': n, 'total': n}.

    Favorable = CLIMBING/BASE (above trend MA), Unfavorable = TIRED/DOWNHILL
    (below trend MA) — matches the "N Favorable / M Unfavorable / K
    Neutral" framing this feature is explicitly modeled on. NOT_ENOUGH_DATA
    counts as neutral (excluded from the favorable/unfavorable tally, never
    silently dropped from the total).
    """
    if states is None:
        states = pd.Series(dtype=object)
    n_total = int(len(states))
    n_fav = int(states.isin(FAVORABLE_STATES).sum())
    n_unfav = int(states.isin(UNFAVORABLE_STATES).sum())
    return {"favorable": n_fav, "unfavorable": n_unfav,
            "neutral": n_total - n_fav - n_unfav, "total": n_total}


def rotation_flow(prev_states: pd.Series, curr_states: pd.Series) -> pd.DataFrame:
    """Per-industry transitions between two snapshot dates.

    Returns a DataFrame indexed by industry with columns [from_state,
    to_state], containing ONLY industries whose state actually changed.
    Industries present in only one of the two inputs are excluded (nothing
    to diff against). Empty (correctly-columned) frame if there's no
    overlap or no changes.
    """
    empty = pd.DataFrame(columns=["from_state", "to_state"]).rename_axis("industry")
    if prev_states is None or curr_states is None:
        return empty
    common = prev_states.index.intersection(curr_states.index)
    if len(common) == 0:
        return empty
    df = pd.DataFrame({
        "from_state": prev_states.reindex(common),
        "to_state": curr_states.reindex(common),
    })
    df.index.name = "industry"
    return df[df["from_state"] != df["to_state"]]


def rotation_flow_summary(transitions: pd.DataFrame, top_n: int = 6) -> str:
    """'9 BASE → DOWNHILL · 4 CLIMBING → BASE · 3 TIRED → BASE · ...'

    Counts of each (from_state, to_state) pair, largest count first; ties
    broken alphabetically by (from_state, to_state) for a deterministic,
    testable ordering. Caps at `top_n` pairs so the line stays scannable —
    a caller wanting the full breakdown should render `transitions` itself
    as a table (the Dashboard tab does both).
    """
    if transitions is None or transitions.empty:
        return "No industry state changes since the last snapshot."
    counts = (transitions.groupby(["from_state", "to_state"]).size()
              .rename("n").reset_index()
              .sort_values(["n", "from_state", "to_state"],
                           ascending=[False, True, True]))
    parts = [f"{int(r['n'])} {r['from_state']} → {r['to_state']}"
             for _, r in counts.head(top_n).iterrows()]
    return " · ".join(parts)