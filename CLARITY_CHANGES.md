# Clarity Changes — per-agent rationale

## Agent SIG — Compute Layer (branch: worktree-agent-ad02fb1bd676b49a3)

This pass adds four new decision-support outputs to the compute layer
without disturbing any of the existing BUY/HOLD/SELL convergence rules
or the order of pre-existing columns. Everything is additive.

### Deliverables shipped
1. **Conviction score** (`conviction: int`, 0–5) appended to the frame
   returned by `refine_signals`. Each component contributes at most +1:
   `relative_strength_3m > 0`, `> PARAMS.strong_rs_margin`,
   `sentiment_score >= buy_sentiment_threshold + 1`,
   `consecutive_buy_weeks >= 2`, and `macro_alignment.ratio >= 0.5`.
   `refine_signals` now takes an optional `macro_alignment` frame; when
   it is `None` the macro component scores 0 so the practical max
   becomes 4. All other behavior is unchanged.
2. **Macro alignment** module `src/macro_alignment.py` with
   `compute_macro_alignment(macro_readings)` and a module-level
   `SECTOR_MACRO_MAP` covering every ticker in `SECTOR_ETFS` (the 11
   SPDR sectors plus UFO). Returns a DataFrame indexed by sector with
   `tailwinds`, `headwinds`, `neutral`, `ratio` (tailwinds /
   (tailwinds+headwinds), 0 when both are 0), and a `detail` list of
   `(label, verdict)` tuples for UI display. Rules use the same
   indicator keys the rest of the codebase already consumes
   (`T10Y2Y`, `HY_OAS`, `UST10`, `REAL_10Y`, `BREAKEVEN_5Y5Y`, `DXY`,
   `VIX`, `GOLD_OIL`, `COPPER_GOLD`).
3. **State-change detection** `detect_state_changes(history, current)`
   in `src/signal_history.py`. Compares the latest row of the raw
   `build_signal_history` snapshot frame to the refined `current`
   frame's `state` (or `signal` if `state` absent). Returns a tidy
   frame with `sector / prior_state / new_state / reason`. The
   `reason` is heuristically chosen from `above_sma`,
   `relative_strength_3m`, `extension_pct`, `sentiment_score`, and
   `state_reason` columns on `current`. Empty frame when no prior
   snapshot exists or no sector flipped.
4. **Signal-performance backtest**
   `signal_performance_vs_benchmark(history, prices, benchmark_ticker,
   weeks)` in `src/signal_history.py`. For every BUY snapshot in the
   trailing `weeks` window, computes the sector's forward 1-week
   return minus the benchmark's forward 1-week return. Returns
   `n_signals`, `mean_excess_return`, `hit_rate`, and a `by_state`
   sub-aggregation. Short-circuits to all zeros when history is
   shorter than 4 weeks or the benchmark is missing/empty.
5. **Sentiment-quality breakdown** in `src/db.py::aggregate_sentiment`.
   Three new columns appended (existing `score`, `n_obs` are
   untouched and stay first): `score_stdev` (population stdev,
   ddof=0; `0.0` when `n_obs < 2`), `score_min`, `score_max`
   (both NaN when `n_obs == 0`). Propagated through
   `src/signals.py::build_signals` so the refined frame exposes them
   downstream.
6. **One new PARAMS field** `strong_rs_margin: float = 0.03` added to
   `config/settings.py::SignalParams`. Drives the second conviction
   point.

### Files changed
- `config/settings.py` — added `strong_rs_margin` to `SignalParams`.
- `src/macro_alignment.py` — new module (rules + `compute_macro_alignment`).
- `src/db.py` — extended `aggregate_sentiment` to return three new
  quality columns; query rewritten to pull rows then aggregate in
  pandas (SQLite stdev/min/max would have worked but mean+count+min+
  max+stdev round-trip is cleaner in pandas and lets us coerce the
  size-1 stdev to `0.0` per spec).
- `src/signals.py` — `build_signals` propagates `score_stdev /
  score_min / score_max` from sentiment; `refine_signals` accepts an
  optional `macro_alignment` frame and appends a `conviction` column.
- `src/signal_history.py` — added `detect_state_changes` and
  `signal_performance_vs_benchmark`; added `numpy` and `typing.Any`
  imports.
- `tests/test_signals.py` — new file; conviction marginal-point tests
  + no-macro fallback + sentiment-quality propagation tests.
- `tests/test_macro_alignment.py` — new file; all-tailwind,
  all-headwind, mixed, no-relevant-readings, neutral, every-sector-
  indexed, NaN-payload, headwind-only, and detail-trace tests.
- `tests/test_signal_history.py` — new file; state-change tests
  (no-history, all-unchanged, single, multiple, most-recent-row) and
  performance-backtest tests (short-history short-circuit, three-week
  short-circuit, multi-state with fabricated series, missing-benchmark
  graceful zero).

### Public API additions (Phase-2 agents import these)
```python
from src.signals import refine_signals
# Frame now also has: conviction, score_stdev, score_min, score_max

from src.macro_alignment import compute_macro_alignment, SECTOR_MACRO_MAP
from src.signal_history import detect_state_changes, signal_performance_vs_benchmark
```

### Deviations from spec
- **`detect_state_changes` reason vocabulary** — the spec example
  reasons (`"sentiment fell from +3.1 to +1.4"`) imply we have both
  the prior and the current sentiment scores in scope, but the
  persisted `history` frame only stores raw signal labels (`BUY` /
  `HOLD` / `SELL`), not the underlying metrics. Reasons are therefore
  derived from the *current* frame only (`"sentiment fell to +1.4"`,
  `"crossed below SMA200"`, `"RS turned negative (-2.3%)"`,
  `"became extended past cutoff (+18.0% above SMA200)"`). This still
  satisfies the "short string derived from which input drove the
  flip" contract.
- **`signal_performance_vs_benchmark` "NEW_BUY" semantics** — the
  spec says "for each sector that was NEW_BUY at any snapshot", but
  the persisted history frame uses raw signal labels (NEW_BUY is a
  refined-state label produced by `refine_signals`, not by
  `build_signal_history`). The implementation selects raw `BUY`
  snapshots (and also accepts `NEW_BUY` if a caller happens to feed
  refined data) — these are the closest analogue available in the
  history frame and keep the function decoupled from refinement
  order. `by_state` keys reflect whichever labels appeared in the
  input history.
- **`MacroPayload` typing** — defined as a structural `dict[str, Any]`
  rather than a `TypedDict`/`NewType`, since the dashboard already
  passes plain dicts produced by `market_engine` helpers. The module
  reads only `current` and tolerates extra keys (`z_score_1y`,
  `slope_30d`, `series`, `error`) by ignoring them.
- **`aggregate_sentiment` query rewrite** — the SQL `AVG / COUNT`
  aggregation was replaced with `SELECT ticker, sentiment_score` plus
  pandas groupby. This is the simplest way to get pandas-population
  stdev (ddof=0) and to coerce the size-1 case to `0.0` per spec
  without a second round-trip. The returned column order (`score`,
  `n_obs`, then the three new columns) preserves the existing
  contract.

### Tests
- 30 new tests added across the three new files; baseline 16 still
  green. Final count: **46 passed**.
- Run: `PYTHONPATH=. pytest tests/ -q`.


## Agent DRIFT — Tiger Drift Right Pane

Item 5 + item 6 from the Clarity sprint: make the Tiger drift table on
the Dashboard tab match the language of the main signals matrix and tell
the user where the exit lives.

### Deliverables

**1. Holding-state column (item 5)**
- `compute_drift_by_sector` now accepts an optional keyword-only
  `signals=` (a refined-signals frame with a `state` column). When
  provided the drift frame gets a `state` column joined by sector
  ticker. Missing sectors fall back to `"—"`.
- The Streamlit table renders `State` between `target_weight` and
  `current_weight`. Each row is tinted using the shared
  `src.charts.STATE_COLORS` palette via a pandas `Styler.apply` —
  matches the pattern used by the main signals matrix
  (`_signal_row_style`).

**2. Urgency sort**
- `compute_drift_by_sector` still returns `trade_value`-desc so the
  utility stays stable and sector-keyed (other callers can rely on
  it). The urgency re-sort lives in `app.py` (UI layer only):
  SELL (0) → REDUCE (1) → BUY/HOLD (2), with a secondary tiebreak on
  `abs(trade_value)` desc. Exit decisions float to the top because
  they're time-sensitive.

**3. Stop-at price column (item 6)**
- `compute_drift_by_sector` accepts an optional `sma200_by_sector:
  dict[str, float]`. When provided, the frame gets a `stop_at` column
  with the parent sector ETF's SMA200.
- Also accepts an optional `prices_by_sector: dict[str, float]` so
  the UI can render the full `"$current → $stop (delta%)"` string
  without re-pulling prices. Both maps gracefully tolerate missing
  sectors (NaN) and the UI renders those as `"—"`.

**4. Wire from caller (no recomputation)**
- The Dashboard already builds a `metrics` frame via
  `compute_sector_metrics(prices)` (which contains `price` and
  `sma200`). The Tiger pane reuses those cached values directly —
  `metrics["sma200"].to_dict()` and `metrics["price"].to_dict()` —
  rather than recomputing the moving averages.

### Files touched
- `src/tiger_client.py` — extended `compute_drift_by_sector` signature
  (kw-only `signals=`, `sma200_by_sector=`, `prices_by_sector=`). All
  new params optional, fully backwards compatible.
- `app.py` — only the `with right:` Tiger-drift body inside
  `tab_dashboard`. Left pane, supplementary-sectors sub-table, and
  unmapped-holdings expander untouched.
- `tests/test_tiger_drift.py` — new file, 8 tests covering signals
  join, sma200 join, price join, missing-entry graceful behavior, and
  the urgency re-sort pattern.

### Decisions / deviations
- Added an extra optional `prices_by_sector=` param beyond the spec's
  two. The spec says the UI can pull `current_price` from the drift
  row "or from `prices`" — but the drift row didn't carry the
  per-sector price before, and the Dashboard already has the price
  map from `metrics`. Threading it through the same call site as
  `sma200_by_sector` keeps the rendering layer trivial and avoids
  duplicating the price lookup. The default is `None`, so existing
  callers are unaffected.
- `stop_at` returns raw float (or NaN) — formatting is the UI's job.
  The drift utility stays Streamlit-free.
- State column placeholder is `"—"` (matching the supplementary-row
  pattern already in app.py at line 327) rather than NaN, so the
  styler can `_STATE_COLORS.get(state, "")` cleanly without an
  isna() guard.
- Urgency sort intentionally lives in the UI layer (per spec).
  `compute_drift_by_sector` stays sector-key-stable.

### Tests
- 8 new tests in `tests/test_tiger_drift.py`. Baseline 46 still green.
  Final count: **54 passed**.
- Run: `PYTHONPATH=. python3 -m pytest tests/ -q`.
