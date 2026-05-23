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
