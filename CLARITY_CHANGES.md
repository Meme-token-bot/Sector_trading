# Clarity sprint — Phase 2 agent changes

Tracks UI clarity improvements made on top of the layout/refactor work.

## Agent EXP — Expressions Tab Sparkline

### Deliverables shipped

1. **Entry-zone Band column (item 9)** — added a new `Band` column to every
   per-sector expressions DataFrame. Renders as `"$28.40 → $34.12"` where:
   - **Left (BROKEN floor)** = SMA200 of the expression.
   - **Right (STRETCHED ceiling)** = `SMA200 × (1 + PARAMS.extension_pct_cutoff × beta_hint)`.
   - Renders `"—"` whenever SMA200 isn't computable (NO_DATA / WARMING_UP
     states, i.e. fewer than `PARAMS.sma_window` stored bars).

2. **"How to read" expander updated** — the single top-of-tab expander is now
   titled *"How to read the Band and Self-check columns"* and opens with an
   explicit Band section that names the same numbers the Self-check uses
   (BROKEN floor = SMA200; STRETCHED ceiling = `SMA200 × (1 + cutoff% × β)`).
   The Self-check legend underneath is preserved verbatim so the original
   convergence story still reads cleanly.

3. **Stretch goal — skipped** as instructed. Plain-text Band column is the
   primary deliverable and gives the user the exact two reference prices
   without any Plotly render cost.

### Files touched

- `app.py` — Expressions tab only.
  - `_cached_expression_signals(...)`: additive — surfaces
    `own_extension_pct` and `beta_scaled_cutoff` on each cached dict so the
    Band can be derived without a second DB pass. Schema is additive; the
    existing `ticker / state / reason` keys are unchanged.
  - Expressions tab row builder: builds `Band` from
    `sma200 = last_close / (1 + own_extension_pct)` and
    `ceiling = sma200 × (1 + beta_scaled_cutoff)`, inserts it into
    `column_order` between `60d` and `Self-check`, and adds a matching
    `st.column_config.TextColumn` entry with a tooltip pointing back to the
    Self-check rules.
  - Top-of-tab expander markdown rewritten as described above.
- `CLARITY_CHANGES.md` — this file (created).

Not touched: `src/expression_signals.py`, `src/charts.py`,
`config/expressions.py`, `config/settings.py`, any other tab.

### Key decisions

- **Did not extend `ExpressionSignal`.** The dataclass already carries
  `own_extension_pct` and `beta_scaled_cutoff`, and the last close is already
  in `_cached_sector_sparklines`. SMA200 back-derives algebraically as
  `last / (1 + ext)` — no second DB pass, no new field on the frozen
  dataclass, no churn on `tests/test_expression_signals.py`. The pure-signal
  module stays untouched.
- **Plain text, not Plotly overlay.** `st.column_config.LineChartColumn`
  doesn't accept overlays, and replacing the column with per-row Plotly
  figures would add real render latency for what is purely a reference
  range. Two numbers in monospace-friendly text communicate the same fact
  with zero overhead.
- **Tooltip mirrors the Self-check rules.** The column tooltip explicitly
  ties "below the floor" to BROKEN and "above the ceiling" to STRETCHED so
  the user can read the column in isolation without scrolling up to the
  expander.

### Deviations from the prompt

- The prompt says CLARITY_CHANGES.md "already has SIG's section." It does
  not exist in this base — git log shows no SIG commit on `main` either.
  Created the file with this section as the first entry; if a SIG section
  arrives later, it can be inserted above without conflict.
- No changes were needed to `src/expression_signals.py` or `src/charts.py`
  (both explicitly optional in the prompt).
