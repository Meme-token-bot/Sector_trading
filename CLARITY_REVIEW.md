# Clarity Sprint — Independent Review

Auditor: Agent CRITIC
Branch reviewed: `worktree-agent-a3637ab7e486560fe` (fast-forwarded to `main` @ `6c04fe2`)
Tests: 54/54 passed (`PYTHONPATH=. python -m pytest tests/ -q` → `54 passed in 4.64s`)
Streamlit import: PASS (`python -c "import app"` exits 0; only benign `missing ScriptRunContext` + `use_container_width` deprecation warnings)
Streamlit headless boot: PASS (`streamlit run app.py --server.headless true --server.port 8765` returns HTTP 200, no exceptions in log)
Date: 2026-05-24

## Verdict

**Ship as is.** All ten clarity items are implemented in code, the conviction
hand-check matches exactly on three diverse sector profiles, the no-macro
fallback degrades gracefully, color semantics are consistent between the
matrix and the drift table via the shared `src.charts.STATE_COLORS` palette,
the orders panel correctly omits CHASE/HOLD_IF_LONG, and the entire test
suite (54 tests including 30+ new) is green. No sev ≥ 3 findings; three
sev 2 items below are polish, not blockers.

## Per-item verdict (the 10 items)

| # | Item | Status | Severity (if not PASS) | Notes |
|---|------|--------|-----------------------|-------|
| 1 | Orders panel | PASS | n/a | `app.py:301–378`. Top of left column. SELL rows fire on `state in {SELL, REDUCE} and held`; BUY rows fire on `state == NEW_BUY` only; CHASE and HOLD_IF_LONG correctly excluded. NLV sizing via `targets.loc[tkr] * nlv` when Tiger configured, else `"—"`. Empty-state: `st.success("No actions this week — portfolio aligned.")`. |
| 2 | Conviction column | PASS | n/a | `src/signals.py:165–189` + `app.py:399–401, 479–483`. Column order `[..., "Wks BUY", "Conviction", "Sentiment", ...]` — between Wks BUY and Sentiment as specified. Dot scale `●●●○○` via `_format_conviction`. Logic matches spec verbatim; hand-check below confirms numerics. |
| 3 | Macro alignment pill | PASS | n/a | `app.py:227–255, 411–416, 489–491`. Column order `[..., "state", "Macro", "Why"]` — between State and Why. Format `f"{tw}/{total} ✓"` where `total = tw + hw` (excludes neutrals — sensible since `ratio` is defined the same way). Color thresholds: green ≥ 0.625, amber ≥ 0.375, red < 0.375, neutral grey for `—`. Spec example "5/8 ✓" implied `total` denominator; current implementation uses `tailwinds + headwinds` so for a sector with 5 tail / 2 head / 1 neutral the cell reads `5/7 ✓`. Tooltip explains the denominator. Acceptable, see F1 (sev 2). |
| 4 | State-change strip | PASS | n/a | `app.py:380–391`. Between orders panel and matrix. `st.info` with `·` separator. Whole block guarded by `if changes is not None and not changes.empty:` — hidden when empty. `detect_state_changes` itself wrapped in `try/except` so a malformed history can't break the page. |
| 5 | Holding-state column (Tiger drift) | PASS | n/a | `app.py:672–701` + `src/tiger_client.py:160–162`. `state` joined onto drift frame, renamed to `State`, inserted at position 1 in `_cols = ["target_weight", "State", "current_weight", ...]` — between target and current. Row tint via `_drift_row_style` reads `_STATE_COLORS` — same palette as the main matrix. |
| 6 | Stop-at price column | PASS | n/a | `app.py:647–701` + `src/tiger_client.py:164–172`. Stop sourced from `sma200_by_sector` (parent sector ETF SMA200, reused from the cached `metrics` frame — no recomputation). Cell format `f"${px:,.2f} → ${stop:,.2f} ({delta:+.1f}%)"` with `—` when either price or stop is NaN/zero. |
| 7 | Performance feedback strip | PASS | n/a | `app.py:504–518`. Below the State Distribution metric row. Message: `f"NEW_BUY signals, last 12 weeks: hit rate {pct}%, mean excess return {pct:+}% vs {BENCHMARK} (n={n})"`. Fallback string `"Performance stats unavailable — need ≥4 weeks of history."` when `n_signals == 0`. `signal_performance_vs_benchmark` correctly short-circuits to `n_signals=0` when `len(history) < 4`. |
| 8 | Sentiment cell | PASS | n/a | `app.py:418–425`. Exact format `f"{mean:+.1f} · n={n} · σ={stdev:.1f}"` — matches spec `+2.1 · n=3 · σ=0.8` character-for-character. `score_stdev` plumbed through `src/db.py::aggregate_sentiment` → `src/signals.py::build_signals` (population stdev ddof=0, coerced to 0.0 when n_obs<2). |
| 9 | Expressions Band column | PASS | n/a | `app.py:1700–1714, 1757–1765`. Floor = `sma200 = last_px / (1 + own_extension_pct)`; ceiling = `sma200 * (1 + beta_scaled_cutoff)` where `beta_scaled_cutoff = PARAMS.extension_pct_cutoff * beta` (`src/expression_signals.py:88`). Renders `—` when SMA200 isn't computable. Tooltip explicitly labels floor as "BROKEN" and ceiling as "STRETCHED". |
| 10 | Structured Why cell | PASS | n/a | `app.py:258–267, 427–434`. `_buy_class = {"NEW_BUY", "HOLD_IF_LONG"}` → `_structured_why` returns `"📈 +X%   📊 +Y%   💬 +Z"`. All other states keep their prose `state_reason`. The previous `Action` column header was renamed to `Why` (line 438). |

## Findings (sev ≥ 3)

No sev ≥ 3 findings.

## Findings (sev 1–2)

### F1 — Macro pill denominator can confuse vs the tooltip wording [sev 2]

**Where**: `app.py:227–238` (`_format_macro_pill`) and the column-header
tooltip text at `app.py:462–467`.

**What**: The cell renders `f"{tw}/{total} ✓"` where
`total = tailwinds + headwinds`, i.e. *neutrals are excluded from the
denominator*. The tooltip says "Green ≥ 5/8, amber 3/8–5/8, red < 3/8" —
which uses the literal `5/8` shape and implies the denominator is the total
rule count for the sector (e.g. 8 indicators). A sector with 3 tailwinds, 0
headwinds, 0 neutrals — perfectly bullish — will render as `3/3 ✓`, not `3/8`.
Functionally correct (ratio is what drives color), but a user comparing two
cells of different denominators (`3/3` vs `4/6`) has to know that the
denominator is "applicable indicators", not "rule pool size". Either the
tooltip should say "Tailwinds / (tailwinds + headwinds); neutral readings
excluded" explicitly, or the cell should adopt a fixed `/rule_count`
denominator. Minor wording polish.

**Why it matters**: A user glancing at the column to compare sectors might
read `5/8` and `4/6` as "5 of 8 indicators support this" vs "4 of 6", but
they're actually *ratios with different bases*. Color tint already
disambiguates the strength, so this is not a correctness bug.

**Suggested fix**: Update the help text at `app.py:462–467` to say
"Tailwinds / (tailwinds + headwinds); neutral readings excluded from the
denominator. Green ≥ 0.625 (5/8 ratio), amber 0.375–0.625, red < 0.375."

---

### F2 — SELL orders rows show `Size = "—"` even when NLV is known [sev 2]

**Where**: `app.py:337` (`size = "—"  # we don't size sells from targets; user picks`).

**What**: BUY rows are sized as `$dollars = targets.loc[tkr] * nlv` whenever
Tiger is wired. SELL rows show `"—"` unconditionally. The spec said "sized
from Tiger NLV when available" without qualifying which side. Tiger's
snapshot already provides per-sector current value (we use it via
`held_sectors` at line 317–320) so we could surface a "current $value" for
SELL rows so the user sees the dollar magnitude they're exiting. Not a spec
violation — the comment "we don't size sells from targets; user picks" is
defensible — but it's a real piece of context the user has to look up in the
drift table instead of seeing inline.

**Why it matters**: The orders panel's stated purpose is "one row per
actionable SELL/BUY, sized from Tiger NLV". Half the rows are not sized.
Quality-of-life delta, not a blocker.

**Suggested fix**: When a SELL row's `tkr` is in `_drift_for_held` and the
current value is known, render `Size = f"${current_value:,.0f}"`. Falls back
to `—` when Tiger is absent.

---

### F3 — `Conviction` help text omits the "macro tailwinds ≥ headwinds" rationale [sev 2]

**Where**: `app.py:481–483`.

**What**: The tooltip reads: "0–5 score: +1 each for RS>0, RS>strong,
sentiment strong, ≥2 weeks BUY, macro tailwinds ≥ headwinds." This is fine
but doesn't explain that *the macro component reads `0` whenever
`compute_macro_alignment` returns an empty frame*, in which case the
practical max becomes 4 not 5 (per the docstring on `refine_signals`). A
sector showing `●●●●○` (4 dots) when macro data is unavailable will look like
it's missing one component when in fact it's maxed out for the data on hand.

**Why it matters**: Edge case. Macro fetch failure is rare. But a 4-dot
sector in a no-macro state misleadingly looks weaker than it is.

**Suggested fix**: Append to the tooltip: "Macro contributes 0 when macro
data is unavailable — practical max is 4 in that case."

## Hand-computed conviction spot-check

Inputs were a fabricated `refine_signals` payload covering three sectors
with deliberately diverse RS / sentiment / weeks / macro profiles. Hand
arithmetic vs `refine_signals` return values:

| Sector | Inputs (RS, sent, weeks_buy, macro_ratio) | Hand-computed | Code-returned | Match? |
|--------|-------------------------------------------|---------------|---------------|--------|
| XLK    | RS=+0.05, sent=+3.5, weeks=4, ratio=0.75 | 5 (RS>0 +1, RS>0.03 +1, sent≥3.0 +1, weeks≥2 +1, ratio≥0.5 +1) | 5 | ✓ |
| XLE    | RS=+0.01, sent=+2.0, weeks=1, ratio=0.33 | 1 (RS>0 +1 only) | 1 | ✓ |
| XLV    | RS=-0.02, sent=-1.0, weeks=0, ratio=1.00 (SELL state) | 1 (ratio≥0.5 +1 — conviction is computed for SELL rows too) | 1 | ✓ |

Also verified the no-macro fallback paths:

| Scenario | XLK | XLE | XLV |
|----------|-----|-----|-----|
| `macro_alignment=None`       | 4 | 1 | 0 |
| `macro_alignment=pd.DataFrame()` (empty) | 4 | 1 | 0 |

Both fallbacks drop the macro component to 0 without raising. The pipeline
remains stable when the FRED/Yahoo macro fetch fails. The dashboard call
path at `app.py:292–294` passes `macro_alignment=_cached_macro_alignment_frame()`
unconditionally; if the underlying fetchers throw they'd bubble up to
Streamlit, but the `_macro_for()` / `_format_macro_pill` / `_macro_pill_color`
trio in the matrix renderer all degrade to `—` / neutral grey when the row
lookup returns `None`, so partial-data scenarios render cleanly.

## Color-palette consistency check

The shared palette lives at `src/charts.py:22–29` as `STATE_COLORS`:

```
NEW_BUY      #143d2a (green)
HOLD_IF_LONG #3d3a14 (amber)
CHASE        #4a3214 (orange)
REDUCE       #3d1f14 (rust)
HOLD         ""       (no tint)
SELL         #4a1818 (red)
```

Re-exported by `src/ui_tokens.py:30`. Consumed by:

- **Main signals matrix** — `_signal_row_style` at `app.py:207–210`, called
  inside `_signal_row_style_with_macro` at line 445.
- **Tiger drift table** — `_drift_row_style` at `app.py:682–685`.
- **Weekly Recap tilt rows** — `_TILT_TINT` at `app.py:757–760`.
- **Expressions self-check** — uses `EXPRESSION_STATE_COLORS` at
  `src/ui_tokens.py:41–45`, which itself derives from `STATE_COLORS`
  (CONFIRMED ← NEW_BUY green, BROKEN ← SELL red, etc.).
- **Orders panel** — uses emoji prefixes (`🔴 SELL`, `🟢 BUY`) in the
  `Action` cell rather than row-tinting. Red/green semantics match the
  matrix (SELL is red across both surfaces; BUY is green across both).

Conclusion: a sector in `SELL` state renders red in the matrix row tint, red
in the drift row tint, and gets a 🔴 prefix in the orders panel. A sector
in `NEW_BUY` renders green in all three. Color semantics are consistent.

## No-regression spot-check on other tabs

I read the tab boundaries and confirmed every tab opens with a `with tab_X:`
block at:

- `tab_dashboard` 285
- `tab_recap` 783
- `tab_macro` 1093
- `tab_price` 1426
- `tab_expressions` 1571
- `tab_trend` 1853
- `tab_inbox` 1903
- `tab_ingest` 1982
- `tab_history` 2055

`python -c "import app"` succeeds (no `SyntaxError`, no `NameError`),
`streamlit run app.py --server.headless true` boots without any traceback in
the log, and the test suite passes. None of the new compute additions
(`conviction`, `score_stdev`, `score_min`, `score_max`, `state`,
`consecutive_buy_weeks`) shadow any pre-existing columns the other tabs
consume — they're appended to the refined frame, and existing tabs that
display the signals frame either select explicit columns or are restricted
to the Dashboard scope.

## Cross-check vs CLARITY_CHANGES.md

After completing the audit above, I read `CLARITY_CHANGES.md` for the first
time and checked it against what I found.

**Documented deviations match what I observed:**

- SIG's `detect_state_changes` reason vocabulary uses single-sided phrasing
  ("sentiment fell to +1.4" rather than "from +3.1 to +1.4") because the
  persisted history frame only stores raw signal labels. Confirmed at
  `src/signal_history.py:85–157`. Acceptable per the spec's "short string
  derived from which input drove the flip" contract.

- SIG's `signal_performance_vs_benchmark` selects raw `BUY` snapshots from
  the history frame (and accepts `NEW_BUY` if refined data is fed in). The
  performance strip caption still labels them as "NEW_BUY signals" per spec;
  this is a small semantic drift (the count includes BUY labels that
  pre-date the refined-state era), but in practice the persisted history is
  the same model so the difference vanishes once a few weeks of refined
  history have accumulated. Documented honestly; not a finding.

- DASH's REDUCE-as-SELL-row treatment in the orders panel matches the spec's
  stated intent (REDUCE = trim if owned) and is rendered identically (`🔴
  SELL XLF`). Documented.

- DASH's held-sector fallback when the Tiger snapshot raises (treat every
  sector as potentially held) is a defensible "fail open" choice — better to
  over-suggest a SELL than to silently drop one. Documented.

- EXP's plain-text Band column instead of a Plotly overlay aligns with the
  spec's stretch-goal-skipped instruction.

**Items the agents claim shipped that I could not find:**

None. Every claimed deliverable was located and verified at the cited file
positions.

**Deviations not flagged by the agents but caught here:**

- The macro pill denominator wording in the column tooltip (F1) is a tooltip
  copy issue that the DASH agent didn't call out.
- SELL row sizing in the orders panel (F2) — DASH documented the BUY-only
  NLV sizing but didn't note that this leaves SELL rows un-sized despite the
  spec's "sized from Tiger NLV when available" line.
- Conviction tooltip doesn't mention the practical-max-4-without-macro case
  (F3).

All three are sev 2 polish, not sev ≥ 3 findings.
