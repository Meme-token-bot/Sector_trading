# Layout Changes — per-agent rationale

## Agent D — design tokens (branch: worktree-agent-a887a12735f582f05)

`src/ui_tokens.py` is the new chrome-layer token hub for the dashboard. It
imports `STATE_COLORS` and `STATE_ACCENTS` directly from `src.charts` and
re-exports them — `STATE_COLORS` stays in `charts.py` because the Plotly
figure builders reference it at module scope and `charts.py` must remain
Streamlit-free. `EXPRESSION_STATE_COLORS` (the seven self-check states for the
Expressions tab) has been moved out of `app.py` where it lived as an inline
dict and is now the canonical definition in `ui_tokens.py`, derived from the
sector palette so the same hue means the same thing on both the Dashboard and
Expressions tabs. The `render_header()` helper replaces the raw `st.title` +
`st.caption` block at app startup and injects `BASE_CSS` once per run to
tighten vertical rhythm. The `section()` helper wraps `st.subheader` +
`st.caption` into a single call and is now wired into every tab's opening
header; inner-tab section headings are left for Agents A/B/C.

### Public surface for downstream agents
- `from src.ui_tokens import section, render_header, STATE_COLORS, STATE_ACCENTS, EXPRESSION_STATE_COLORS`
- `from src.ui_tokens import GAP_TIGHT, GAP_DEFAULT, GAP_LARGE`  — spacing references (rem floats, not Streamlit gap= strings)
- `from src.ui_tokens import FS_H1, FS_H2, FS_H3, FS_CAPTION`  — font-size strings for inline markdown spans
- `from src.ui_tokens import inject_base_css`  — idempotent; safe to call from tab helpers
- `from src.ui_tokens import state_badge`  — returns `<span>` HTML for inline state labels in markdown

### Files changed
- `src/ui_tokens.py` — new file (design tokens + helpers)
- `app.py` — added import; replaced global title block with `render_header()`; replaced six tab-opening `st.subheader` + `st.caption` pairs with `section()`; removed inline `EXPRESSION_STATE_COLORS` dict (now imported)

## Agent B — Price Action + Expressions (branch: worktree-agent-ae9049e00fbe87ecb)

The Price Action control strip was reorganised from a flat four-column row (with
indicator checkboxes on a second row below) into two logical groups: a left
`[3, 2]` split whose left side has a nested `st.columns(3)` for Sector/Timeframe/
Lookback, and a right side with an `st.popover("Indicators")` collapsing the
three overlay checkboxes, a Compare-to-SPY checkbox, and the Update button — all
in one visual row. The sector grid's "View {tk}" buttons now use `type="primary"`
for NEW_BUY/HOLD_IF_LONG states and `type="secondary"` for all others, making the
grid a quick-scan state map without requiring chart-title inspection. `st.divider`
+ ad-hoc `##### ` heading above the grid were replaced with `section(level=3)`.

The Expressions tab's per-sector "How to read the Self-check column" expander (11
copies) was lifted out of the for-loop and placed once at the top of the tab.
The redundant `st.success("BUY signals: …")` green banner was dropped; the 🟢/⚪
expander-title prefix remains the sole BUY-class signal. The Note column is now
conditionally included: if all notes in a sector are empty it is omitted entirely,
reducing noise for unannotated sectors. The top control row was widened from
`[1, 4]` to `[1, 2, 2]` to add an "as-of" timestamp and a "Show only
BUY/HOLD_IF_LONG" toggle (`key="exp_show_only_buys"`).

The `_render_update_price_data_button(key, extra_clears)` helper (app.py line 45)
extracts the ~25-line duplicated update-button body. It always clears
`_cached_ohlcv` and `_cached_ohlcv_multi`; the Expressions call site passes
`extra_clears=[_cached_sector_sparklines, _cached_expression_signals]` to
replicate the additional invalidation that existed before. No cache TTLs, function
bodies, or session-state key names were changed.

Using `type="primary"/"secondary"` for the mini-grid buttons is sufficient because
the chart title color (rendered by `build_mini_chart`, which reads `STATE_COLORS`)
already carries the precise six-state semantic; the button type adds a coarse
actionable/non-actionable distinction without requiring CSS injection or HTML
workarounds inside `st.button`.

### Files changed
- `app.py` — `_render_update_price_data_button` helper (line 45); `with tab_price:` control strip + sector grid; `with tab_expressions:` top controls, single how-to-read expander, Note column gate, toggle filter
