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

## Agent A — Dashboard + Macro (branch: worktree-agent-a5f0aa188989fdfe2)

Dashboard tab: columns rebalanced from `[2, 1]` to `[3, 2]`; inner headings ("Sector Relative Strength Matrix", "Tiger Portfolio Drift") converted to `section(level=3)`; matrix rendered with explicit `column_config` (Sector=medium, numeric/state cols=small, Action=large); state-count metric strip moved immediately below the dataframe with a "State distribution" caption replacing the former orphaned position; target-weights expander auto-expands when `targets` is non-empty and heading clarified to "actionable allocation"; the vestigial "📡 Macro regime indicators have moved…" caption removed; force-refresh button relocated to a small footer diagnostics row at the bottom of the tab (button itself and `st.cache_data.clear()` call unchanged).

Macro tab: all eight indicators reflowed into 2-column `st.columns(2)` grids per cluster (Risk/Vol: 2+1, Growth/Cycle: 2, Rates/Inflation: 2+2); "Reading the panel together" guidance promoted from the bottom expander into a `st.info()` block immediately above the first cluster; remaining help text (bands caveats + "not yet wired" note) kept in the bottom expander, retitled "bands & caveats"; `st.divider()` between clusters replaced by `section(level=3)` calls; cluster captions folded into the `help=` argument of each `section()` call.

`_render_macro_indicator` compat note: a `compact: bool = False` parameter was added. All existing call sites that do not pass `compact` continue using the original `[1, 2]` side-by-side layout with 200 px charts — no behavior change. Only calls from the macro tab grid pass `compact=True`, which switches to a stacked layout with 160 px charts. The signature is keyword-only so no positional-argument break. The helper is defined inside `app.py`; it is not exported; B and C are not affected.

Orchestrator merge note: this branch was rebased onto `main` (after Agent D's merge) via a fast-forward `git merge main` before any edits. Only `app.py` was modified. No import lines were added — `section` was already imported by Agent D's pass.

### Files changed
- `app.py` — Dashboard tab body (~lines 109–267) and Macro tab body + `_render_macro_indicator` helper (~lines 329–630)
