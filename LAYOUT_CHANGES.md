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

## Agent C — Trend + Inbox + Ingest + History (branch: worktree-agent-a52f706acc18c8c33)

**Trend:** The two inner `st.markdown("##### …")` headings replaced with `section(..., level=3)` calls; the BUY/SELL threshold caption moved into the first section's `help=` parameter (sits next to the heading, no longer floating below the chart); the "Raw weekly snapshots" expander repositioned immediately after the heatmap and relabeled "Underlying data" so it reads as a data drawer for the heatmap above it.

**Inbox:** Account/Filter pair now rendered via `connection_status_card()` instead of bare markdown lines, providing consistent label/monospace layout. The two checkbox toggles (`Follow whitelisted links / PDFs` and `Mark messages as read`) were extracted from the cramped third column of the button row into their own `st.columns(2)` row immediately below the buttons — labels are no longer truncated.

**Ingest:** Text-area height reduced from 380 to 260. The parse result is now a structured panel: a 3-column `st.metric` row for author, publication date, and macro bias; the summary field as plain markdown prose; sector ratings as a `st.dataframe` with ticker/score/reasoning columns. The raw JSON is still accessible behind a "Show raw JSON" expander — the JSON path is not deleted.

**History:** The duplicate "Current Aggregate Sentiment" table (already shown on the Trend tab) and the `st.divider()` preceding it have been dropped entirely. Recent-newsletters table height reduced from 400 to 320. The delete expander is now labeled "🗑 Delete an entry" so its destructive nature is visible without opening it. All three `_cached_*.clear()` call sites are preserved verbatim.

**Orchestrator notes:** `connection_status_card` is defined at lines 1024–1043 in `app.py`, inserted just above `with tab_trend:` (line 1046). This is squarely within Agent C's scope and below the global-helpers band (which ends before line ~110). Agents A and B working on earlier tabs should not have any edits near line 1024. The only file modified is `app.py`.

### Files changed
- `app.py` — `connection_status_card` helper added (lines 1024–1043); `tab_trend`, `tab_inbox`, `tab_ingest`, `tab_history` blocks refactored per scope
