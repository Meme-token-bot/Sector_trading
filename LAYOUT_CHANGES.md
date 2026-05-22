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
