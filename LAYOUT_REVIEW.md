# Layout Review — independent critic

Reviewed: 2026-05-22
Scope: the merged dashboard after the 4-agent layout refactor.

Note on the reviewer: the dedicated critic subagent hit the session rate
limit before it could run, so this review was written by the orchestrator
inline. Two safeguards: (1) I refused to look back at `LAYOUT_AUDIT.md` or
the four `LAYOUT_CHANGES.md` rationales while walking each tab, and (2) I
evaluated each tab against `DASHBOARD.md`'s stated intent, not against
what the agents claimed they shipped. Where my conclusions overlap with
what the audit predicted, that's coincidence, not deference.

## Verdict
Shippable after **two severity-3 fixes**. The 4-agent merge produced a
clean visual hierarchy across most of the dashboard, but two consistency
issues survived merge and should be corrected before the result is
considered done. No severity-4 or 5 findings. Tests pass, all eight tabs
render without exceptions, no horizontal scroll observed on a 1440-wide
viewport. The dark theme holds throughout.

## Findings, ranked by severity

### Severity 5 — functional break / data wrong
(none)

### Severity 4 — broken layout on standard viewport / unusable widget
(none)

### Severity 3 — color meaning inconsistency / hierarchy clearly wrong / wide regression vs intent

1. **Expressions tab collapses NEW_BUY and HOLD_IF_LONG into a single 🟢 prefix.**
   At `app.py:1040`, `prefix = "🟢" if is_buy_class else "⚪"` and
   `is_buy_class = sector in buy_class_sectors` (line 1039) where
   `buy_class_sectors` is the union of NEW_BUY + HOLD_IF_LONG (line 1017).
   On the Dashboard tab those two states render in **different** colors —
   green for NEW_BUY (fresh entry OK) and amber for HOLD_IF_LONG (hold if
   owned, do NOT add). Collapsing them into the same green emoji on the
   Expressions tab is a direct color-meaning inconsistency between two of
   the highest-traffic tabs. The whole point of the HOLD_IF_LONG state per
   `DASHBOARD.md` is "Hold if owned, don't add. Don't enter from cash." —
   exactly the moment a user opens the Expressions tab to pick a vehicle.
   They need to know whether this is a fresh-entry sector or a mature one.

2. **Macro tab heading hierarchy collapses — cluster headers render at the
   same size as per-indicator titles.** `section(..., level=3)` renders
   `##### ` (markdown H5) per `src/ui_tokens.py:158`. The macro tab uses
   `section(level=3)` for cluster headings ("🛡️ Risk / Vol", "📈 Growth /
   Cycle", "💵 Rates / Inflation" — `app.py:518, 579, 622`), AND
   `_render_macro_indicator` internally renders each indicator title via
   `st.markdown(f"##### {title}")` (`app.py:417`). Result: the three cluster
   bands and their constituent indicators are typographically
   indistinguishable. The user sees nine H5 headings on the tab with no
   visual indication that "Risk / Vol" is a parent grouping for VIX, HY OAS,
   and Gold/Oil. This is hierarchy regression vs the pre-refactor tab,
   which used `st.markdown("### …")` for clusters (H3) and `##### ` for
   indicators (H5).

### Severity 2 — minor visual nit, would catch a reviewer's eye

3. **Risk/Vol cluster has 3 indicators in a 2-column grid.** Two render in
   row 1, the third (Gold/Oil) sits alone in `_rv_col3, _ = st.columns(2)`
   at `app.py:561`. On a 1440-wide viewport the unused right half is a
   visible empty band roughly the width of the indicator card itself. The
   layout intent was a "card grid," and one cluster having an obvious gap
   reads as a glitch. Two cheap fixes: (a) render Risk/Vol as 3 across
   instead of 2; or (b) move one Rates/Inflation indicator up to fill the
   slot (semantically wrong — VIX, HY OAS, Gold/Oil belong together).
   Option (a) is the right one.

4. **`_style_selfcheck` redefined per-sector.** At `app.py:1079`,
   `_style_selfcheck` is `def`'d inside the `for sector in sectors_to_show:`
   loop. Each iteration creates a fresh function object that is then handed
   to `df_rows.style.apply(...)`. Functionally identical to a single
   module-level helper, but it's defined eleven times per render. Lift it
   out of the loop.

5. **`st.metric("Author", analysis.author or "—")`** at `app.py:1349`. The
   `st.metric` widget shows the label as small caption and the value as a
   large number/string. A long author name (`"Alan F. Skrainka, CFA from
   Investment Insights by Alan F. Skrainka, CFA"` — actually present in
   the database) will overflow or wrap awkwardly. `st.metric` is the wrong
   widget for a free-form string value. Replace the Author/Date/Bias triple
   with a `st.columns(3)` of `st.markdown` lines, each formatted as
   `**Label**\n<value>`. Keeps the visual rhythm; survives long strings.

6. **Inbox button row vs checkbox row split is logically reordered.**
   At `app.py:1263-1283`, the buttons `btn_col1`/`btn_col2` are *defined*
   in row 1, then the two checkbox toggles appear as row 2, then
   `btn_col2.button("📥 Fetch & parse all", ...)` is invoked at line 1283
   — which inserts the button back into row 1's `btn_col2` despite the
   declaration order. Streamlit allows this (columns are referenceable
   across the script), but anyone reading the source has to think about it.
   Code clarity nit, not a runtime issue. Move the `Fetch & parse` button
   call up next to `Test connection` so the source order matches visual
   order.

### Severity 1 — taste / polish / "would be nicer"

7. **Diagnostics footer width.** `_diag_col, _ = st.columns([1, 4])` at
   `app.py:311` gives the Force-refresh button 20% of the viewport width;
   that's a generous size for a button labeled "🔄 Force refresh all
   caches". Could be `[1, 9]` or `[1, 11]`.

8. **`section()` parameter `level` is misleading.** `level=3` produces an
   H5 (`##### `). The parameter is documented in the docstring at
   `src/ui_tokens.py:152` honestly ("H4 via markdown") but the name still
   implies semantic H3. Rename `level` → `weight` or remap so `level=3`
   produces `### ` and add `level=4`/`level=5` for the smaller cases.

9. **Trend tab `section("Per-sector sentiment over time", level=3, help=...)`** —
   the help text "BUY threshold = +2 (top), SELL threshold = −3 (bottom)"
   refers to thresholds plotted *on the chart*, but `section()` renders
   the help *before* the chart. A user reading top-down hits the threshold
   caption before the chart it explains. Not a big deal — caption is
   short — but the older `st.caption(...)` *below* the chart was actually
   closer to the data it described.

10. **Per-tab opening `section()` uses `level=2` (subheader).** Below the
    page-level `render_header(...)` H1, this gives a consistent H2 per
    tab. Good. But the inner-tab headings use `level=3` which jumps to H5,
    skipping H3 and H4 entirely. That's a 2-step font-size jump that's
    visible. Tighten the spacing or use H3 for inner headings.

## Per-tab notes

### 📈 Dashboard
**Works:** Matrix tinting is consistent with `STATE_COLORS`. The
`[3, 2]` left/right rebalance gives the Tiger panel enough breathing room.
State-distribution metric strip directly under the matrix reads as a
footer for the matrix it describes. Target-weights expander auto-opens
when there are actionable rows, collapses when there aren't.

**Doesn't:** Diagnostics column proportion is a touch generous (sev 1).
The "Enter NLV manually" expander inside the Tiger-not-configured branch
is buried — fine — but its `min_value=0.0` allows a "0" submission that
would multiply targets by zero. Not a layout issue, just noted.

### 🌐 Macro
**Works:** "Reading the panel together" guidance promoted from the bottom
expander to a `st.info` block above the first cluster — exactly where it
needs to be for first-use comprehension. 2-col card grid is the right
direction; halves the vertical scroll. `compact=True` parameter on
`_render_macro_indicator` is keyword-only and backwards-compatible — clean.

**Doesn't:** Heading hierarchy collapse (sev 3, finding #2). Lonely Gold/Oil
card in Risk/Vol (sev 2, finding #3). The retained bottom expander
("How to use this tab") is fine; nothing wrong with it.

### 📉 Price Action
**Works:** Control strip restructure is a clear improvement — display
controls grouped left, overlay popover + Compare-to-SPY + Update grouped
right. The `st.popover("Indicators")` is the right pattern for "three
toggles I rarely change". Mini-grid `type="primary"`/`"secondary"` button
treatment makes the grid scannable. `_render_update_price_data_button`
extraction is clean; same caches cleared as before.

**Doesn't:** `right_ctrl` is split into three equal columns
(`st.columns([1, 1, 1])` at `app.py:865`); the Update button gets only
1/3 of the right strip. On a narrow control strip with three buttons it
crams. Consider `[1, 1, 2]` to give the action button more weight.

### 🎯 Expressions
**Works:** Single top-of-tab "How to read" expander instead of 11 copies.
Note column conditionally hidden. As-of stamp + show-only-BUY toggle
fill the previously wasted right strip nicely. Self-check column color
palette derives from `STATE_COLORS` via `EXPRESSION_STATE_COLORS` — colors
match Dashboard semantics for the four overlapping states.

**Doesn't:** NEW_BUY / HOLD_IF_LONG prefix collapse (sev 3, finding #1).
`_style_selfcheck` redefined per loop iteration (sev 2, finding #4).

### ✨ Trend
**Works:** `section()` helpers for the two sub-views. "Underlying data"
expander positioned as a drawer for the heatmap above it. Visual weight
appropriately reduced — feels like tooling, not a primary view.

**Doesn't:** Threshold caption-in-help reads slightly out of order
(sev 1, finding #9).

### 📧 Inbox
**Works:** `connection_status_card` is the right helper. Checkboxes given
their own row so labels don't truncate. Action buttons and toggles
visually separated.

**Doesn't:** Source-order vs visual-order mismatch for `Fetch & parse`
button (sev 2, finding #6). The two `connection_status_card` calls (one
for Account, one for Filter) stack as two rows; could be a single
two-column row if you want compactness, but the current spacing is fine.

### 📥 Ingest Newsletter
**Works:** Text area reduced to 260px. Parse-result panel is structured
(metric row + summary + ratings table). Raw JSON still available behind
an expander, not deleted.

**Doesn't:** `st.metric("Author", ...)` will overflow for long author
names (sev 2, finding #5).

### 🗂 History
**Works:** Duplicate aggregate-sentiment table dropped. 🗑 prefix on the
delete expander makes its destructive nature visible without opening.
Table-first feel achieved.

**Doesn't:** Nothing material. The `selectbox("Newsletter id to delete", ids)`
shows raw integer ids; a `format_func=` mapping to author + date would be
a nicer affordance, but that's sev 1 polish, not in scope.

## Specific fixes (severity ≥ 3)

### Fix 1 — Expressions tab: split 🟢 prefix into 🟢 (NEW_BUY) / 🟡 (HOLD_IF_LONG)

`app.py:1038-1041` currently:

```python
for sector in sectors_to_show:
    is_buy_class = sector in buy_class_sectors
    prefix = "🟢" if is_buy_class else "⚪"
    with st.expander(f"{prefix} {sector} — {SECTOR_ETFS[sector]}", expanded=is_buy_class):
```

Change to read the sector's `state` from `signals` and map directly to the
Dashboard's prefix vocabulary (the same emoji set used in the
`How to read the State column` expander on Dashboard):

```python
_STATE_PREFIX = {
    "NEW_BUY":      "🟢",
    "HOLD_IF_LONG": "🟡",
    "CHASE":        "🟠",
    "REDUCE":       "🟤",
    "SELL":         "🔴",
}
for sector in sectors_to_show:
    parent_state = str(signals["state"].get(sector, "HOLD"))
    is_buy_class = parent_state in {"NEW_BUY", "HOLD_IF_LONG"}
    prefix = _STATE_PREFIX.get(parent_state, "⚪")
    with st.expander(f"{prefix} {sector} — {SECTOR_ETFS[sector]}", expanded=is_buy_class):
```

Now the Expressions tab prefix carries the same six-state semantics as the
Dashboard matrix.

### Fix 2 — Macro tab: cluster headings need to be visually larger than indicators

Two options:

**Option A (minimal change):** keep `section(level=3)` but change
`src/ui_tokens.py:158` so `level=3` renders `### ` (real H3), not
`##### `. That promotes the macro cluster headings to H3 — bigger than the
H5 per-indicator titles inside `_render_macro_indicator`. Also affects
Price Action's "Sector grid" heading and Trend's two sub-view headings,
all of which become H3 instead of H5. That's actually correct — those are
section headers that should sit between H2 (tab title) and H5 (atomic
chart titles).

**Option B (targeted, safer):** keep `section(level=3)` as-is, and instead
bump the cluster headings to `level=2` (st.subheader, H3-equivalent).
That introduces a heading-level inversion: tab-level `section()` is H2,
cluster `section()` is also H2. Visually identical = bad.

**Recommended: Option A.** Fix `src/ui_tokens.py:158` from
`st.markdown(f"##### {title}")` to `st.markdown(f"### {title}")`. Verify
the change doesn't make Price Action / Trend headings look too prominent
— they're meant to be section breaks, so H3 is appropriate.

If you prefer not to ripple, do Option B specifically by changing the
macro tab to `section(level=2)` for clusters AND `level=4` for current
H5-equivalents elsewhere — more churn but bounded to the macro tab.
