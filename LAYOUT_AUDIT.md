# Layout Audit — `app.py`

Read first: `DASHBOARD.md` (intent of each tab). This audit is the brief Phase-2
specialist agents work from. **Layout only — no signal / data / cache changes.**

---

## Cross-cutting observations (drives Agent D)

These patterns repeat across every tab and should be consolidated before
specialist agents touch their scopes:

1. **No shared section helper.** Every tab opens with `st.subheader(...)` +
   `st.caption(...)` and tabs use ad-hoc `st.markdown("### …")` for inner
   sections (Macro tab) or `st.markdown("##### …")` (Trend / Expressions /
   inner cards). Hierarchy is inconsistent — H3, H5, "subheader", and "##### "
   are all used interchangeably.
2. **Two parallel color palettes.** `src/charts.py::STATE_COLORS` + `STATE_ACCENTS`
   for sector states; a separate `EXPRESSION_STATE_COLORS` dict inline at
   `app.py:626` for expression states. Same idea, different home. Should live
   in a single `src/ui_tokens.py` module that **extends** the chart palette
   (chart figures must keep working unchanged).
3. **Divider abuse.** `st.divider()` is used both as a section break inside a
   tab (Macro, Price Action, History) and not at all in others (Dashboard,
   Trend, Inbox, Ingest). Inconsistent.
4. **No global header / branding.** `st.title` + a single sentence caption at
   the very top of `app.py` (lines 92-97). Tabs do their own intro block.
   Tab subtitles repeat the global context unnecessarily.
5. **No custom CSS.** No `unsafe_allow_html=True` anywhere. Safe to add a
   tiny CSS block via `st.markdown(..., unsafe_allow_html=True)` from a token
   helper if needed — Streamlit default density is loose and the dashboard
   would benefit from tightened vertical rhythm.
6. **Inconsistent "Update price data" button.** Appears in both Price Action
   and Expressions tabs (`app.py:725` and `app.py:847`) with near-identical
   ~25-line bodies. Should be one helper.

---

## Tab 1 — 📈 Dashboard (`app.py:106-243`)

### What it shows / primary action
The sector relative-strength matrix (one row per sector, state-tinted) is
**the** primary view. Secondary: state-count metrics row, target-weights
expander, Tiger drift panel (right column).

### Hierarchy issues
- **Primary table is fine** — state tinting works, `_signal_row_style`
  pulls from `STATE_COLORS`. ✓
- **State-count metric row** (6 columns: NEW_BUY/HOLD_IF_LONG/CHASE/REDUCE/HOLD/SELL)
  sits below the matrix with no separator. Visually orphans from the table
  yet logically belongs to it. Looks like a footer; reads like another section.
- **Target-weights expander** is collapsed by default. It's the actionable
  output of the model — should be the second-most-prominent element after the
  matrix, not buried.
- **"How to read the State column" expander** is in the right place but uses
  inline f-strings with config values (`PARAMS.extension_pct_cutoff*100`),
  which is fine, but the markdown formatting is dense — bullets run long.

### Spacing / proportions
- Left/right `[2, 1]` columns: with Tiger configured, right column gets
  4 metric blocks + drift table + (possibly) unmapped holdings expander.
  That's ~ as much vertical content as the left column has. The 2:1 ratio
  forces the right side to wrap awkwardly. Try `[3, 2]` or `[5, 4]`.
- On a 1440-wide viewport with `layout="wide"`, the matrix has plenty of
  horizontal room but the column widths inside it are streamlit defaults
  (auto-sized). `Action` (state_reason) column gets very wide and wraps;
  `Wks BUY` is wider than it needs to be.

### Inconsistencies
- "📡 Macro regime indicators have moved to the **🌐 Macro** tab." (line 184)
  is a stranded caption — vestigial signpost from a refactor. Either delete
  or move to a single FAQ/help block.
- "🔄 Force refresh all caches" sits naked at the bottom of the tab.
  Belongs in a small toolbar or expander, not as a floating destructive
  button.

### Agent A action items
- Adopt `section()` helper everywhere.
- Pull state-count metrics into a strip directly above (or attached to) the
  matrix — they describe the matrix, they shouldn't read like a separate
  block.
- Expand the target-weights expander by default when there are NEW_BUY /
  HOLD_IF_LONG rows; collapse otherwise.
- Re-balance left/right columns to `[3, 2]`.
- Delete the stranded "macro moved" caption (line 184).
- Tighten the matrix column widths via `column_config`: short headers get
  small widths, `Action` gets `large`.
- Move "Force refresh all caches" into a small footer row (with the
  generated-at-timestamp if it exists).

---

## Tab 2 — 🌐 Macro (`app.py:375-567`)

### What it shows / primary action
Eight macro indicators, each rendered by `_render_macro_indicator` in three
clusters: Risk/Vol, Growth/Cycle, Rates/Inflation. Each indicator is a
full-width block with a metric + bands + 1y line chart.

### Hierarchy issues
- The three cluster headings use `st.markdown("### 🛡️ Risk / Vol")` etc.
  — fine, but the **eight indicator blocks below them are also full width**.
  Result: an extremely tall, scrolling page. On 1440×900, you see 1–2
  indicators per viewport. The user has to scroll a lot to compare.
- The "Reading the panel together" guidance at the bottom is critical
  context for using the tab — it tells you to look for *agreement* across
  multiple indicators — but it's buried in a collapsed expander at the
  very bottom that most users will never open.

### Spacing / proportions
- Each indicator uses `st.columns([1, 2])` (metric+bands on left, chart on
  right) — that's actually well-chosen.
- Cluster section is divided by `st.divider()` + an H3 markdown heading.
  Combined effect is heavy vertical separation between clusters that already
  have visual identity from the emoji header.
- 200-pixel chart height (`payload["series"].tail(252)`, `height=200` in
  `_render_macro_indicator`) is small; OK for a quick read but the y-axis
  often hides regime-band thresholds.

### Inconsistencies
- The three cluster headings use `### ` (H3), but the per-indicator title
  inside `_render_macro_indicator` uses `##### ` (H5). Two-level jump
  with nothing at H4.
- "All signals are *displayed*, not yet wired into `build_signals()`" — an
  honest caveat, but it's in the bottom-of-tab expander where it'll never
  be seen at first use.

### Agent A action items
- Re-flow each cluster as a 2-column **grid of compact indicator cards**
  rather than 8 stacked full-width blocks. On 1440-wide, 2 columns gives
  ~700px per card — enough for a 240px chart + 80px metric+bands block.
- Reduce chart height to ~160px so a cluster of 3 indicators fits on one
  screen.
- Replace `### ` cluster headings with the new `section()` helper.
- Move the "Reading the panel together" guidance to a sticky-feeling caption
  *above* the first cluster, not a bottom-of-page expander. Keep the
  "Bands are calibrated to post-GFC norms" caveat as the expander.
- Drop redundant `st.divider()` between clusters (the section header is
  enough visual break).

---

## Tab 3 — 📉 Price Action (`app.py:680-828`)

### What it shows / primary action
A large candle chart with optional overlays, driven by a control bar above
it. A sector grid of 11 mini-charts below acts as the picker.

### Hierarchy issues
- **The control bar is the worst single piece of layout in the app.**
  `st.columns([1.3, 1, 1.2, 1.2])` packs: (1) Sector selectbox, (2) Timeframe
  radio, (3) Lookback radio, (4) "Compare to SPY" checkbox **AND** "🔄 Update
  price data" button stacked vertically in one cell. The Update button is a
  global-state mutation; it does not belong next to a display preference
  checkbox. The indicator checkboxes (RSI/MACD/BB) then live on a **separate
  row below the toolbar** with no visual grouping.
- Sector grid headline: `st.markdown("##### Sector grid — click a ticker button to load it above")` — H5, ad-hoc.

### Spacing / proportions
- Toolbar columns at `[1.3, 1, 1.2, 1.2]` are nearly equal, but the actual
  widget widths inside them differ wildly — the radio buttons compress, the
  selectbox expands, the Update button is taller than the checkboxes it
  sits next to. Awkward.
- Mini-chart grid is 3×4. Mini charts are mid-sized; on 1440 each gets
  ~430px wide which is comfortable. Button styling under each chart is
  default streamlit (full-width); fine, but the State color isn't visually
  applied to the button — only the chart's title color carries state, which
  is easy to miss.

### Inconsistencies
- "🔄 Update price data" is duplicated in the Expressions tab (~25 lines
  each). Should be a shared helper.
- The Update button uses default `st.button` styling; the **Tiger Test
  Connection** button in Inbox uses default; the **Parse & Save** in Ingest
  uses `type="primary"`. No consistency on which actions get `type="primary"`.
- `key=f"pa_mini_{tk}"` for plotly charts and `key=f"pa_mini_btn_{tk}"` for
  buttons — fine; just noting the naming convention should propagate to
  any new helpers.

### Agent B action items
- Reorganize controls into **one clear control strip**, two logical groups:
  - **Left group** (display state): Sector selectbox · Timeframe radio · Lookback radio
  - **Right group** (overlays + actions): Compare-to-SPY checkbox · RSI / MACD / BB checkboxes (collapse into a single popover or expander labeled "Indicators") · Update-price-data button (right-aligned, clearly an action not a toggle)
- Extract the "Update price data" body into a helper `_update_price_data_button(key, on_success_invalidations=[...])` and call from both tabs.
- Apply state tint to the **View {tk} button** under each mini-chart (not just the chart title) so the grid reads as a state map at a glance.
- Use `section()` helper for "Sector grid".

---

## Tab 4 — 🎯 Expressions (`app.py:831-1017`)

### What it shows / primary action
Per-sector expanders containing one expressions table (with the new
Self-check columns) and a "View full chart for…" click-through.

### Hierarchy issues
- **Lots of redundant chrome.** Each sector renders:
  expander → caption (sometimes) → table → "How to read…" expander → selectbox → chart.
  Across 11 sectors × N expressions that's a lot of vertical real estate.
- The "How to read the Self-check column" expander is per-sector (one per
  expander). Should be **once**, at the top of the tab, not repeated per
  sector. Currently the user sees it 11 times.
- BUY sectors are surfaced via two paths: the `st.success("BUY signals: …")`
  banner at top AND the 🟢 prefix in the expander title. Belt-and-suspenders.

### Spacing / proportions
- "Update price data" button is in `st.columns([1, 4])` — gives it ~20% of
  width on the left, leaves a big empty strip on the right. That space
  could carry an "as-of" timestamp + a sector filter (Show only BUY / Show
  all).
- Expression table columns: `Ticker · Label · Kind · β hint · 60d · Self-check · Self-check reason · Note`. 8 columns; `Note` is empty for most rows. Could be folded into a tooltip or only shown when non-empty.

### Inconsistencies (cross-tab color meaning — Agent B core deliverable)
- Sector states use the `STATE_COLORS` palette (green/amber/orange/rust/red).
- Expression states use a parallel palette (`EXPRESSION_STATE_COLORS` in
  `app.py:626`) where:
  - `CONFIRMED` reuses the NEW_BUY green ✓
  - `LAGGING` reuses HOLD_IF_LONG amber ✓
  - `STRETCHED` reuses CHASE orange ✓
  - `BROKEN` reuses SELL red ✓
  - `WARMING_UP` is grey (new)
  - `PARENT_INACTIVE` is faint grey (new)
  - `NO_DATA` is faint red (new)
  
  This alignment is good but should be enforced by **deriving** the expression
  colors from the chart palette in `ui_tokens.py`, not by happenstance.

### Agent B action items
- Single "How to read" expander at the top of the tab.
- Drop one of the two BUY-surfacing paths (recommend: keep the 🟢 prefix,
  drop the green banner).
- Fold `Note` into a row tooltip (`help=` on the Ticker cell) or only render
  when there's content.
- Use the new `ui_tokens.EXPRESSION_STATE_COLORS` (extends `STATE_COLORS`).
- Add a "Show only BUY/HOLD_IF_LONG" toggle in the unused right side of the
  top control row.

---

## Tab 5 — ✨ Trend (`app.py:1019-1061`)

### What it shows / primary action
Line chart of per-sector sentiment over time + a sectors×weeks heatmap.
Diagnostic / reference tab, not actionable.

### Hierarchy issues
- Two `st.markdown("##### …")` headings for the two views — H5 should be
  `section()` once tokens land.
- "Raw weekly snapshots" expander at the bottom is fine but uses default
  collapsed state — could be moved into a small "Data" / "Raw" toggle near
  the heatmap title.

### Spacing / proportions
- Line chart at `height=320` is sensible.
- Heatmap height auto-scales `max(320, 36 * len(heat.index))` — OK.

### Inconsistencies
- This is the only tab using `px.imshow` directly instead of going through
  a `src/charts.py` builder. Not a layout issue, but if Agent D defines
  shared color tokens, the heatmap colorscale should use them
  (currently hardcoded `RdYlGn`).

### Agent C action items
- `section()` helper for the two sub-views.
- Reduce visual weight: drop the BUY/SELL threshold caption from below the
  line chart and use horizontal threshold lines on the chart itself (out of
  scope if it requires re-plotting via plotly — leave as-is and just tighten
  spacing).
- The whole tab should feel like a "reference" view — slightly muted colors,
  smaller header.

---

## Tab 6 — 📧 Inbox (`app.py:1064-1129`)

### What it shows / primary action
Gmail connection status, two action buttons (Test connection · Fetch & parse),
two checkbox options, an ingest-report table.

### Hierarchy issues
- "🔌 Test connection" and "📥 Fetch & parse all" live in the same 3-column
  strip as the two checkbox toggles, with `st.columns([1, 1, 2])`. The
  checkboxes are jammed into the rightmost column with the labels truncated
  by streamlit's default checkbox width.
- The post-ingest metrics row (`Ingested · Skipped · Errors`) appears only
  *after* clicking Fetch — fine — but uses 3-column metrics which is the
  same widget pattern as Dashboard's state counts. Should match style.

### Spacing / proportions
- `Account` and `Filter` labels at the top are markdown lines, not metric
  cards. Inconsistent with the metric-strip pattern elsewhere.
- Form chrome dominates; the actual ingest-report table (the useful output)
  is at the bottom and only present after a button-press.

### Inconsistencies
- Inbox and Ingest share the structure: "configure → action → result". They
  should reuse a shared form helper.
- "Gmail not configured" warning is a block of help text; Tiger's "not
  configured" warning on the Dashboard uses identical language pattern.
  Could be unified.

### Agent C action items
- Extract a `connection_status_card(label, value, ok)` helper used by Gmail
  (Inbox), Tiger (Dashboard), and OpenAI (Ingest — could surface model name).
- Move both checkbox toggles to their own row below the action buttons so
  the buttons are clean and the toggles are clearly secondary.
- Use `section()` helper.

---

## Tab 7 — 📥 Ingest Newsletter (`app.py:1132-1165`)

### What it shows / primary action
Paste-text form for a single newsletter; calls GPT-4o-mini; shows JSON
result.

### Hierarchy issues
- The 3:1 left/right column split for "newsletter text + author/date hints"
  is OK — but the result block (`st.json(...)`) renders as one giant unformatted
  blob. Fine for debugging; not great for visual hierarchy.

### Spacing / proportions
- Text area at 380px height eats the viewport. Could be 280px and still be
  enough to verify the paste.

### Inconsistencies
- Shares the configure-→-action-→-result structure with Inbox (see Agent C
  action items).

### Agent C action items
- Reduce text-area height to ~260px.
- Render the parse result as a small structured panel (author / date / bias
  → big metric, then sectors as a small table) rather than `st.json` blob.
- Reuse the shared form helper from Inbox refactor.

---

## Tab 8 — 🗂 History (`app.py:1168-1196`)

### What it shows / primary action
Table of last 50 ingested newsletters + a "Current Aggregate Sentiment"
table below.

### Hierarchy issues
- "Delete an entry" is the only mutation on this tab, hidden inside an
  expander — that's correct (destructive action, gated). Good.
- The two tables ("Recent Newsletters" and "Current Aggregate Sentiment")
  are visually equal-weight, but the second one duplicates information
  already available on Trend tab.

### Spacing / proportions
- `height=400` on the recent-newsletters table is generous on a 900-tall
  viewport — could be 320 with a "View all" link if longer history is
  needed.

### Inconsistencies
- `st.divider()` between the two tables — only place in this tab a divider
  is used.

### Agent C action items
- Should feel "table-first" — drop the duplicate aggregate-sentiment table
  (it's already on the Trend tab) OR collapse it into a small footer
  panel.
- Use `section()` helper.

---

## Cross-tab interaction inventory (for Agent D + reviewer)

These widgets are *coupled* across tabs and any refactor must preserve them:

| Widget / state | Set by | Read by |
|---|---|---|
| `st.session_state.pa_sector` | Price Action selectbox + mini-grid buttons | Price Action chart |
| `st.session_state.exp_showing_<sector>` | Expressions tab selectbox | Expressions chart |
| `_cached_*` invalidations after price update | Update buttons in Price / Expressions tabs | Dashboard / Price / Expressions on next rerun |
| `_signal_row_style` | Dashboard matrix styler | Reads `STATE_COLORS` from `src.charts` |
| `_style_selfcheck` | Expressions table styler | Reads `EXPRESSION_STATE_COLORS` from `app.py` (target: move to `src/ui_tokens.py`) |

Anything that crosses these lines belongs to Agent D, not a tab specialist.

---

## Definition-of-done reminders for the merge phase

- 1440×900 viewport, no horizontal scroll, all 8 tabs.
- State colors mean the same thing in every tab that uses them.
- `src/ui_tokens.py` exists and is imported by `app.py`.
- `STATE_COLORS` in `src/charts.py` remains the **source of truth** for chart
  figures; tokens module extends it for UI chrome.
- All current widgets still present and functional (selectboxes, buttons,
  checkboxes, expanders, dataframes, metrics).
- No cache TTL changed; no cache function body changed; no signal logic
  touched.
