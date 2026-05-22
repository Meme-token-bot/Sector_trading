"""Shared UI design tokens for the Sector Rotation dashboard.

Single source of truth for chrome-layer colors, spacing constants, font-size
constants, and small reusable Streamlit helpers.  Chart-figure colors live in
`src.charts` (the Plotly layer must not depend on Streamlit); this module
imports those palettes and re-exports them so tab code has one place to import
from.

Public surface for downstream agents (A/B/C):
    from src.ui_tokens import (
        STATE_COLORS,          # chart + table row tinting (from charts.py)
        STATE_ACCENTS,         # brighter hues for text/borders (from charts.py)
        EXPRESSION_STATE_COLORS,  # 7-state self-check palette
        GAP_TIGHT, GAP_DEFAULT, GAP_LARGE,  # spacing references (rem)
        FS_H1, FS_H2, FS_H3, FS_CAPTION,   # font-size strings for md spans
        inject_base_css,       # call once at app startup
        render_header,         # replaces st.title + st.caption block
        section,               # replaces st.subheader + st.caption pairs
        state_badge,           # inline HTML span with state color
    )
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Re-export chart-layer palettes (STATE_COLORS stays in charts.py — Plotly
# figures depend on it directly and that module must not import Streamlit).
# ---------------------------------------------------------------------------
from src.charts import STATE_COLORS, STATE_ACCENTS  # noqa: F401 (re-export)

# ---------------------------------------------------------------------------
# Expression self-check state palette
# (background, accent/text)
# Derived from sector palette where the semantic meaning overlaps:
#   CONFIRMED  → same background as NEW_BUY   + #2ecc71 accent (fresh/green)
#   LAGGING    → same background as HOLD_IF_LONG + #f1c40f accent (amber)
#   STRETCHED  → same background as CHASE      + #e67e22 accent (orange)
#   BROKEN     → same background as SELL       + #e74c3c accent (red)
# ---------------------------------------------------------------------------
EXPRESSION_STATE_COLORS: dict[str, tuple[str, str]] = {
    "CONFIRMED":       (STATE_COLORS["NEW_BUY"],      "#2ecc71"),
    "LAGGING":         (STATE_COLORS["HOLD_IF_LONG"], "#f1c40f"),
    "STRETCHED":       (STATE_COLORS["CHASE"],        "#e67e22"),
    "BROKEN":          (STATE_COLORS["SELL"],         "#e74c3c"),
    "WARMING_UP":      ("#2a2a2a", "#888888"),
    "PARENT_INACTIVE": ("",        "#666666"),
    "NO_DATA":         ("#2a1414", "#aa6666"),
}

# ---------------------------------------------------------------------------
# Spacing constants (rem) — used as references in CSS strings or inline
# style comments; not consumed by Streamlit's gap= parameter.
# ---------------------------------------------------------------------------
GAP_TIGHT:   float = 0.5
GAP_DEFAULT: float = 1.0
GAP_LARGE:   float = 1.5

# ---------------------------------------------------------------------------
# Font-size tokens — use in f-string markdown spans, e.g.:
#   st.markdown(f'<span style="font-size:{FS_H2}">{text}</span>',
#               unsafe_allow_html=True)
# ---------------------------------------------------------------------------
FS_H1:      str = "2rem"
FS_H2:      str = "1.5rem"
FS_H3:      str = "1.125rem"
FS_CAPTION: str = "0.85rem"

# ---------------------------------------------------------------------------
# Base CSS — tightens vertical rhythm without touching the dark theme.
# Injected once via inject_base_css().
# ---------------------------------------------------------------------------
BASE_CSS: str = """
<style>
/* Reduce default top padding in the main block container */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* Tighten margins around h1/h2/h3 Streamlit injects via st.markdown */
.stMarkdown h1 {
    margin-top: 0.5rem;
    margin-bottom: 0.25rem;
}
.stMarkdown h2 {
    margin-top: 0.75rem;
    margin-bottom: 0.2rem;
}
.stMarkdown h3 {
    margin-top: 0.5rem;
    margin-bottom: 0.15rem;
}

/* Section divider line — used by the section() helper */
.ui-section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 0.5rem 0 0.25rem 0;
}
</style>
"""

# Module-level flag: inject CSS only once per Streamlit run so repeated
# calls to render_header() or inject_base_css() are a no-op.
_css_injected: bool = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def inject_base_css() -> None:
    """Inject BASE_CSS into the page (idempotent — first call only)."""
    global _css_injected
    if not _css_injected:
        st.markdown(BASE_CSS, unsafe_allow_html=True)
        _css_injected = True


def render_header(title: str, subtitle: str | None = None) -> None:
    """Render the page-level title block and inject base CSS.

    Replaces the raw ``st.title`` + ``st.caption`` block at the top of
    app.py.  Calling this again in the same Streamlit run is safe (CSS is
    only injected once; subsequent title/caption calls are idempotent from
    Streamlit's perspective).

    Args:
        title:    Main page title (rendered as H1 via st.title).
        subtitle: Optional one-line caption below the title.
    """
    inject_base_css()
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def section(title: str, help: str | None = None, *, level: int = 2) -> None:
    """Render a tab-level section heading with an optional help caption.

    Replaces ad-hoc ``st.subheader(title)`` + ``st.caption(help)`` pairs at
    the opening of each tab body.  Inner-tab section headings (Agent A/B/C
    scope) are not touched.

    Args:
        title: Section heading text.
        help:  Optional explanatory caption shown below the heading.
        level: Heading level — 1 through 4 (default 2).  Maps to:
               1 → st.header  (H2 — tab-level)
               2 → st.subheader (H3 — default; opens each tab)
               3 → markdown #### (H4 — inner section breaks)
               4 → markdown ##### (H5 — atomic block titles)
               This gives a continuous H2→H5 hierarchy.  Previously
               level=3 rendered H5, which collided with per-indicator
               H5 titles inside _render_macro_indicator on the Macro tab.
    """
    if level == 1:
        st.header(title)
    elif level == 2:
        st.subheader(title)
    elif level == 3:
        st.markdown(f"#### {title}")
    else:
        st.markdown(f"##### {title}")

    if help:
        st.caption(help)


def state_badge(state: str, palette: dict | None = None) -> str:
    """Return a small inline HTML ``<span>`` styled with the state's colors.

    Useful for inserting a colored state label inside a markdown string:
        st.markdown(f"State: {state_badge('NEW_BUY')}", unsafe_allow_html=True)

    Args:
        state:   State key, e.g. ``"NEW_BUY"``, ``"CONFIRMED"``.
        palette: Color dict to look up.  If None, tries EXPRESSION_STATE_COLORS
                 first (tuple values), then STATE_COLORS (str values).

    Returns:
        An HTML ``<span>`` string.  Falls back to plain text if the state is
        not found in either palette.
    """
    if palette is None:
        # Try expression palette first (tuple), then sector palette (str).
        if state in EXPRESSION_STATE_COLORS:
            bg, fg = EXPRESSION_STATE_COLORS[state]
        elif state in STATE_COLORS:
            bg = STATE_COLORS[state]
            fg = STATE_ACCENTS.get(state, "#eeeeee")
        else:
            return f"<span>{state}</span>"
    else:
        val = palette.get(state)
        if val is None:
            return f"<span>{state}</span>"
        if isinstance(val, tuple):
            bg, fg = val
        else:
            bg = val
            fg = "#eeeeee"

    style_parts = [f"color:{fg}", "padding:1px 6px", "border-radius:3px",
                   f"font-size:{FS_CAPTION}", "font-weight:600"]
    if bg:
        style_parts.insert(0, f"background:{bg}")
    style = ";".join(style_parts)
    return f'<span style="{style}">{state}</span>'
