"""Streamlit dashboard — entrypoint."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from config.settings import (
    BENCHMARK, GMAIL_ADDRESS, GMAIL_FILTER_ADDRESS, PARAMS,
    SECTOR_ETFS, gmail_configured, tiger_configured,
)
from src.charts import STATE_COLORS as _STATE_COLORS, build_etf_chart, build_mini_chart, compute_chart_overlays
from src.ui_tokens import EXPRESSION_STATE_COLORS, render_header, section
from src.db import aggregate_sentiment, delete_newsletter, init_db, recent_newsletters
from src.expression_signals import compute_expressions_for_sector
from src.market_engine import (
    compute_sector_metrics, copper_gold_ratio, dxy_level,
    fetch_fred_indicators, fetch_macro_prices, fetch_prices,
    gold_oil_ratio, vix_level, yield_curve_spread,
)
from src.nlp_pipeline import fetch_and_ingest, ingest
from src.price_store import load_ohlcv, load_ohlcv_multi, update_all
from src.signal_history import build_signal_history
from src.signals import build_signals, refine_signals, target_weights
from src.trend import build_sentiment_trend

st.set_page_config(
    page_title="Sector Rotation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()


def _full_price_universe() -> list[str]:
    """Tickers seeded into the OHLCV cache: signals + benchmark + all
    expression tickers, deduped while preserving signal-first order."""
    from config.expressions import all_expression_tickers
    return list(dict.fromkeys([*SECTOR_ETFS, BENCHMARK, *all_expression_tickers()]))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_prices() -> pd.DataFrame:
    return fetch_prices(list(SECTOR_ETFS.keys()) + [BENCHMARK])

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_macro_prices() -> pd.DataFrame:
    return fetch_macro_prices()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_yield_curve() -> dict:
    return yield_curve_spread()

# FRED indicators are cached separately from price data: the FRED endpoint
# occasionally flaps and we don't want a transient FRED failure to evict the
# yfinance price cache (or vice versa).
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_fred_indicators() -> dict:
    return fetch_fred_indicators()

@st.cache_data(ttl=10 * 60, show_spinner=False)
def _cached_sentiment(as_of_iso: str) -> pd.DataFrame:
    return aggregate_sentiment(as_of=date.fromisoformat(as_of_iso))

@st.cache_data(ttl=10 * 60, show_spinner=False)
def _cached_trend(as_of_iso: str, lookback_days: int) -> pd.DataFrame:
    return build_sentiment_trend(lookback_days=lookback_days,
                                 end=date.fromisoformat(as_of_iso))

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_signal_history(as_of_iso: str) -> pd.DataFrame:
    prices = _cached_prices()
    return build_signal_history(prices, end=date.fromisoformat(as_of_iso))

@st.cache_data(ttl=5 * 60, show_spinner=False)
def _cached_tiger_snapshot():
    from src.tiger_client import fetch_account_snapshot
    return fetch_account_snapshot()


def _signal_row_style(row: pd.Series) -> list[str]:
    state = row.get("State", row.get("state", row.get("Signal", row.get("signal", ""))))
    color = _STATE_COLORS.get(state, "")
    return [f"background-color: {color}; color: #eee" if color else "" for _ in row]


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%" if pd.notna(x) else "—"


render_header(
    "📊 Sector Rotation — Macro-Filtered Convergence Model",
    subtitle=(
        f"11 US SPDR Select Sector ETFs · benchmark **{BENCHMARK}** · "
        f"weekly cadence · BUY threshold sentiment >= +{PARAMS.buy_sentiment_threshold:.0f}, "
        f"SELL <= {PARAMS.sell_sentiment_threshold:+.0f}"
    ),
)

(tab_dashboard, tab_macro, tab_price, tab_expressions, tab_trend,
 tab_inbox, tab_ingest, tab_history) = st.tabs(
    ["📈 Dashboard", "🌐 Macro", "📉 Price Action", "🎯 Expressions", "✨ Trend",
     "📧 Inbox", "📥 Ingest Newsletter", "🗂 History"]
)

with tab_dashboard:
    with st.spinner("Refreshing market data…"):
        prices = _cached_prices()
        metrics = compute_sector_metrics(prices)
        sentiment = _cached_sentiment(date.today().isoformat())
        raw_signals = build_signals(metrics, sentiment)
        history = _cached_signal_history(date.today().isoformat())
        signals = refine_signals(raw_signals, history)
        targets = target_weights(signals)

    left, right = st.columns([3, 2], gap="large")

    with left:
        section("Sector Relative Strength Matrix", level=3)

        display = signals.copy()
        display["3M vs SPY"] = display["relative_strength_3m"].map(_fmt_pct)
        display["Ext vs SMA"] = display["extension_pct"].map(_fmt_pct)
        display["Wks BUY"] = display["consecutive_buy_weeks"].astype(int)
        display["Sentiment"] = display.apply(
            lambda r: f"{r['sentiment_score']:+.1f} (n={int(r['n_obs'])})", axis=1
        )

        view = display[["name", "3M vs SPY", "Ext vs SMA", "Wks BUY",
                        "Sentiment", "state", "state_reason"]].rename(
            columns={"name": "Sector", "state": "State", "state_reason": "Action"}
        )

        styled = view.style.apply(_signal_row_style, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            height=460,
            column_config={
                "Sector":    st.column_config.TextColumn("Sector",    width="medium"),
                "3M vs SPY": st.column_config.TextColumn("3M vs SPY", width="small"),
                "Ext vs SMA":st.column_config.TextColumn("Ext vs SMA",width="small"),
                "Wks BUY":   st.column_config.NumberColumn("Wks BUY", width="small"),
                "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "State":     st.column_config.TextColumn("State",     width="small"),
                "Action":    st.column_config.TextColumn("Action",    width="large"),
            },
        )

        st.caption("State distribution")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col, state in zip(
            [c1, c2, c3, c4, c5, c6],
            ["NEW_BUY", "HOLD_IF_LONG", "CHASE", "REDUCE", "HOLD", "SELL"],
        ):
            col.metric(state, int((signals["state"] == state).sum()))

        _has_buy_signals = not targets.empty
        with st.expander(
            "Target weights — actionable allocation (equal-weight NEW_BUY + HOLD_IF_LONG, 5% cash buffer)",
            expanded=_has_buy_signals,
        ):
            if targets.empty:
                st.info("No actionable BUY-class signals — model says stay defensive / in cash.")
            else:
                tdf = targets.to_frame()
                tdf["sector"] = tdf.index.map(SECTOR_ETFS)
                tdf["state"] = signals.loc[tdf.index, "state"]
                tdf["target_weight"] = tdf["target_weight"].map(lambda x: f"{x:.1%}")
                st.dataframe(tdf[["sector", "state", "target_weight"]],
                             use_container_width=True)
                st.caption(
                    "**Important:** if a row shows `HOLD_IF_LONG` and you don't currently own it, "
                    "do NOT enter — the trend is mature. The target weight is what you'd hold if "
                    "you already had a position. CHASE rows are excluded from targets entirely."
                )

        with st.expander("How to read the State column", expanded=False):
            st.markdown(
                f"""
- 🟢 **NEW_BUY** — convergence test passes, sector is not extended
  ({PARAMS.extension_pct_cutoff*100:.0f}% above SMA200 cap), and has been BUY
  for fewer than {PARAMS.stale_buy_weeks} weeks. **Fresh entry OK.**
- 🟡 **HOLD_IF_LONG** — still passes BUY test but has been BUY for
  ≥ {PARAMS.stale_buy_weeks} consecutive weekly snapshots. **If you already own
  it, hold. If you don't, sit it out — the trend is mature, don't chase.**
- 🟠 **CHASE** — would be BUY but price is more than
  {PARAMS.extension_pct_cutoff*100:.0f}% above SMA200. **Too extended for fresh entry.**
  Wait for a pullback to the SMA before considering.
- 🟤 **REDUCE** — was BUY in the last {PARAMS.history_weeks} weeks but no longer
  qualifies (sentiment cooled, RS turned, etc.). **Trim if owned.**
- ⚪ **HOLD** — doesn't qualify as BUY and never did recently. **Wait and see.**
- 🔴 **SELL** — fails one of the hard SELL rules (price < SMA200,
  bottom-3 RS rank, or sentiment ≤ {PARAMS.sell_sentiment_threshold:+.0f}). **Exit.**

`Wks BUY` = consecutive weekly snapshots (last {PARAMS.history_weeks} weeks)
where the raw convergence test passed. `Ext vs SMA` = (price − SMA200) / SMA200.
                """
            )

    with right:
        section("Tiger Portfolio Drift", level=3)

        if not tiger_configured():
            st.warning(
                "Tiger SDK not configured. Add `TIGER_ID`, `TIGER_ACCOUNT`, "
                "and `TIGER_PRIVATE_KEY_PATH` to `.env` to enable live drift tracking."
            )
            with st.expander("Enter NLV manually for a dry-run drift table"):
                manual_nlv = st.number_input("Net liquidation value ($)",
                                             min_value=0.0, value=100_000.0, step=1000.0)
                drift_manual = pd.DataFrame({
                    "target_weight": targets.reindex(SECTOR_ETFS.keys()).fillna(0.0),
                    "target_value": (targets.reindex(SECTOR_ETFS.keys()).fillna(0.0)
                                     * manual_nlv),
                })
                drift_manual["target_weight"] = drift_manual["target_weight"].map("{:.1%}".format)
                drift_manual["target_value"] = drift_manual["target_value"].map("${:,.0f}".format)
                st.dataframe(drift_manual, use_container_width=True)
        else:
            try:
                snap = _cached_tiger_snapshot()
                from src.tiger_client import compute_drift_by_sector
                drift = compute_drift_by_sector(snap, targets)

                t1, t2 = st.columns(2)
                t1.metric("Net Liq Value", f"${snap.net_liquidation:,.0f}")
                t2.metric("Cash", f"${snap.cash:,.0f}",
                          delta=f"{snap.cash / snap.net_liquidation:.1%}" if snap.net_liquidation else None)

                show = drift.copy()
                show["target_weight"] = show["target_weight"].map("{:.1%}".format)
                show["current_weight"] = show["current_weight"].map("{:.1%}".format)
                show["drift"] = show["drift"].map("{:+.1%}".format)
                show["trade_value"] = show["trade_value"].map(
                    lambda v: f"BUY ${v:,.0f}" if v > 100
                              else (f"SELL ${-v:,.0f}" if v < -100 else "—")
                )
                st.dataframe(
                    show[["target_weight", "current_weight", "drift", "trade_value"]],
                    use_container_width=True,
                )

                unmapped = drift.attrs.get("unmapped", {})
                if unmapped:
                    with st.expander(f"Unmapped holdings ({len(unmapped)})"):
                        umdf = pd.DataFrame(
                            [(sym, mv) for sym, mv in unmapped.items()],
                            columns=["symbol", "market_value"],
                        ).sort_values("market_value", ascending=False)
                        umdf["market_value"] = umdf["market_value"].map("${:,.0f}".format)
                        st.dataframe(umdf, use_container_width=True, hide_index=True)
                        st.caption("These holdings are not in any expression list — "
                                   "they don't count toward sector targets.")
            except Exception as e:
                st.error(f"Tiger fetch failed: {e}")

    # ---- Diagnostics footer --------------------------------------------------
    st.caption("Diagnostics")
    _diag_col, _ = st.columns([1, 4])
    with _diag_col:
        if st.button("🔄 Force refresh all caches"):
            st.cache_data.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Macro tab
# ---------------------------------------------------------------------------
#
# Each indicator is rendered as one self-contained block: title + description,
# metric (current + z or 30d slope), regime badge from the bands table, the
# full band legend so the user can see where the current reading sits in
# range, a one-line sector-rotation signal, and a trailing-1y line chart.
#
# Band tuples are (label, emoji, range_label, predicate). Bands are tested
# top-to-bottom; the first matching predicate wins. Ranges are anchored to
# regime-detection convention, not strict statistical thresholds — they're
# meant as "is this reading meaningful" guideposts, not trade triggers.

_VIX_BANDS = [
    ("Complacent", "🟢", "< 12",  lambda v: v < 12),
    ("Normal",     "🟡", "12–20", lambda v: 12 <= v < 20),
    ("Stressed",   "🟠", "20–30", lambda v: 20 <= v < 30),
    ("Crisis",     "🔴", "≥ 30",  lambda v: v >= 30),
]
_HY_OAS_BANDS = [
    ("Tight (risk-on)", "🟢", "< 3.5%",   lambda v: v < 3.5),
    ("Normal",          "🟡", "3.5–5.0%", lambda v: 3.5 <= v < 5.0),
    ("Stress building", "🟠", "5.0–7.0%", lambda v: 5.0 <= v < 7.0),
    ("Credit crisis",   "🔴", "≥ 7.0%",   lambda v: v >= 7.0),
]
_GOLD_OIL_BANDS = [
    ("Oil-rich / pro-cyclical", "🟢", "< 15",  lambda v: v < 15),
    ("Normal",                  "🟡", "15–25", lambda v: 15 <= v < 25),
    ("Risk-off bid",            "🟠", "25–35", lambda v: 25 <= v < 35),
    ("Recession-likely",        "🔴", "≥ 35",  lambda v: v >= 35),
]
# Copper/Gold absolute level varies with overall metal prices — band on
# 1y z-score instead so the regime call is comparable across cycles.
_COPPER_GOLD_Z_BANDS = [
    ("Deflationary",     "🔴", "z < -1",   lambda v: v < -1),
    ("Softening",        "🟠", "-1 to 0",  lambda v: -1 <= v < 0),
    ("Reflation",        "🟢", "0 to +1",  lambda v: 0 <= v < 1),
    ("Strong reflation", "🟢", "z ≥ +1",   lambda v: v >= 1),
]
_DXY_BANDS = [
    ("Weak dollar",   "🟢", "< 95",    lambda v: v < 95),
    ("Normal-weak",   "🟡", "95–100",  lambda v: 95 <= v < 100),
    ("Normal-strong", "🟠", "100–105", lambda v: 100 <= v < 105),
    ("Strong dollar", "🔴", "≥ 105",   lambda v: v >= 105),
]
_T10Y2Y_BANDS = [
    ("Inverted (recession warning)", "🔴", "< 0",       lambda v: v < 0),
    ("Flat",                          "🟠", "0–0.5%",    lambda v: 0 <= v < 0.5),
    ("Normal",                        "🟡", "0.5–1.5%",  lambda v: 0.5 <= v < 1.5),
    ("Steep",                         "🟢", "≥ 1.5%",    lambda v: v >= 1.5),
]
_UST10_BANDS = [
    ("Easy",        "🟢", "< 3%", lambda v: v < 3),
    ("Normal",      "🟡", "3–4%", lambda v: 3 <= v < 4),
    ("Restrictive", "🟠", "4–5%", lambda v: 4 <= v < 5),
    ("Tight",       "🔴", "≥ 5%", lambda v: v >= 5),
]
_REAL_10Y_BANDS = [
    ("Financial repression", "🟢", "< 0",  lambda v: v < 0),
    ("Normal",               "🟡", "0–1%", lambda v: 0 <= v < 1),
    ("Restrictive",          "🟠", "1–2%", lambda v: 1 <= v < 2),
    ("Tight",                "🔴", "≥ 2%", lambda v: v >= 2),
]
_BREAKEVEN_BANDS = [
    ("Deflationary fears", "🔴", "< 1.8%",   lambda v: v < 1.8),
    ("Anchored",           "🟢", "1.8–2.5%", lambda v: 1.8 <= v < 2.5),
    ("Unanchored",         "🟠", "≥ 2.5%",   lambda v: v >= 2.5),
]


def _find_band(value, bands):
    if value is None or value != value:  # NaN check without numpy dependency
        return None
    for entry in bands:
        if entry[3](value):
            return entry
    return None


def _render_macro_indicator(*, label, payload, title, description,
                            fmt, bands, signal, delta_kind="z",
                            band_input="current", compact: bool = False):
    """Render one macro indicator block.

    `band_input` is "current" for absolute-level bands, or "z" when bands
    are defined on the trailing 1y z-score instead (Copper/Gold). `delta_kind`
    is "z" for z-score deltas, "slope" for 30d slope deltas.

    `compact` controls the internal layout:
    - False (default, legacy): metric+bands on left, chart on right via
      st.columns([1, 2]).  Use for any full-width call sites.
    - True: stacked layout — metric + regime + bands caption on top, chart
      below.  Reduces chart height to 160 px.  Use when this helper is called
      inside an outer column (the inner columns would be too narrow).
    """
    cur = payload.get("current")
    has_data = pd.notna(cur)

    st.markdown(f"##### {title}")
    st.caption(description)

    if compact:
        # Stacked layout: metric block above, chart below
        if has_data:
            if delta_kind == "z":
                z = payload.get("z_score_1y")
                delta = f"z={z:+.2f}" if pd.notna(z) else None
            else:
                slope = payload.get("slope_30d")
                delta = (f"{slope*30:+.2f}/mo (30d)"
                         if pd.notna(slope) else None)
            st.metric(label, fmt.format(cur), delta=delta)

            band_val = (cur if band_input == "current"
                        else payload.get("z_score_1y"))
            band = _find_band(band_val, bands)
            if band:
                blabel, bemoji, _, _ = band
                st.markdown(f"**Regime:** {bemoji} {blabel}")
            else:
                st.markdown("**Regime:** ⚪ —")
        else:
            st.metric(label, "—",
                      help=payload.get("error", "source unavailable"))

        st.caption("**Bands:** " +
                   " · ".join(f"{e} {r}" for _, e, r, _ in bands))
        st.caption(f"**Sector signal:** {signal}")

        if "series" in payload:
            st.line_chart(payload["series"].tail(252), height=160,
                          use_container_width=True)
    else:
        left, right = st.columns([1, 2])
        with left:
            if has_data:
                if delta_kind == "z":
                    z = payload.get("z_score_1y")
                    delta = f"z={z:+.2f}" if pd.notna(z) else None
                else:
                    slope = payload.get("slope_30d")
                    delta = (f"{slope*30:+.2f}/mo (30d)"
                             if pd.notna(slope) else None)
                st.metric(label, fmt.format(cur), delta=delta)

                band_val = (cur if band_input == "current"
                            else payload.get("z_score_1y"))
                band = _find_band(band_val, bands)
                if band:
                    blabel, bemoji, _, _ = band
                    st.markdown(f"**Regime:** {bemoji} {blabel}")
                else:
                    st.markdown("**Regime:** ⚪ —")
            else:
                st.metric(label, "—",
                          help=payload.get("error", "source unavailable"))

            st.caption("**Bands:** " +
                       " · ".join(f"{e} {r}" for _, e, r, _ in bands))
            st.caption(f"**Sector signal:** {signal}")

        with right:
            if "series" in payload:
                st.line_chart(payload["series"].tail(252), height=200,
                              use_container_width=True)


with tab_macro:
    section(
        "Macro Regime Indicators",
        help=(
            "Cross-asset signals that shape sector rotation. Each indicator "
            "shows its current reading, trailing-1y z-score (or 30d slope for "
            "yields), a regime band, and the sector implication. **Bands are "
            "rules-of-thumb, not trade triggers** — they're meant to tell you "
            "what kind of regime you're in, not to time entries."
        ),
    )

    st.info(
        "**Reading the panel together** — "
        "Look for **agreement**: VIX up + HY OAS up + DXY up + curve flattening = "
        "a coherent risk-off regime. Trim cyclicals (XLF, XLY, XLI, XLB) and lean "
        "defensive (XLP, XLU, XLV). "
        "Look for **divergence**: equity vol calm but credit spreads widening is an "
        "early-stress signal — credit cracks before equities. "
        "**Direction matters more than level.** A 'Normal' reading that's rising fast "
        "(z > +1) is often a stronger signal than a 'Stressed' reading that's stable."
    )

    macro_prices = _cached_macro_prices()
    gor = gold_oil_ratio(macro_prices)
    cgr = copper_gold_ratio(macro_prices)
    dxy = dxy_level(macro_prices)
    vix = vix_level(macro_prices)
    yc = _cached_yield_curve()
    fred = _cached_fred_indicators()

    # ---- Risk / Vol ---------------------------------------------------
    section(
        "🛡️ Risk / Vol",
        help=(
            "Equity-vol and credit-stress gauges. Elevated readings push the "
            "playbook toward defensives (XLP, XLU, XLV) and away from cyclicals "
            "(XLF, XLY, XLI, XLB)."
        ),
        level=3,
    )

    _rv_col1, _rv_col2 = st.columns(2)
    with _rv_col1:
        _render_macro_indicator(
            label="VIX",
            payload=vix,
            title="VIX — S&P 500 Implied Volatility",
            description=("30-day expected S&P 500 volatility implied by option "
                         "prices. Spikes when realized risk rises or when "
                         "investors bid up tail-protection."),
            fmt="{:.1f}",
            bands=_VIX_BANDS,
            signal=("VIX > 25 → tilt to defensives, trim cyclicals. "
                    "VIX < 13 → complacency; growth re-engagement OK but "
                    "watch for vol expansion."),
            compact=True,
        )
    with _rv_col2:
        _render_macro_indicator(
            label="HY OAS",
            payload=fred.get("HY_OAS", {}),
            title="HY OAS — High-Yield Credit Spread",
            description=("ICE BofA US High-Yield option-adjusted spread over "
                         "Treasuries. The single best gauge of risk-asset stress; "
                         "credit cracks before equities do."),
            fmt="{:.2f}%",
            bands=_HY_OAS_BANDS,
            signal=("OAS > 5% → reduce cyclical risk (XLF, XLY, XLI); rising z "
                    "regardless of level is a warning. OAS < 3.5% → credit "
                    "supportive of risk-on rotation."),
            compact=True,
        )

    # Third Risk/Vol indicator sits alone in a half-width cell (odd count)
    _rv_col3, _ = st.columns(2)
    with _rv_col3:
        _render_macro_indicator(
            label="Gold / Oil",
            payload=gor,
            title="Gold / Oil Ratio",
            description=("GC=F front-month / CL=F front-month. Rises when gold "
                         "(safe-haven, real-asset hedge) outperforms oil "
                         "(growth-sensitive demand)."),
            fmt="{:.1f}",
            bands=_GOLD_OIL_BANDS,
            signal=("Ratio > 30 → recession/risk-off bid; favor XLP, XLU, XLV. "
                    "Ratio < 15 → strong oil cycle; XLE tailwind. "
                    "Big z-spikes have led peaks historically."),
            compact=True,
        )

    # ---- Growth / Cycle -----------------------------------------------
    section(
        "📈 Growth / Cycle",
        help=(
            "Cyclical-vs-defensive cross-asset signals. These move first when "
            "the global growth impulse shifts."
        ),
        level=3,
    )

    _gc_col1, _gc_col2 = st.columns(2)
    with _gc_col1:
        _render_macro_indicator(
            label="Copper / Gold",
            payload=cgr,
            title="Copper / Gold Ratio",
            description=("HG=F / GC=F. Copper is industrial-demand sensitive; "
                         "gold is monetary/safe-haven. The ratio is a classic "
                         "growth/reflation barometer."),
            fmt="{:.4f}",
            bands=_COPPER_GOLD_Z_BANDS,
            band_input="z",
            signal=("Z > +1 → reflation regime; tailwind for XLB, XLI, XLE. "
                    "Z < -1 → deflationary impulse; rotate to bond proxies "
                    "(XLU, XLRE) and quality defensives."),
            compact=True,
        )
    with _gc_col2:
        _render_macro_indicator(
            label="DXY",
            payload=dxy,
            title="DXY — US Dollar Index",
            description=("Trade-weighted USD vs a basket of major currencies. "
                         "Tighter US financial conditions and risk-off flows "
                         "tend to lift the dollar."),
            fmt="{:.2f}",
            bands=_DXY_BANDS,
            signal=("DXY > 105 → headwind for commodities (XLB, XLE) and "
                    "multinational earnings (XLK, XLI). DXY < 95 → commodity "
                    "tailwind, EM-sensitive sectors benefit."),
            compact=True,
        )

    # ---- Rates / Inflation --------------------------------------------
    section(
        "💵 Rates / Inflation",
        help=(
            "Treasury curve and inflation-expectation signals. The level of "
            "rates and their direction matter for duration-sensitive sectors "
            "(XLRE, XLU, XLK) and financial-margin sectors (XLF)."
        ),
        level=3,
    )

    _ri_col1, _ri_col2 = st.columns(2)
    with _ri_col1:
        _render_macro_indicator(
            label="10Y - 2Y",
            payload=yc,
            title="10Y - 2Y Treasury Spread",
            description=("DGS10 - DGS2 from FRED. Inversion has historically "
                         "preceded recessions by 12-18 months; the steepening "
                         "out of inversion is the actual recession trigger."),
            fmt="{:+.2f}%",
            bands=_T10Y2Y_BANDS,
            signal=("Inverted → late-cycle; trim cyclicals, build defensives. "
                    "Steepening from inversion → bull steepener supports XLF; "
                    "bear steepener (long end leading) pressures XLRE/XLU."),
            delta_kind="slope",
            compact=True,
        )
    with _ri_col2:
        _render_macro_indicator(
            label="10Y nominal",
            payload=fred.get("UST10", {}),
            title="10Y Nominal Yield",
            description=("Constant-maturity 10-year Treasury yield. The "
                         "discount-rate input for everything; rising long-end "
                         "rates compress long-duration multiples."),
            fmt="{:.2f}%",
            bands=_UST10_BANDS,
            signal=("> 5% → duration headwind, pressure on XLRE, XLU, XLK. "
                    "Rising slope (regardless of level) → defensive long-duration "
                    "rotation; falling slope → growth/duration re-engagement."),
            delta_kind="slope",
            compact=True,
        )

    _ri_col3, _ri_col4 = st.columns(2)
    with _ri_col3:
        _render_macro_indicator(
            label="10Y real",
            payload=fred.get("REAL_10Y", {}),
            title="10Y Real Yield (TIPS)",
            description=("10-year TIPS yield: the real (inflation-adjusted) "
                         "cost of capital. The cleanest read on monetary policy "
                         "stance; arguably more important than the nominal yield."),
            fmt="{:+.2f}%",
            bands=_REAL_10Y_BANDS,
            signal=("Real > 2% → restrictive; headwind for gold miners, XLRE, "
                    "and long-duration growth. Real < 0% → financial repression; "
                    "supportive for risk assets and real-asset proxies."),
            delta_kind="slope",
            compact=True,
        )
    with _ri_col4:
        _render_macro_indicator(
            label="5Y5Y breakeven",
            payload=fred.get("BREAKEVEN_5Y5Y", {}),
            title="5Y5Y Forward Inflation Breakeven",
            description=("Market-implied inflation expectation for the 5 years "
                         "starting 5 years from now. The Fed's preferred gauge "
                         "of long-run inflation credibility."),
            fmt="{:.2f}%",
            bands=_BREAKEVEN_BANDS,
            signal=("< 1.8% → deflation fears, risk-off for cyclicals. "
                    "1.8-2.5% → anchored, neutral. > 2.5% → unanchored / "
                    "reflation; tailwind for XLE, XLB, but watch for hawkish Fed."),
            compact=True,
        )

    with st.expander("📖 How to use this tab — bands & caveats"):
        st.markdown(
            """
**Reading a single indicator**
- **Level + regime band** tells you where we are in the cycle.
- **Z-score (or 30d slope)** tells you the *change* — often the actionable signal.
- **Sector signal** is the if-then rule for rotation. Treat as a tilt, not a switch.

**Caveats**
- Bands are calibrated to post-GFC norms. Treat readings in extreme regimes (2020, 2022) as outliers.
- All signals are *displayed*, not yet wired into `build_signals()`. Use them as a manual sanity overlay on the convergence model output for now.
            """
        )


# ---------------------------------------------------------------------------
# Price Action tab
# ---------------------------------------------------------------------------

_LOOKBACK_DAYS = {"3M": 92, "6M": 183, "1Y": 365, "2Y": 730, "5Y": 1825}

# Warmup buffer prepended to whatever the user asked for. SMA200 needs ~200
# bars before it produces a non-NaN value, so we load extra history and clip
# back to the requested window inside `build_etf_chart` via `visible_start`.
# Daily: ~300 calendar days covers 200 trading days with slack. Weekly: 300
# weeks ≈ 2100 calendar days covers 200 weekly bars with slack. The prices DB
# holds ~5y so this is always available; SQL clamps at the earliest stored bar.
_WARMUP_DAYS_BY_TF = {"1d": 300, "1wk": 300 * 7}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_ohlcv(ticker: str, timeframe: str, start_iso: str) -> pd.DataFrame:
    return load_ohlcv(ticker, timeframe, start=date.fromisoformat(start_iso))


@st.cache_data(ttl=300, show_spinner=False)
def _cached_ohlcv_multi(tickers: tuple[str, ...], timeframe: str,
                        start_iso: str) -> pd.DataFrame:
    return load_ohlcv_multi(list(tickers), timeframe,
                            start=date.fromisoformat(start_iso))


@st.cache_data(ttl=300, show_spinner=False)
def _cached_sector_sparklines(sector: str, as_of_iso: str) -> dict[str, list[float]]:
    """Last 60 daily closes for each expression ticker in `sector`.

    Loads ~90 calendar days from the prices DB (≈ 60 trading days) and returns
    a dict {ticker: list[float]} suitable for `st.column_config.LineChartColumn`.
    Tickers with no stored data map to an empty list.
    """
    from config.expressions import EXPRESSIONS
    tickers = [e.ticker for e in EXPRESSIONS.get(sector, [])]
    if not tickers:
        return {}
    spark_start = date.fromisoformat(as_of_iso) - timedelta(days=90)
    frame = load_ohlcv_multi(tickers, "1d", start=spark_start)
    top_level = (set(frame.columns.get_level_values(0))
                 if not frame.empty else set())
    out: dict[str, list[float]] = {}
    for tkr in tickers:
        if tkr in top_level:
            try:
                series = frame[tkr]["close"].dropna().tail(60)
                out[tkr] = [float(v) for v in series.tolist()]
            except KeyError:
                out[tkr] = []
        else:
            out[tkr] = []
    return out



@st.cache_data(ttl=300, show_spinner=False)
def _cached_expression_signals(
    sector: str, parent_state: str, as_of_iso: str
) -> list[dict]:
    """Per-expression self-check signals for one sector.

    Returns a list of plain dicts (not the dataclass) so streamlit's cache
    doesn't choke on the frozen dataclass and so the call site doesn't have
    to re-import the type.
    """
    warmup_start = date.fromisoformat(as_of_iso) - timedelta(days=300)

    def _loader(ticker: str) -> pd.Series:
        df = load_ohlcv(ticker, "1d", start=warmup_start)
        if df.empty:
            return pd.Series(dtype=float)
        return df["close"]

    sigs = compute_expressions_for_sector(sector, parent_state, _loader)
    return [
        {
            "ticker": s.ticker,
            "state": s.state,
            "reason": s.reason,
        }
        for s in sigs
    ]


@st.cache_data(ttl=300, show_spinner=False)
def _cached_signals_bundle(as_of_iso: str) -> pd.DataFrame:
    """Re-compute the same signals bundle the Dashboard tab uses. Cached so
    repeated Price Action reruns (toggling indicators, switching tickers) don't
    re-run the metrics + signals pipeline each time."""
    prices = _cached_prices()
    metrics = compute_sector_metrics(prices)
    sentiment = _cached_sentiment(as_of_iso)
    raw_signals = build_signals(metrics, sentiment)
    history = _cached_signal_history(as_of_iso)
    return refine_signals(raw_signals, history)


with tab_price:
    section(
        "Price Action",
        help=(
            "Candles, SMA50/200, optional RSI/MACD/Bollinger. Data is served from "
            "the local prices DB (5y of 1d + 1wk). Use **Update price data** to "
            "incrementally pull the latest bars from yfinance."
        ),
    )

    signals = _cached_signals_bundle(date.today().isoformat())

    all_tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]

    # ---- session-state defaults (one-time init) ----
    if "pa_sector" not in st.session_state:
        # Default = freshest NEW_BUY in the signals frame; fallback XLK.
        new_buys = signals.index[signals["state"] == "NEW_BUY"].tolist()
        st.session_state.pa_sector = new_buys[0] if new_buys else "XLK"
    if "pa_timeframe" not in st.session_state:
        st.session_state.pa_timeframe = "Daily"
    if "pa_lookback" not in st.session_state:
        st.session_state.pa_lookback = "1Y"
    if "pa_compare_spy" not in st.session_state:
        st.session_state.pa_compare_spy = False
    if "pa_show_rsi" not in st.session_state:
        st.session_state.pa_show_rsi = False
    if "pa_show_macd" not in st.session_state:
        st.session_state.pa_show_macd = False
    if "pa_show_bb" not in st.session_state:
        st.session_state.pa_show_bb = False

    # ---- toolbar ----
    tb1, tb2, tb3, tb4 = st.columns([1.3, 1, 1.2, 1.2])
    with tb1:
        st.selectbox(
            "Sector", all_tickers, key="pa_sector",
            format_func=lambda t: f"{t} — {SECTOR_ETFS.get(t, 'Benchmark')}",
        )
    with tb2:
        st.radio("Timeframe", ["Daily", "Weekly"], key="pa_timeframe",
                 horizontal=True)
    with tb3:
        st.radio("Lookback", list(_LOOKBACK_DAYS.keys()), key="pa_lookback",
                 horizontal=True)
    with tb4:
        st.checkbox("Compare to SPY", key="pa_compare_spy")
        if st.button("🔄 Update price data", key="pa_update_btn"):
            prog_bar = st.progress(0.0)
            prog_caption = st.empty()
            # Update the full price universe (signals + expressions) so the
            # Expressions tab has data too — not just the signal sectors.
            tickers_to_update = _full_price_universe()
            total_steps = len(tickers_to_update) * 2  # 1d + 1wk
            done = {"n": 0}

            def _progress(tkr: str, tf: str, status: str) -> None:
                done["n"] += 1
                prog_bar.progress(done["n"] / total_steps)
                prog_caption.caption(f"{tkr} ({tf}): {status}")

            with st.spinner("Updating from yfinance…"):
                try:
                    update_all(tickers=tickers_to_update, progress=_progress)
                except Exception as e:
                    st.error(f"Update failed: {e}")
                else:
                    # Targeted invalidation — don't nuke Dashboard caches.
                    _cached_ohlcv.clear()
                    _cached_ohlcv_multi.clear()
                    st.success("Price data refreshed.")
                    st.rerun()

    # Indicator checkboxes
    ic1, ic2, ic3 = st.columns(3)
    ic1.checkbox("RSI(14)", key="pa_show_rsi")
    ic2.checkbox("MACD(12,26,9)", key="pa_show_macd")
    ic3.checkbox("Bollinger Bands (20, 2σ)", key="pa_show_bb")

    # ---- main chart ----
    ticker = st.session_state.pa_sector
    timeframe = "1d" if st.session_state.pa_timeframe == "Daily" else "1wk"
    lookback_days = _LOOKBACK_DAYS[st.session_state.pa_lookback]
    start = date.today() - timedelta(days=lookback_days)

    # Load with extra warmup so SMA200 is populated across the visible window.
    # `build_etf_chart` clips back to `visible_start` for display.
    warmup_days = _WARMUP_DAYS_BY_TF[timeframe]
    fetch_start = start - timedelta(days=warmup_days)
    visible_start = pd.Timestamp(start)

    ohlcv = _cached_ohlcv(ticker, timeframe, fetch_start.isoformat())
    spy_ohlcv = (_cached_ohlcv(BENCHMARK, timeframe, fetch_start.isoformat())
                 if st.session_state.pa_compare_spy and ticker != BENCHMARK
                 else None)

    if ohlcv.empty:
        st.warning(
            f"No price data for **{ticker}** ({timeframe}). "
            "Click **Update price data** to populate the local DB."
        )
    else:
        signal_row = signals.loc[ticker] if ticker in signals.index else None

        fig = build_etf_chart(
            ohlcv=ohlcv,
            ticker=ticker,
            timeframe=timeframe,
            signal_row=signal_row,
            show_rsi=st.session_state.pa_show_rsi,
            show_macd=st.session_state.pa_show_macd,
            show_bollinger=st.session_state.pa_show_bb,
            compare_to_spy=st.session_state.pa_compare_spy and ticker != BENCHMARK,
            spy_ohlcv=spy_ohlcv,
            visible_start=visible_start,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("##### Sector grid — click a ticker button to load it above")

    # Batch-load all 11 sector daily frames (mini-chart always uses the same
    # daily timeframe and the user-selected lookback, for visual consistency).
    sector_tickers = tuple(SECTOR_ETFS.keys())
    grid_frame = _cached_ohlcv_multi(sector_tickers, "1d", start.isoformat())

    # 3 columns × 4 rows; UX: candles on top, st.button under each as selector.
    grid_cols_per_row = 3
    rows_needed = (len(sector_tickers) + grid_cols_per_row - 1) // grid_cols_per_row
    for r in range(rows_needed):
        cols = st.columns(grid_cols_per_row)
        for c in range(grid_cols_per_row):
            i = r * grid_cols_per_row + c
            if i >= len(sector_tickers):
                continue
            tk = sector_tickers[i]
            state = (signals.loc[tk, "state"]
                     if tk in signals.index else "HOLD")
            if not grid_frame.empty and tk in grid_frame.columns.get_level_values(0):
                tk_frame = grid_frame[tk].dropna(how="all")
            else:
                tk_frame = pd.DataFrame()
            with cols[c]:
                mini = build_mini_chart(tk_frame, tk, state)
                st.plotly_chart(mini, use_container_width=True,
                                key=f"pa_mini_{tk}",
                                config={"displayModeBar": False})
                if st.button(f"View {tk}", key=f"pa_mini_btn_{tk}",
                             use_container_width=True):
                    st.session_state.pa_sector = tk
                    st.rerun()


with tab_expressions:
    from config.expressions import EXPRESSIONS

    section(
        "Expression Picker — what to actually buy when a sector fires BUY",
        help=(
            "Each sector signal (XLK, XLB, ...) maps to plain and operating-leverage "
            "equity ETFs. All expressions are unleveraged equity funds — operating "
            "leverage comes from the underlying businesses (e.g. miners' fixed costs), "
            "not from derivatives or daily rebalancing. `beta_hint` is a rough "
            "3-month price beta vs the signal ETF; use it to size positions, not to "
            "calculate anything."
        ),
    )

    # ---- Update price data (full universe, same wiring as Price Action) ----
    upd_col, _spacer = st.columns([1, 4])
    with upd_col:
        if st.button("🔄 Update price data", key="exp_update_btn"):
            prog_bar = st.progress(0.0)
            prog_caption = st.empty()
            tickers_to_update = _full_price_universe()
            total_steps = len(tickers_to_update) * 2  # 1d + 1wk
            done = {"n": 0}

            def _progress(tkr: str, tf: str, status: str) -> None:
                done["n"] += 1
                prog_bar.progress(done["n"] / total_steps)
                prog_caption.caption(f"{tkr} ({tf}): {status}")

            with st.spinner("Updating from yfinance…"):
                try:
                    update_all(tickers=tickers_to_update, progress=_progress)
                except Exception as e:
                    st.error(f"Update failed: {e}")
                else:
                    _cached_ohlcv.clear()
                    _cached_ohlcv_multi.clear()
                    _cached_sector_sparklines.clear()
                    _cached_expression_signals.clear()
                    st.success("Price data refreshed.")
                    st.rerun()

    # Reuse the cached bundle (raw signals are derived from it; refine_signals
    # adds the `state` column but leaves the underlying `signal` column intact).
    signals = _cached_signals_bundle(date.today().isoformat())

    buys = signals.index[signals["signal"] == "BUY"].tolist()
    if not buys:
        st.info("No BUY signals at the moment. The full expression map is shown below for reference.")
        sectors_to_show = list(EXPRESSIONS.keys())
    else:
        st.success(f"BUY signals: {', '.join(buys)}")
        sectors_to_show = buys + [s for s in EXPRESSIONS if s not in buys]

    today_iso = date.today().isoformat()
    for sector in sectors_to_show:
        is_buy = sector in buys
        prefix = "🟢" if is_buy else "⚪"
        with st.expander(f"{prefix} {sector} — {SECTOR_ETFS[sector]}", expanded=is_buy):
            spark_closes = _cached_sector_sparklines(sector, today_iso)
            missing = [t for t, vals in spark_closes.items() if not vals]
            if missing:
                st.caption(f"⚠ {len(missing)} ticker(s) missing price data "
                           f"({', '.join(missing)}) — click 🔄 Update price data above.")

            parent_state = str(signals["state"].get(sector, "HOLD"))
            exp_sigs = _cached_expression_signals(sector, parent_state, today_iso)
            sig_by_ticker = {d["ticker"]: d for d in exp_sigs}

            rows = []
            for e in EXPRESSIONS[sector]:
                s = sig_by_ticker.get(e.ticker, {"state": "NO_DATA", "reason": ""})
                rows.append({
                    "Ticker": e.ticker,
                    "Label": e.label,
                    "Kind": e.kind.replace("_", " "),
                    "β hint": f"{e.beta_hint:.2f}x",
                    "60d": spark_closes.get(e.ticker, []),
                    "Self-check": s["state"],
                    "Self-check reason": s["reason"],
                    "Note": e.note,
                })
            df_rows = pd.DataFrame(rows)
            column_order = ["Ticker", "Label", "Kind", "β hint", "60d",
                            "Self-check", "Self-check reason", "Note"]
            df_rows = df_rows[column_order]

            def _style_selfcheck(col: pd.Series) -> list[str]:
                out = []
                for v in col:
                    bg, fg = EXPRESSION_STATE_COLORS.get(str(v), ("", ""))
                    if bg:
                        out.append(f"background-color: {bg}; color: {fg}")
                    elif fg:
                        out.append(f"color: {fg}")
                    else:
                        out.append("")
                return out

            styled = df_rows.style.apply(_style_selfcheck, subset=["Self-check"])
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "60d": st.column_config.LineChartColumn(
                        "60d", width="medium",
                        help="Last 60 trading days of daily closes",
                    ),
                    "Self-check": st.column_config.TextColumn(
                        "Self-check", width="small",
                        help="Per-expression participation check vs the parent sector",
                    ),
                    "Self-check reason": st.column_config.TextColumn(
                        "Self-check reason", width="large",
                    ),
                },
            )

            with st.expander("How to read the Self-check column"):
                cutoff_pct = PARAMS.extension_pct_cutoff * 100
                st.markdown(
                    f"""
- **CONFIRMED** — Parent is NEW_BUY/HOLD_IF_LONG, the expression is above its
  own SMA200, its 3-month return ≥ the parent's, and its own extension is
  within the beta-scaled cutoff ({cutoff_pct:.0f}% × β). Safe participating vehicle.
- **LAGGING** — Parent BUY-class, expression up-trending and not extended, but
  3-month return < parent's. Vehicle is rising slower than the sector — pick a
  different expression.
- **STRETCHED** — Parent BUY-class, above own SMA200, but own extension >
  beta-scaled cutoff ({cutoff_pct:.0f}% × β). Too far above its own trend; wait.
- **BROKEN** — Parent BUY-class, but price < own SMA200. The vehicle is in
  its own downtrend regardless of the sector — avoid.
- **WARMING_UP** — Fewer than {PARAMS.sma_window} daily bars stored; SMA200
  not computable yet.
- **PARENT_INACTIVE** — Parent sector is not in NEW_BUY/HOLD_IF_LONG. No
  expression-level call — defer to the parent signal.
- **NO_DATA** — No price data stored for this ticker. Hit *🔄 Update price data*.
"""
                )

            # ---- click-through full chart ----
            # Selectbox is the sole driver: picking a ticker renders the chart
            # immediately; picking the sentinel hides it. No Show/Hide buttons.
            choices = [e.ticker for e in EXPRESSIONS[sector]]
            show_key = f"exp_showing_{sector}"
            HIDE = "— hide chart —"
            options = [HIDE] + choices
            default_index = 0
            if st.session_state.get(show_key) in choices:
                default_index = options.index(st.session_state[show_key])

            picked = st.selectbox(
                "View full chart for:", options,
                index=default_index,
                key=f"exp_select_{sector}",
                label_visibility="collapsed",
            )
            st.session_state[show_key] = picked if picked in choices else None

            active_ticker = st.session_state.get(show_key)
            if active_ticker:
                # Same warmup pattern as Price Action: load 6M + 300d warmup,
                # then slice to the visible 6M window inside build_etf_chart.
                warmup_days = 300
                visible_days = 183  # 6M
                fetch_start = date.today() - timedelta(days=visible_days + warmup_days)
                ohlcv_full = _cached_ohlcv(active_ticker, "1d",
                                           fetch_start.isoformat())
                if ohlcv_full.empty:
                    st.warning(
                        f"No price data stored for **{active_ticker}**. "
                        "Click 🔄 Update price data above."
                    )
                else:
                    visible_start = pd.Timestamp(
                        date.today() - timedelta(days=visible_days)
                    )
                    fig = build_etf_chart(
                        ohlcv_full, active_ticker, "1d",
                        signal_row=None,
                        visible_start=visible_start,
                    )
                    st.plotly_chart(
                        fig, use_container_width=True,
                        key=f"exp_chart_{sector}_{active_ticker}",
                    )


with tab_trend:
    section(
        "Sentiment Trend",
        help=(
            f"Weekly snapshots of the rolling-window aggregate sentiment, reconstructed "
            f"from your full ingest history. Window = {PARAMS.sentiment_lookback_days} days "
            f"(set in `config/settings.py`). NaN cells = no coverage in that window."
        ),
    )

    trend = _cached_trend(date.today().isoformat(), PARAMS.sentiment_lookback_days)

    if trend.empty or trend.dropna(how="all", axis=1).empty:
        st.info("No sentiment history yet. Ingest a few newsletters to populate the trend.")
    else:
        active = trend.dropna(how="all", axis=1)
        sector_labels = {t: f"{t} — {SECTOR_ETFS[t]}" for t in active.columns}

        st.markdown("##### Per-sector sentiment over time")
        line_df = active.rename(columns=sector_labels)
        st.line_chart(line_df, height=320, use_container_width=True)
        st.caption("BUY threshold = +2 (top), SELL threshold = −3 (bottom).")

        st.markdown("##### Sectors × weeks heatmap")
        try:
            import plotly.express as px
            heat = active.T  # rows = sectors, cols = weeks
            heat.index = [sector_labels[t] for t in heat.index]
            fig = px.imshow(
                heat,
                color_continuous_scale="RdYlGn",
                zmin=-5, zmax=5,
                aspect="auto",
                labels={"x": "Week", "y": "Sector", "color": "Score"},
            )
            fig.update_xaxes(tickformat="%Y-%m-%d")
            fig.update_layout(height=max(320, 36 * len(heat.index)),
                              margin=dict(l=4, r=4, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap unavailable: {e}")
            st.dataframe(active.round(2), use_container_width=True)

        with st.expander("Raw weekly snapshots"):
            st.dataframe(active.round(2), use_container_width=True)


with tab_inbox:
    section(
        "Gmail Inbox",
        help=(
            "Pulls unread mail matching your filter address, enriches with whitelisted "
            "links + PDF attachments, and pushes the assembled context through gpt-4o-mini. "
            "Each successful ingest also stamps the Gmail Message-ID so a re-run is a no-op."
        ),
    )

    if not gmail_configured():
        st.warning(
            "Gmail not configured. Add `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` to "
            "`.env`. Generate the app password at "
            "[myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) "
            "(2FA must be enabled on your Google account first). See SETUP.md for the full walkthrough."
        )
    else:
        c1, c2 = st.columns(2)
        c1.markdown(f"**Account:** `{GMAIL_ADDRESS}`")
        c2.markdown(f"**Filter:** `{GMAIL_FILTER_ADDRESS or '(none — all unread)'}`")

        b1, b2, b3 = st.columns([1, 1, 2])
        if b1.button("🔌 Test connection"):
            from src.gmail_client import test_connection
            with st.spinner("Connecting…"):
                ok, msg = test_connection()
            (st.success if ok else st.error)(msg)

        follow = b3.checkbox("Follow whitelisted links / PDFs",
                             value=True,
                             help="Disable for a faster, cheaper, body-only run.")
        mark_seen = b3.checkbox("Mark messages as read after ingesting",
                                value=True,
                                help="Required for incremental runs. Uncheck while testing.")

        if b2.button("📥 Fetch & parse all", type="primary"):
            with st.spinner("Fetching mail and calling OpenAI…"):
                try:
                    report = fetch_and_ingest(mark_seen=mark_seen, follow_links=follow)
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
                    report = []

            if not report:
                st.info("No unread messages matched the filter.")
            else:
                ingested = sum(1 for r in report if r["status"] == "ingested")
                skipped  = sum(1 for r in report if r["status"].startswith("skipped"))
                errored  = sum(1 for r in report if r["status"] == "error")
                m1, m2, m3 = st.columns(3)
                m1.metric("Ingested", ingested)
                m2.metric("Skipped (dupe)", skipped)
                m3.metric("Errors", errored)

                rep_df = pd.DataFrame(report)
                show_cols = [c for c in
                             ["status", "from", "date", "subject",
                              "sectors", "bias", "links_used", "pdfs_used",
                              "chars", "truncated", "error"]
                             if c in rep_df.columns]
                st.dataframe(rep_df[show_cols], use_container_width=True, hide_index=True)

                if ingested:
                    _cached_sentiment.clear()
                    _cached_trend.clear()
                    st.success(f"{ingested} new newsletter(s) ingested. "
                               f"Sentiment + trend caches cleared.")


with tab_ingest:
    section(
        "Ingest a Newsletter",
        help=(
            "Paste the body of a macro newsletter. GPT-4o-mini will extract a "
            "structured rating per sector and persist it to SQLite."
        ),
    )

    col_l, col_r = st.columns([3, 1])
    with col_r:
        author_hint = st.text_input("Author (optional)", "")
        date_hint = st.date_input("Publication date", date.today())
    with col_l:
        raw_text = st.text_area("Newsletter text", height=380,
                                placeholder="Paste the full text of the piece here…")

    if st.button("Parse & Save", type="primary", disabled=not raw_text.strip()):
        with st.spinner("Calling OpenAI…"):
            try:
                analysis, nid = ingest(raw_text, author_hint or None, date_hint)
            except Exception as e:
                st.error(f"Parse failed: {e}")
            else:
                if nid is None:
                    st.warning("Duplicate of an existing entry — nothing saved.")
                else:
                    st.success(f"Saved newsletter #{nid}")
                st.json({
                    "author": analysis.author,
                    "publication_date": analysis.publication_date.isoformat(),
                    "overall_macro_bias": analysis.overall_macro_bias.value,
                    "summary": analysis.summary,
                    "sector_ratings": [r.model_dump() for r in analysis.sector_ratings],
                })
                _cached_sentiment.clear()
                _cached_trend.clear()
                _cached_signal_history.clear()


with tab_history:
    section("Recent Newsletters")
    hist = recent_newsletters(50)
    if hist.empty:
        st.info("No newsletters ingested yet. Use the Ingest tab.")
    else:
        st.dataframe(hist, use_container_width=True, height=400)
        with st.expander("Delete an entry"):
            ids = hist["id"].tolist()
            target_id = st.selectbox("Newsletter id to delete", ids)
            if st.button("Delete", type="secondary"):
                delete_newsletter(int(target_id))
                _cached_sentiment.clear()
                _cached_trend.clear()
                _cached_signal_history.clear()
                st.success(f"Deleted #{target_id}")
                st.rerun()

    st.divider()
    st.subheader("Current Aggregate Sentiment (rolling window)")
    sent = _cached_sentiment(date.today().isoformat())
    if sent.empty:
        st.info("No sentiment in the rolling window.")
    else:
        sent_view = sent.copy()
        sent_view["sector"] = sent_view.index.map(SECTOR_ETFS)
        sent_view["score"] = sent_view["score"].map("{:+.2f}".format)
        st.dataframe(sent_view[["sector", "score", "n_obs"]].sort_values("n_obs", ascending=False),
                     use_container_width=True)
