"""Streamlit dashboard — entrypoint."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from config.settings import (
    BENCHMARK, GMAIL_ADDRESS, GMAIL_FILTER_ADDRESS, PARAMS,
    SECTOR_ETFS, gmail_configured, tiger_configured,
)
from src.db import aggregate_sentiment, delete_newsletter, init_db, recent_newsletters
from src.market_engine import (
    compute_sector_metrics, fetch_macro_prices, fetch_prices,
    gold_oil_ratio, yield_curve_spread,
)
from src.nlp_pipeline import fetch_and_ingest, ingest
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

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_prices() -> pd.DataFrame:
    return fetch_prices(list(SECTOR_ETFS.keys()) + [BENCHMARK])

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_macro_prices() -> pd.DataFrame:
    return fetch_macro_prices()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_yield_curve() -> dict:
    return yield_curve_spread()

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


_STATE_COLORS = {
    "NEW_BUY":      "#143d2a",  # green — fresh entry OK
    "HOLD_IF_LONG": "#3d3a14",  # amber — hold if owned, don't add
    "CHASE":        "#4a3214",  # orange — too late, don't enter
    "REDUCE":       "#3d1f14",  # rust — was BUY, now degraded
    "HOLD":         "",         # neutral — wait
    "SELL":         "#4a1818",  # red — exit
}

def _signal_row_style(row: pd.Series) -> list[str]:
    state = row.get("State", row.get("state", row.get("Signal", row.get("signal", ""))))
    color = _STATE_COLORS.get(state, "")
    return [f"background-color: {color}; color: #eee" if color else "" for _ in row]


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%" if pd.notna(x) else "—"


st.title("📊 Sector Rotation — Macro-Filtered Convergence Model")
st.caption(
    f"11 US SPDR Select Sector ETFs · benchmark **{BENCHMARK}** · "
    f"weekly cadence · BUY threshold sentiment >= +{PARAMS.buy_sentiment_threshold:.0f}, "
    f"SELL <= {PARAMS.sell_sentiment_threshold:+.0f}"
)

tab_dashboard, tab_expressions, tab_trend, tab_inbox, tab_ingest, tab_history = st.tabs(
    ["📈 Dashboard", "🎯 Expressions", "✨ Trend", "📧 Inbox",
     "📥 Ingest Newsletter", "🗂 History"]
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

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Sector Relative Strength Matrix")

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
        st.dataframe(styled, use_container_width=True, height=460)

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

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col, state in zip(
            [c1, c2, c3, c4, c5, c6],
            ["NEW_BUY", "HOLD_IF_LONG", "CHASE", "REDUCE", "HOLD", "SELL"],
        ):
            col.metric(state, int((signals["state"] == state).sum()))

        with st.expander("Target weights (equal-weight across NEW_BUY + HOLD_IF_LONG, 5% cash buffer)"):
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

    with right:
        st.subheader("Macro Regime Indicators")

        macro_prices = _cached_macro_prices()
        gor = gold_oil_ratio(macro_prices)
        yc = _cached_yield_curve()

        m1, m2 = st.columns(2)
        if pd.notna(gor["current"]):
            m1.metric(
                "Gold / Oil",
                f"{gor['current']:.1f}",
                delta=f"z={gor['z_score_1y']:+.2f}",
                help="GC=F / CL=F. Higher = risk-off bid for gold / weak oil demand.",
            )
        else:
            m1.metric("Gold / Oil", "—")

        if pd.notna(yc["current"]):
            slope_per_month = yc["slope_30d"] * 30
            m2.metric(
                "10Y - 2Y (FRED)",
                f"{yc['current']:+.2f}%",
                delta=f"{slope_per_month:+.2f}%/mo (30d)",
                help="Treasury 10Y minus 2Y.",
            )
        else:
            m2.metric("10Y - 2Y", "—", help=yc.get("error", ""))

        if "series" in gor:
            st.line_chart(gor["series"].tail(180), height=140, use_container_width=True)
        if "series" in yc:
            st.line_chart(yc["series"], height=140, use_container_width=True)

        st.divider()
        st.subheader("Tiger Portfolio Drift")

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

    if st.button("🔄 Force refresh all caches"):
        st.cache_data.clear()
        st.rerun()


with tab_expressions:
    from config.expressions import EXPRESSIONS

    st.subheader("Expression Picker — what to actually buy when a sector fires BUY")
    st.caption(
        "Each sector signal (XLK, XLB, ...) maps to plain and operating-leverage "
        "equity ETFs. All expressions are unleveraged equity funds — operating "
        "leverage comes from the underlying businesses (e.g. miners' fixed costs), "
        "not from derivatives or daily rebalancing. `beta_hint` is a rough "
        "3-month price beta vs the signal ETF; use it to size positions, not to "
        "calculate anything."
    )

    prices = _cached_prices()
    metrics = compute_sector_metrics(prices)
    sentiment = _cached_sentiment(date.today().isoformat())
    signals = build_signals(metrics, sentiment)

    buys = signals.index[signals["signal"] == "BUY"].tolist()
    if not buys:
        st.info("No BUY signals at the moment. The full expression map is shown below for reference.")
        sectors_to_show = list(EXPRESSIONS.keys())
    else:
        st.success(f"BUY signals: {', '.join(buys)}")
        sectors_to_show = buys + [s for s in EXPRESSIONS if s not in buys]

    for sector in sectors_to_show:
        is_buy = sector in buys
        prefix = "🟢" if is_buy else "⚪"
        with st.expander(f"{prefix} {sector} — {SECTOR_ETFS[sector]}", expanded=is_buy):
            rows = [
                {
                    "Ticker": e.ticker,
                    "Label": e.label,
                    "Kind": e.kind.replace("_", " "),
                    "β hint": f"{e.beta_hint:.2f}x",
                    "Note": e.note,
                }
                for e in EXPRESSIONS[sector]
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab_trend:
    st.subheader("Sentiment Trend")
    st.caption(
        f"Weekly snapshots of the rolling-window aggregate sentiment, reconstructed "
        f"from your full ingest history. Window = {PARAMS.sentiment_lookback_days} days "
        f"(set in `config/settings.py`). NaN cells = no coverage in that window."
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
    st.subheader("Gmail Inbox")
    st.caption(
        "Pulls unread mail matching your filter address, enriches with whitelisted "
        "links + PDF attachments, and pushes the assembled context through gpt-4o-mini. "
        "Each successful ingest also stamps the Gmail Message-ID so a re-run is a no-op."
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
    st.subheader("Ingest a Newsletter")
    st.caption("Paste the body of a macro newsletter. GPT-4o-mini will extract a "
               "structured rating per sector and persist it to SQLite.")

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
    st.subheader("Recent Newsletters")
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
