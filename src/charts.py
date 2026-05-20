"""Chart helpers.

`compute_chart_overlays` is the original contract function that produces SMA /
relative-strength columns used both by the chart and by `compute_sector_metrics`,
so the chart and the signal table never disagree.

`build_etf_chart` and `build_mini_chart` are the Plotly figure builders for the
Price Action tab. Both are pure functions — data in, figure out. No streamlit
imports, no DB access, no globals beyond the state color palette.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators import sma, rsi as rsi_ind, macd as macd_ind, bollinger as bb_ind


# Shared state color palette — single source of truth (app.py imports this).
STATE_COLORS: dict[str, str] = {
    "NEW_BUY":      "#143d2a",  # green — fresh entry OK
    "HOLD_IF_LONG": "#3d3a14",  # amber — hold if owned, don't add
    "CHASE":        "#4a3214",  # orange — too late, don't enter
    "REDUCE":       "#3d1f14",  # rust — was BUY, now degraded
    "HOLD":         "",         # neutral — wait
    "SELL":         "#4a1818",  # red — exit
}

# Slightly brighter version of the same hues for borders / title text where
# the dark fill colors would be invisible against a dark page background.
STATE_ACCENTS: dict[str, str] = {
    "NEW_BUY":      "#2ecc71",
    "HOLD_IF_LONG": "#f1c40f",
    "CHASE":        "#e67e22",
    "REDUCE":       "#c97c5d",
    "HOLD":         "#888888",
    "SELL":         "#e74c3c",
}

# Sector names (kept local so charts.py is independent of config import order
# for plain plotting use; the lookup will fall back to "" if ticker is unknown).
_SECTOR_NAMES = {
    "XLK":  "Technology",
    "XLY":  "Consumer Discretionary",
    "XLC":  "Communication Services",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "SPY":  "Benchmark",
}

# Plot palette
_BG_DARK       = "#0e1117"
_PANEL_DARK    = "#11161d"
_GRID          = "rgba(255,255,255,0.08)"
_VOL_UP        = "#26a69a"
_VOL_DOWN      = "#ef5350"
_CANDLE_UP     = "#26a69a"
_CANDLE_DOWN   = "#ef5350"
_SMA200_COLOR  = "#f0e68c"
_SMA50_COLOR   = "#a0a0a0"
_BB_FILL       = "rgba(120,140,200,0.12)"
_BB_STROKE     = "rgba(120,140,200,0.45)"


def compute_chart_overlays(ohlcv: pd.DataFrame,
                           spy_close: pd.Series | None = None) -> pd.DataFrame:
    """Compute the overlays used by the price chart.

    Parameters
    ----------
    ohlcv : DataFrame indexed by DatetimeIndex with at least a 'close' column
            (the shape returned by `price_store.load_ohlcv`).
    spy_close : optional SPY close series, indexed by DatetimeIndex. When
                provided, a 63-trading-day rolling relative strength is
                computed (sector 3m return minus SPY 3m return).

    Returns
    -------
    DataFrame aligned to `ohlcv.index` with columns:
        sma200 : 200-period simple moving average of close
        sma50  : 50-period simple moving average of close
        rs_3m  : sector 63-day return - SPY 63-day return, or NaN if no SPY

    The SMA200 and 63-day return definitions are the same ones used in
    `market_engine.compute_sector_metrics`, so a snapshot taken from the last
    row of `compute_chart_overlays(load_ohlcv(...))` will agree to the cent
    with the signal table for the same `as_of` date.
    """
    if ohlcv.empty:
        return pd.DataFrame(columns=["sma200", "sma50", "rs_3m"])

    close = ohlcv["close"]
    out = pd.DataFrame(index=ohlcv.index)
    out["sma200"] = sma(close, 200)
    out["sma50"] = sma(close, 50)

    if spy_close is not None and not spy_close.empty:
        spy_aligned = spy_close.reindex(close.index).ffill()
        sector_ret = close / close.shift(63) - 1.0
        spy_ret = spy_aligned / spy_aligned.shift(63) - 1.0
        out["rs_3m"] = sector_ret - spy_ret
    else:
        out["rs_3m"] = pd.Series(index=ohlcv.index, dtype="float64")

    return out


# ---------------------------------------------------------------------------
# Main price chart
# ---------------------------------------------------------------------------

def _row_layout(show_rsi: bool, show_macd: bool) -> tuple[list[float], dict[str, int]]:
    """Return (row_heights, panel_index_map). Index map keys: main, volume,
    rsi, macd (rsi/macd absent if their flag is False). Row indices are 1-based.
    """
    if show_rsi and show_macd:
        heights = [0.50, 0.15, 0.17, 0.18]
        idx = {"main": 1, "volume": 2, "rsi": 3, "macd": 4}
    elif show_rsi:
        heights = [0.55, 0.15, 0.30]
        idx = {"main": 1, "volume": 2, "rsi": 3}
    elif show_macd:
        heights = [0.55, 0.15, 0.30]
        idx = {"main": 1, "volume": 2, "macd": 3}
    else:
        heights = [0.75, 0.25]
        idx = {"main": 1, "volume": 2}
    return heights, idx


def build_etf_chart(
    ohlcv: pd.DataFrame,
    ticker: str,
    timeframe: str,
    signal_row: pd.Series | None = None,
    show_rsi: bool = False,
    show_macd: bool = False,
    show_bollinger: bool = False,
    compare_to_spy: bool = False,
    spy_ohlcv: pd.DataFrame | None = None,
    visible_start: pd.Timestamp | None = None,
) -> go.Figure:
    """Build the main candlestick + indicators figure for the Price Action tab.

    Pure function. Returns a Plotly figure ready to hand to st.plotly_chart.

    `visible_start`: if provided, all indicators (SMA200/50, RSI, MACD, BB)
    are computed against the FULL input `ohlcv` frame for proper warmup, and
    only bars with `index >= visible_start` are displayed. This avoids leading
    NaN regions on short-lookback views (e.g. 6M daily where SMA200 needs ~10
    months of prior history). When None, behaviour is unchanged.
    """
    row_heights, panel = _row_layout(show_rsi, show_macd)
    n_rows = len(row_heights)

    # Build subplot grid. Main panel uses secondary_y for the SPY comparison.
    specs: list[list[dict]] = [[{"secondary_y": True}]]
    for _ in range(n_rows - 1):
        specs.append([{"secondary_y": False}])

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
        specs=specs,
    )

    if ohlcv.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=_BG_DARK,
            plot_bgcolor=_PANEL_DARK,
            title=f"{ticker} — no data",
            height=600,
        )
        return fig

    # Compute indicators on the FULL frame (warmup-aware), then mask the
    # visible window. `vis_mask` selects what gets drawn; everything else is
    # used only to bootstrap the rolling windows.
    if visible_start is not None:
        vis_mask = ohlcv.index >= pd.Timestamp(visible_start)
        if not vis_mask.any():
            vis_mask = ohlcv.index >= ohlcv.index[0]  # safety: show everything
    else:
        vis_mask = pd.Series(True, index=ohlcv.index).values

    full_close = ohlcv["close"]
    sma200_full = sma(full_close, 200)
    sma50_full = sma(full_close, 50)

    visible_ohlcv = ohlcv.loc[vis_mask]
    idx = visible_ohlcv.index
    sma200 = sma200_full.loc[vis_mask]
    sma50 = sma50_full.loc[vis_mask]

    # ---- main: candles ----
    fig.add_trace(
        go.Candlestick(
            x=idx,
            open=visible_ohlcv["open"], high=visible_ohlcv["high"],
            low=visible_ohlcv["low"], close=visible_ohlcv["close"],
            increasing_line_color=_CANDLE_UP,
            decreasing_line_color=_CANDLE_DOWN,
            increasing_fillcolor=_CANDLE_UP,
            decreasing_fillcolor=_CANDLE_DOWN,
            name=ticker,
            showlegend=False,
        ),
        row=panel["main"], col=1, secondary_y=False,
    )

    # ---- main: SMAs ----
    fig.add_trace(
        go.Scatter(
            x=idx, y=sma200, mode="lines",
            name="SMA 200",
            line=dict(color=_SMA200_COLOR, width=2),
            hovertemplate="SMA200: %{y:.2f}<extra></extra>",
        ),
        row=panel["main"], col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=idx, y=sma50, mode="lines",
            name="SMA 50",
            line=dict(color=_SMA50_COLOR, width=1),
            opacity=0.6,
            hovertemplate="SMA50: %{y:.2f}<extra></extra>",
        ),
        row=panel["main"], col=1, secondary_y=False,
    )

    # ---- main: Bollinger overlay ----
    if show_bollinger:
        bb_full = bb_ind(full_close, period=20, num_std=2.0)
        bb = bb_full.loc[vis_mask]
        # Upper first, then lower with fill='tonexty' for the band.
        fig.add_trace(
            go.Scatter(
                x=idx, y=bb["upper"], mode="lines",
                name="BB upper",
                line=dict(color=_BB_STROKE, width=1),
                hovertemplate="BB upper: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=panel["main"], col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=bb["lower"], mode="lines",
                name="BB lower",
                line=dict(color=_BB_STROKE, width=1),
                fill="tonexty", fillcolor=_BB_FILL,
                hovertemplate="BB lower: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=panel["main"], col=1, secondary_y=False,
        )

    # ---- main: SPY comparison on secondary y ----
    # Rebase SPY at the VISIBLE window start, not at the warmup start.
    if compare_to_spy and spy_ohlcv is not None and not spy_ohlcv.empty:
        spy_close = spy_ohlcv["close"].reindex(idx).ffill()
        first_valid = spy_close.dropna()
        if not first_valid.empty:
            base = float(first_valid.iloc[0])
            spy_rebased = spy_close / base * 100.0
            fig.add_trace(
                go.Scatter(
                    x=idx, y=spy_rebased, mode="lines",
                    name="SPY (rebased=100)",
                    line=dict(color="#6cb4ff", width=1.4, dash="dot"),
                    hovertemplate="SPY: %{y:.1f}<extra></extra>",
                ),
                row=panel["main"], col=1, secondary_y=True,
            )
            fig.update_yaxes(title_text="SPY idx", row=panel["main"], col=1,
                             secondary_y=True, showgrid=False,
                             color="#6cb4ff")

    # ---- state-aware overlay on main ----
    # Use the LAST value of the warmup-aware SMA200 so the floor / CHASE band
    # always anchors to a real number even on short lookbacks.
    state = ""
    last_close = float(visible_ohlcv["close"].iloc[-1])
    last_sma200_full = (float(sma200_full.iloc[-1])
                       if pd.notna(sma200_full.iloc[-1]) else None)
    ext_pct = None
    if signal_row is not None:
        state = str(signal_row.get("state", ""))
        ext_pct = signal_row.get("extension_pct", None)
        if ext_pct is not None and pd.isna(ext_pct):
            ext_pct = None

        if state in ("NEW_BUY", "HOLD_IF_LONG") and last_sma200_full is not None:
            fig.add_hline(
                y=last_sma200_full,
                line=dict(color=STATE_ACCENTS.get(state, "#888"),
                          dash="dash", width=1.2),
                annotation_text="entry-quality floor (SMA200)",
                annotation_position="top left",
                annotation_font=dict(size=10,
                                     color=STATE_ACCENTS.get(state, "#888")),
                row=panel["main"], col=1,
            )
        elif state == "CHASE" and last_sma200_full is not None:
            cutoff = last_sma200_full * 1.12
            y_top = float(visible_ohlcv["high"].max())
            y_top = max(y_top, cutoff * 1.02)
            fig.add_shape(
                type="rect",
                xref=f"x{panel['main']}" if panel["main"] > 1 else "x",
                yref=f"y{panel['main']}" if panel["main"] > 1 else "y",
                x0=idx[0], x1=idx[-1],
                y0=cutoff, y1=y_top,
                fillcolor="rgba(230,126,34,0.14)",
                line=dict(width=0),
                layer="below",
            )

    # ---- volume panel ----
    up = visible_ohlcv["close"] >= visible_ohlcv["open"]
    vol_colors = np.where(up, _VOL_UP, _VOL_DOWN)
    fig.add_trace(
        go.Bar(
            x=idx, y=visible_ohlcv["volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ),
        row=panel["volume"], col=1,
    )
    # Clip volume y-axis if there is a huge spike.
    vol = visible_ohlcv["volume"].astype(float)
    if not vol.empty:
        med = float(vol.median())
        mx = float(vol.max())
        if med > 0 and mx > 5 * med:
            cap = float(vol.quantile(0.99))
            fig.update_yaxes(range=[0, cap * 1.05],
                             row=panel["volume"], col=1)

    # ---- RSI panel ----
    if show_rsi:
        rsi_s = rsi_ind(full_close, period=14).loc[vis_mask]
        fig.add_trace(
            go.Scatter(
                x=idx, y=rsi_s, mode="lines",
                name="RSI(14)",
                line=dict(color="#bb86fc", width=1.3),
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
                showlegend=False,
            ),
            row=panel["rsi"], col=1,
        )
        fig.add_hline(y=70, line=dict(color="rgba(239,83,80,0.5)", dash="dot", width=1),
                      row=panel["rsi"], col=1)
        fig.add_hline(y=30, line=dict(color="rgba(38,166,154,0.5)", dash="dot", width=1),
                      row=panel["rsi"], col=1)
        fig.update_yaxes(range=[0, 100], row=panel["rsi"], col=1,
                         title_text="RSI")

    # ---- MACD panel ----
    if show_macd:
        m = macd_ind(full_close, fast=12, slow=26, signal=9).loc[vis_mask]
        hist_colors = np.where(m["histogram"] >= 0, _VOL_UP, _VOL_DOWN)
        fig.add_trace(
            go.Bar(
                x=idx, y=m["histogram"],
                marker_color=hist_colors,
                name="MACD hist",
                showlegend=False,
                hovertemplate="hist: %{y:.3f}<extra></extra>",
                opacity=0.6,
            ),
            row=panel["macd"], col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=m["macd"], mode="lines",
                name="MACD",
                line=dict(color="#4fc3f7", width=1.3),
                hovertemplate="MACD: %{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=panel["macd"], col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=m["signal"], mode="lines",
                name="signal",
                line=dict(color="#ffb74d", width=1.2, dash="dot"),
                hovertemplate="sig: %{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=panel["macd"], col=1,
        )
        fig.update_yaxes(title_text="MACD", row=panel["macd"], col=1)

    # ---- title ----
    sector_name = _SECTOR_NAMES.get(ticker, "")
    title_bits = [f"<b>{ticker}</b>"]
    if sector_name:
        title_bits.append(sector_name)
    if state:
        accent = STATE_ACCENTS.get(state, "#ccc")
        title_bits.append(
            f"<span style='color:{accent}'>{state}</span>"
        )
    title_bits.append(f"${last_close:,.2f}")
    if ext_pct is not None:
        title_bits.append(f"ext {float(ext_pct)*100:+.1f}% / 12%")
    title_text = " · ".join(title_bits)

    # ---- layout ----
    height = 720 + (120 if show_rsi else 0) + (120 if show_macd else 0)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG_DARK,
        plot_bgcolor=_PANEL_DARK,
        title=dict(text=title_text, x=0.01, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=30),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.005,
            xanchor="right", x=1.0,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.0,
        hovermode="x unified",
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False,
                     rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)

    # X-axis: skip weekends on daily only.
    if timeframe == "1d":
        for r in range(1, n_rows + 1):
            fig.update_xaxes(
                rangebreaks=[dict(bounds=["sat", "mon"])],
                row=r, col=1,
            )

    # Y-axis titles
    fig.update_yaxes(title_text="Price ($)",
                     row=panel["main"], col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=panel["volume"], col=1)

    return fig


# ---------------------------------------------------------------------------
# Mini chart (for the sector grid)
# ---------------------------------------------------------------------------

def build_mini_chart(ohlcv: pd.DataFrame, ticker: str, state: str) -> go.Figure:
    """Tiny candles-only chart for the mini-grid. 150px tall, ticker label only,
    state-colored border via paper_bgcolor trick.
    """
    border_color = STATE_ACCENTS.get(state, "#444") if state else "#444"

    fig = go.Figure()
    if ohlcv.empty:
        fig.update_layout(
            paper_bgcolor=border_color,
            plot_bgcolor=_PANEL_DARK,
            height=150,
            margin=dict(l=4, r=4, t=4, b=4),
            annotations=[dict(text=f"{ticker} — no data",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False,
                              font=dict(size=11, color="#aaa"))],
        )
        return fig

    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["open"], high=ohlcv["high"],
            low=ohlcv["low"], close=ohlcv["close"],
            increasing_line_color=_CANDLE_UP,
            decreasing_line_color=_CANDLE_DOWN,
            increasing_fillcolor=_CANDLE_UP,
            decreasing_fillcolor=_CANDLE_DOWN,
            showlegend=False,
            name=ticker,
        ),
    )
    fig.update_layout(
        paper_bgcolor=border_color,   # acts as the "border"
        plot_bgcolor=_PANEL_DARK,
        height=150,
        margin=dict(l=4, r=4, t=4, b=4),
        showlegend=False,
        xaxis=dict(visible=False, rangeslider=dict(visible=False)),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=f"<b>{ticker}</b>",
                x=0.02, y=0.97, xref="paper", yref="paper",
                showarrow=False, xanchor="left", yanchor="top",
                font=dict(size=11, color="#eee"),
                bgcolor="rgba(0,0,0,0.35)",
                borderpad=2,
            ),
        ],
        hovermode="x",
    )
    return fig
