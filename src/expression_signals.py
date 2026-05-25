"""Per-expression self-check signal.

Sits one tier below `signals.py`. When the parent sector fires NEW_BUY or
HOLD_IF_LONG, this module tells the user — for each candidate expression
vehicle inside that sector — whether the vehicle is participating, lagging,
broken, or overextended at its own level.

Pure functions. No IO, no globals, no streamlit imports. The sector-level
helper takes an `ohlcv_loader` callable so the caller decides where prices
come from (DB, cache, fixture).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import pandas as pd

from config.expressions import EXPRESSIONS, Expression
from config.settings import EXPRESSION, PARAMS
from config.themes import theme_for_ticker


BUY_CLASS_PARENT_STATES = {"NEW_BUY", "HOLD_IF_LONG"}

# Best-to-buy ordering for the picker. Lower = surface first.
_STATE_RANK = {
    "CONFIRMED": 0, "LAGGING": 1, "STRETCHED": 2, "WARMING_UP": 3,
    "PARENT_INACTIVE": 4, "BROKEN": 5, "NO_DATA": 6,
}


@dataclass(frozen=True)
class ExpressionSignal:
    ticker: str
    state: str
    reason: str
    above_own_sma: bool | None
    own_extension_pct: float | None
    own_return_3m: float | None
    parent_return_3m: float | None
    rs_vs_parent: float | None
    beta_scaled_cutoff: float | None
    # --- theme-news overlay (additive; never reorder) ---
    theme_key: str | None = None          # owning theme, None for plain proxies
    theme_sentiment: float | None = None  # blended newsletter+news score
    theme_n_obs: int = 0                  # contributing observations
    news_flag: str | None = None          # NEWS_CONTRADICTS | NEWS_DIVERGENCE | None


def _return_over(series: pd.Series, window: int) -> float | None:
    """Simple return over `window` bars; None if not enough history."""
    if series is None or len(series) < window + 1:
        return None
    start = float(series.iloc[-(window + 1)])
    end = float(series.iloc[-1])
    if start == 0:
        return None
    return end / start - 1.0


def _compute_technical_signal(
    expression: Expression,
    parent_state: str,
    expression_close: pd.Series,
    parent_close: pd.Series,
) -> ExpressionSignal:
    """Classify a single expression vs its parent sector on PRICE alone.

    See the module docstring for the seven-state vocabulary. Order matters:
    NO_DATA → WARMING_UP → PARENT_INACTIVE → BROKEN → STRETCHED → LAGGING →
    CONFIRMED. The expression whose ticker equals the parent sector
    mechanically falls through to CONFIRMED (rs_vs_parent == 0, rule 7 is
    strict `< 0`). Theme-news fields are left at defaults here; the public
    `compute_expression_signal` attaches them.
    """
    ticker = expression.ticker
    beta = float(expression.beta_hint)

    # 1. No data at all for this expression.
    if expression_close is None or len(expression_close) == 0:
        return ExpressionSignal(
            ticker=ticker,
            state="NO_DATA",
            reason="no price data stored — hit 🔄 Update price data",
            above_own_sma=None,
            own_extension_pct=None,
            own_return_3m=None,
            parent_return_3m=None,
            rs_vs_parent=None,
            beta_scaled_cutoff=None,
        )

    own_ret = _return_over(expression_close, PARAMS.momentum_window)
    parent_ret = (
        _return_over(parent_close, PARAMS.momentum_window)
        if parent_close is not None and len(parent_close) > 0
        else None
    )
    rs = (own_ret - parent_ret) if (own_ret is not None and parent_ret is not None) else None
    beta_scaled_cutoff = PARAMS.extension_pct_cutoff * beta

    # 2. Not enough history for SMA200.
    if len(expression_close) < PARAMS.sma_window:
        return ExpressionSignal(
            ticker=ticker,
            state="WARMING_UP",
            reason=(
                f"only {len(expression_close)} bars stored, "
                f"need {PARAMS.sma_window} for SMA200"
            ),
            above_own_sma=None,
            own_extension_pct=None,
            own_return_3m=own_ret,
            parent_return_3m=parent_ret,
            rs_vs_parent=rs,
            beta_scaled_cutoff=beta_scaled_cutoff,
        )

    sma200 = float(expression_close.tail(PARAMS.sma_window).mean())
    last = float(expression_close.iloc[-1])
    extension = (last - sma200) / sma200 if sma200 else 0.0
    above_sma = last > sma200

    # 3. Parent not in a BUY-class state. Still populate diagnostics so the
    # table can show RS/extension numbers for context.
    if parent_state not in BUY_CLASS_PARENT_STATES:
        return ExpressionSignal(
            ticker=ticker,
            state="PARENT_INACTIVE",
            reason=f"parent sector state is {parent_state}",
            above_own_sma=above_sma,
            own_extension_pct=extension,
            own_return_3m=own_ret,
            parent_return_3m=parent_ret,
            rs_vs_parent=rs,
            beta_scaled_cutoff=beta_scaled_cutoff,
        )

    # 5. Below own SMA200 — vehicle is in its own downtrend.
    if not above_sma:
        return ExpressionSignal(
            ticker=ticker,
            state="BROKEN",
            reason=(
                f"price {extension*100:+.1f}% below own SMA200 — "
                f"vehicle in own downtrend"
            ),
            above_own_sma=False,
            own_extension_pct=extension,
            own_return_3m=own_ret,
            parent_return_3m=parent_ret,
            rs_vs_parent=rs,
            beta_scaled_cutoff=beta_scaled_cutoff,
        )

    # 6. Too far above own SMA200 (beta-scaled).
    if extension > beta_scaled_cutoff:
        return ExpressionSignal(
            ticker=ticker,
            state="STRETCHED",
            reason=(
                f"extension {extension*100:+.1f}% > beta-scaled cutoff "
                f"{beta_scaled_cutoff*100:.1f}% "
                f"({PARAMS.extension_pct_cutoff*100:.0f}% × β {beta:.1f})"
            ),
            above_own_sma=True,
            own_extension_pct=extension,
            own_return_3m=own_ret,
            parent_return_3m=parent_ret,
            rs_vs_parent=rs,
            beta_scaled_cutoff=beta_scaled_cutoff,
        )

    # 7. Lagging the parent on 3m return.
    if rs is not None and rs < 0:
        return ExpressionSignal(
            ticker=ticker,
            state="LAGGING",
            reason=(
                f"3m {own_ret*100:+.1f}% vs parent {parent_ret*100:+.1f}%, "
                f"lagging by {rs*100:+.1f}%"
            ),
            above_own_sma=True,
            own_extension_pct=extension,
            own_return_3m=own_ret,
            parent_return_3m=parent_ret,
            rs_vs_parent=rs,
            beta_scaled_cutoff=beta_scaled_cutoff,
        )

    # 8. Participating.
    own_txt = f"{own_ret*100:+.1f}%" if own_ret is not None else "n/a"
    parent_txt = f"{parent_ret*100:+.1f}%" if parent_ret is not None else "n/a"
    return ExpressionSignal(
        ticker=ticker,
        state="CONFIRMED",
        reason=(
            f"3m {own_txt} (parent {parent_txt}), "
            f"ext {extension*100:+.1f}% (cutoff {beta_scaled_cutoff*100:.1f}%)"
        ),
        above_own_sma=True,
        own_extension_pct=extension,
        own_return_3m=own_ret,
        parent_return_3m=parent_ret,
        rs_vs_parent=rs,
        beta_scaled_cutoff=beta_scaled_cutoff,
    )


def _news_flag(state: str, theme_sentiment: float | None) -> str | None:
    """Flag a price-vs-news contradiction. Pure.

    NEWS_CONTRADICTS — vehicle is trending up (CONFIRMED/LAGGING) but theme news
        is clearly negative: the technicals may be late.
    NEWS_DIVERGENCE — vehicle is BROKEN (own downtrend) but theme news is clearly
        positive: a potential turn worth watching.
    """
    if theme_sentiment is None:
        return None
    thr = EXPRESSION.theme_news_flag_threshold
    if state in ("CONFIRMED", "LAGGING") and theme_sentiment <= -thr:
        return "NEWS_CONTRADICTS"
    if state == "BROKEN" and theme_sentiment >= thr:
        return "NEWS_DIVERGENCE"
    return None


def blend_theme_sentiment(
    nl_score: float | None, nl_n: int,
    news_score: float | None, news_n: int,
    weight: float | None = None,
) -> tuple[float | None, int]:
    """Combine newsletter + automated-news theme sentiment. Pure.

    `weight` is the news weight (newsletter weight = 1 - weight). When only one
    source has observations, return it untouched; when neither does, (None, 0).
    """
    weight = EXPRESSION.theme_news_weight if weight is None else weight
    have_nl = nl_n > 0 and nl_score is not None
    have_news = news_n > 0 and news_score is not None
    if have_nl and have_news:
        blended = (1.0 - weight) * float(nl_score) + weight * float(news_score)
        return blended, nl_n + news_n
    if have_nl:
        return float(nl_score), nl_n
    if have_news:
        return float(news_score), news_n
    return None, 0


def build_theme_sentiment_loader(
    newsletter_theme_df: pd.DataFrame | None,
    news_df: pd.DataFrame | None,
    weight: float | None = None,
) -> Callable[[str], tuple[float | None, int]]:
    """Build a `ticker -> (blended_theme_sentiment, n_obs)` loader from the two
    DB frames (db.aggregate_theme_sentiment and db.latest_theme_news). Pure: the
    DB reads happen in the caller; this just indexes the frames by the ticker's
    theme. Tickers with no theme (plain proxies) return (None, 0).
    """
    def _row(df, key, col):
        if df is None or df.empty or key not in df.index:
            return None
        val = df.loc[key, col]
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def loader(ticker: str) -> tuple[float | None, int]:
        theme = theme_for_ticker(ticker)
        if theme is None:
            return None, 0
        key = theme.key
        nl_score = _row(newsletter_theme_df, key, "score")
        nl_n = int(_row(newsletter_theme_df, key, "n_obs") or 0)
        news_score = _row(news_df, key, "score")
        news_n = int(_row(news_df, key, "n_headlines") or 0)
        return blend_theme_sentiment(nl_score, nl_n, news_score, news_n, weight)

    return loader


def compute_expression_signal(
    expression: Expression,
    parent_state: str,
    expression_close: pd.Series,
    parent_close: pd.Series,
    theme_sentiment: float | None = None,
    theme_n_obs: int = 0,
) -> ExpressionSignal:
    """Technical classification + theme-news overlay.

    `state`/`reason` and all price diagnostics come purely from price (so the
    technical behaviour is unchanged when no theme sentiment is supplied). The
    theme fields and `news_flag` are attached on top.
    """
    base = _compute_technical_signal(
        expression, parent_state, expression_close, parent_close)
    theme = theme_for_ticker(expression.ticker)
    return replace(
        base,
        theme_key=(theme.key if theme else None),
        theme_sentiment=theme_sentiment,
        theme_n_obs=theme_n_obs,
        news_flag=_news_flag(base.state, theme_sentiment),
    )


def rank_expressions(signals: list[ExpressionSignal]) -> list[ExpressionSignal]:
    """Order expressions best-to-buy: technical state first, then theme news as
    the tiebreaker (higher theme sentiment surfaces first). Pure, stable."""
    def key(s: ExpressionSignal):
        return (_STATE_RANK.get(s.state, 99),
                -(s.theme_sentiment if s.theme_sentiment is not None else 0.0))
    return sorted(signals, key=key)


def compute_expressions_for_sector(
    sector: str,
    parent_state: str,
    ohlcv_loader: Callable[[str], pd.Series],
    theme_sentiment_loader: Callable[[str], tuple[float | None, int]] | None = None,
) -> list[ExpressionSignal]:
    """Run `compute_expression_signal` for every Expression in EXPRESSIONS[sector].

    `ohlcv_loader(ticker)` returns an ascending close-price Series. The parent
    close is fetched via `ohlcv_loader(sector)`; if the parent itself has no
    data, every expression returns NO_DATA with a parent-specific reason.

    `theme_sentiment_loader(ticker)` (optional) returns `(theme_sentiment, n_obs)`
    for the ticker's theme; when omitted, expressions carry no theme overlay.
    """
    exprs = EXPRESSIONS.get(sector, [])
    if not exprs:
        return []

    def _theme(ticker: str) -> tuple[float | None, int]:
        return theme_sentiment_loader(ticker) if theme_sentiment_loader else (None, 0)

    parent_close = ohlcv_loader(sector)
    if parent_close is None or len(parent_close) == 0:
        out_nodata: list[ExpressionSignal] = []
        for e in exprs:
            ts, tn = _theme(e.ticker)
            theme = theme_for_ticker(e.ticker)
            out_nodata.append(ExpressionSignal(
                ticker=e.ticker,
                state="NO_DATA",
                reason="parent ETF has no price data — hit 🔄 Update price data",
                above_own_sma=None,
                own_extension_pct=None,
                own_return_3m=None,
                parent_return_3m=None,
                rs_vs_parent=None,
                beta_scaled_cutoff=None,
                theme_key=(theme.key if theme else None),
                theme_sentiment=ts,
                theme_n_obs=tn,
            ))
        return out_nodata

    out: list[ExpressionSignal] = []
    for e in exprs:
        own_close = ohlcv_loader(e.ticker)
        ts, tn = _theme(e.ticker)
        out.append(compute_expression_signal(
            e, parent_state, own_close, parent_close,
            theme_sentiment=ts, theme_n_obs=tn))
    return out
