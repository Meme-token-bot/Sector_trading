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

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config.expressions import EXPRESSIONS, Expression
from config.settings import PARAMS


BUY_CLASS_PARENT_STATES = {"NEW_BUY", "HOLD_IF_LONG"}


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


def _return_over(series: pd.Series, window: int) -> float | None:
    """Simple return over `window` bars; None if not enough history."""
    if series is None or len(series) < window + 1:
        return None
    start = float(series.iloc[-(window + 1)])
    end = float(series.iloc[-1])
    if start == 0:
        return None
    return end / start - 1.0


def compute_expression_signal(
    expression: Expression,
    parent_state: str,
    expression_close: pd.Series,
    parent_close: pd.Series,
) -> ExpressionSignal:
    """Classify a single expression vs its parent sector.

    See the module docstring for the seven-state vocabulary. Order matters:
    NO_DATA → WARMING_UP → PARENT_INACTIVE → BROKEN → STRETCHED → LAGGING →
    CONFIRMED. The expression whose ticker equals the parent sector
    mechanically falls through to CONFIRMED (rs_vs_parent == 0, rule 7 is
    strict `< 0`).
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


def compute_expressions_for_sector(
    sector: str,
    parent_state: str,
    ohlcv_loader: Callable[[str], pd.Series],
) -> list[ExpressionSignal]:
    """Run `compute_expression_signal` for every Expression in EXPRESSIONS[sector].

    `ohlcv_loader(ticker)` returns an ascending close-price Series. The parent
    close is fetched via `ohlcv_loader(sector)`; if the parent itself has no
    data, every expression returns NO_DATA with a parent-specific reason.
    """
    exprs = EXPRESSIONS.get(sector, [])
    if not exprs:
        return []

    parent_close = ohlcv_loader(sector)
    if parent_close is None or len(parent_close) == 0:
        return [
            ExpressionSignal(
                ticker=e.ticker,
                state="NO_DATA",
                reason="parent ETF has no price data — hit 🔄 Update price data",
                above_own_sma=None,
                own_extension_pct=None,
                own_return_3m=None,
                parent_return_3m=None,
                rs_vs_parent=None,
                beta_scaled_cutoff=None,
            )
            for e in exprs
        ]

    out: list[ExpressionSignal] = []
    for e in exprs:
        own_close = ohlcv_loader(e.ticker)
        out.append(compute_expression_signal(e, parent_state, own_close, parent_close))
    return out
