"""Historical backtest of the mechanical sector-rotation core.

What this module does
---------------------
Walks the local ``prices.db`` history week-by-week, computes the *same* signal
pipeline the live dashboard uses (``compute_sector_metrics`` →
``build_signals`` → ``refine_signals``), forms an equal-weight portfolio across
the BUY-class states (NEW_BUY + HOLD_IF_LONG with a configurable cash buffer),
and tracks an equity curve vs SPY buy-and-hold.

What this module does NOT do
----------------------------
- It does NOT fabricate historical newsletter sentiment. The live model
  requires sentiment ≥ +2 to buy; the project only has ~2 weeks of meaningful
  newsletter coverage. So the mechanical core is run with the sentiment
  gate *bypassed* (sentiment is synthesised as +2 so the convergence rule
  passes on that dimension). The result measures trend + relative strength +
  staleness + extension — i.e., everything except the sentiment overlay.
  See BACKTEST_REPORT.md for the honest framing.
- It does NOT apply the macro veto. Historical macro readings (FRED) are not
  in ``prices.db``; threading them through would change what's being measured
  and obscure look-ahead audit. The macro overlay is documented as a *current-
  date* filter and tested forward in Step 3 (signal_snapshots).

No look-ahead
-------------
At each weekly rebalance date ``t``:
  * metrics are computed by ``compute_sector_metrics(prices, as_of=t)``
    which slices the price frame with ``loc[:t]`` before any rolling.
  * signal history (for staleness / REDUCE detection) is built incrementally
    from prior weekly snapshots only.
  * fills happen on the NEXT trading day's open (default), or the same
    close (configurable). The signal is *never* allowed to read a bar
    dated after ``t``.

Outputs
-------
``BacktestResult`` carries the equity curve, headline stats, turnover, and
trade log; ``run_backtest()`` is the public entrypoint. ``save_equity_csv``
writes the strategy + SPY series to ``data/<name>.csv`` for the dashboard.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from config.settings import BENCHMARK, DATA_DIR, PARAMS, SECTOR_ETFS, SUPPLEMENTARY_SECTORS
from src.market_engine import compute_sector_metrics
from src.price_store import load_ohlcv_multi
from src.signals import build_signals, refine_signals, target_weights


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Knobs for `run_backtest`. All defaults match the live model where
    applicable; sentiment_gate defaults to 'off' because there is no historical
    sentiment to gate on (see module docstring)."""
    start: date | None = None           # None => earliest tradeable week
    end: date | None = None             # None => last full week in price DB
    execution: Literal["next_open", "same_close"] = "next_open"
    cost_bps: float = 5.0               # round-trip cost = 2x this (each side)
    slippage_bps: float = 0.0           # additive to cost_bps per side
    cash_buffer: float = 0.05           # matches live target_weights default
    initial_capital: float = 100_000.0
    # Trade policy:
    #   "event_driven" (default, matches live behaviour) — buy on transition
    #     into BUY-class (NEW_BUY/HOLD_IF_LONG), sell on transition out
    #     (SELL/REDUCE/CHASE/HOLD). Position sized as nav * target_weight at
    #     the fill bar. NO mid-position rebalancing — accept drift between
    #     entry and exit. This is what the live dashboard's orders panel does.
    #   "rebalance_to_target" — every rebal, drag every held position back
    #     to the target weight. More principled "equal weight" but generates
    #     a flurry of small adjust trades each week.
    trade_policy: Literal["event_driven", "rebalance_to_target"] = "event_driven"
    # 'off' synthesises sentiment = +2.0 so the sentiment leg of build_signals
    #   passes automatically — pure mechanical core. This is the headline
    #   honest backtest given the data available.
    # 'on' uses aggregate_sentiment() at each rebalance date — only useful
    #   over the last ~3 months where there's any coverage at all.
    sentiment_gate: Literal["off", "on"] = "off"
    # Override fields on `config.settings.PARAMS` for the duration of this
    # backtest only. Keyed by SignalParams field name; values must match the
    # field's type. Used by the walk-forward sweep so a single PARAMS object
    # doesn't have to be physically mutated across hundreds of runs.
    param_overrides: dict | None = None
    # CHASE-as-partial knob. When None, defers to `PARAMS.chase_weight_fraction`
    # (the live default selected by walk-forward). Override to a float to
    # force a value for sweeps / sensitivity tests. 0 fully excludes CHASE.
    chase_weight_fraction: float | None = None
    # Regime-aware bull overlay. When True, on each rebalance date in a
    # confirmed STRONG uptrend, CHASE-state leaders are promoted to full
    # confirmed weight and the cash buffer drops to `bull_cash_buffer`.
    # Otherwise the defensive defaults are untouched.
    #
    # The first cut keyed off src.regime_analysis's BULL band (SPY within 5% of
    # its trailing 252-day high) and was net-NEGATIVE on the full window: that
    # band also covers the euphoric run-up right before a top, so going fully
    # invested + promoting extended leaders there raised bull DOWN-capture
    # (0.83→0.93) and degraded drawdown protection (8/11→7/11 wins). The strong
    # gate below instead requires BOTH:
    #   * SPY above a RISING `bull_sma_window`-day SMA (trend confirmed, not a
    #     dead-cat bounce), AND
    #   * SPY within `bull_proximity_pct` of its trailing 252-day high (near the
    #     highs, i.e. an established advance — not a deep-but-recovering tape).
    # Both conditions read only bars <= the rebalance date (no look-ahead).
    # Default False = exact current behaviour (no-op).
    regime_aware: bool = False
    bull_cash_buffer: float = 0.0
    bull_proximity_pct: float = 0.02   # within 2% of the trailing 252d high
    bull_sma_window: int = 200         # SPY must be above a RISING SMA of this


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity: pd.Series                   # strategy equity, indexed by date
    benchmark_equity: pd.Series         # SPY buy-and-hold, same index
    trades: pd.DataFrame                # one row per executed fill
    stats: dict                         # CAGR, vol, sharpe, MDD, etc.
    weights_history: pd.DataFrame       # one row per rebalance date
    states_history: pd.DataFrame        # refined state per ticker per rebal


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Only the 11 SPDR sectors participate in the live equal-weight target.
# Supplementary tickers (UFO) are tactical overlays in the UI — exclude here
# to match `target_weights()`.
def _universe() -> list[str]:
    return [t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS]


def load_price_panel(tickers: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (closes, opens) DataFrames indexed by date, columns = tickers.

    Pulls from the local ``prices.db`` cache via ``load_ohlcv_multi``. Both
    frames are aligned to the same union-of-dates index; missing bars are NaN.
    The benchmark (SPY) is always included.
    """
    tickers = list(tickers) if tickers is not None else _universe() + [BENCHMARK]
    if BENCHMARK not in tickers:
        tickers = list(tickers) + [BENCHMARK]
    wide = load_ohlcv_multi(tickers, "1d")
    if wide.empty:
        raise RuntimeError("prices.db has no daily bars for the requested tickers")
    closes = wide.xs("close", axis=1, level=1).sort_index()
    opens = wide.xs("open", axis=1, level=1).sort_index()
    closes.index = pd.to_datetime(closes.index)
    opens.index = pd.to_datetime(opens.index)
    return closes, opens


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------

def weekly_rebalance_dates(closes: pd.DataFrame,
                           start: date | None,
                           end: date | None) -> list[pd.Timestamp]:
    """Return the last trading day of each calendar week within [start, end].

    A trading day is "any date with at least one non-NaN close among the
    universe + benchmark". The last such date per ISO week is the rebalance
    point — equivalent to the live model's "weekly close" cadence and
    robust to holidays.
    """
    idx = closes.dropna(how="all").index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) == 0:
        return []
    iso = pd.Series(idx, index=idx).groupby([idx.isocalendar().year, idx.isocalendar().week]).last()
    return list(pd.DatetimeIndex(iso.values))


# ---------------------------------------------------------------------------
# Signal generation at a single rebalance date
# ---------------------------------------------------------------------------

def _synthetic_sentiment(tickers: Iterable[str]) -> pd.DataFrame:
    """All-positive sentiment frame so `build_signals` passes the sentiment
    leg without changing the convergence rule's structure. Mirrors the column
    contract of ``aggregate_sentiment``.
    """
    score = float(PARAMS.buy_sentiment_threshold)  # exactly at threshold => BUY
    df = pd.DataFrame({
        "score": score,
        "n_obs": 1,
        "score_stdev": 0.0,
        "score_min": score,
        "score_max": score,
    }, index=pd.Index(list(tickers), name="ticker"))
    return df


def _sentiment_at(as_of: pd.Timestamp, gate: str) -> pd.DataFrame:
    if gate == "off":
        return _synthetic_sentiment(_universe())
    # gate == 'on' -- use the real DB. Returns whatever coverage exists.
    from src.db import aggregate_sentiment
    return aggregate_sentiment(as_of=as_of.date())


def _signals_at(closes: pd.DataFrame, as_of: pd.Timestamp,
                gate: str, prior_history: pd.DataFrame) -> pd.DataFrame:
    """Refined signals frame for one rebalance date.

    `prior_history` is the prior weekly raw-signal frame (index=week,
    columns=ticker, values='BUY'/'HOLD'/'SELL') — supplied by the caller so
    we don't recompute the whole replay every week.
    """
    metrics = compute_sector_metrics(closes, as_of=as_of)
    if metrics.empty:
        return pd.DataFrame()
    sentiment = _sentiment_at(as_of, gate)
    sig = build_signals(metrics, sentiment)
    refined = refine_signals(sig, history=prior_history if not prior_history.empty else None,
                             macro_alignment=None)
    return refined


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def _execution_price(opens: pd.DataFrame, closes: pd.DataFrame,
                     signal_date: pd.Timestamp,
                     execution: str) -> tuple[pd.Timestamp | None, pd.Series | None]:
    """Resolve (fill_date, prices) for a trade decided on `signal_date`.

    'same_close' fills at signal_date's close. 'next_open' fills at the open of
    the next trading day; returns (None, None) when no next session exists
    (i.e. signal_date is the last bar in the panel — we just don't trade).
    """
    if execution == "same_close":
        if signal_date not in closes.index:
            return None, None
        return signal_date, closes.loc[signal_date]
    # next_open
    later = opens.index[opens.index > signal_date]
    if len(later) == 0:
        return None, None
    fill_date = later[0]
    return fill_date, opens.loc[fill_date]


def _equal_weight_targets(refined: pd.DataFrame, cash_buffer: float,
                          chase_weight_fraction: float | None = None,
                          promote_chase: bool = False
                          ) -> pd.Series:
    """Equal weight across NEW_BUY + HOLD_IF_LONG, plus an optional partial
    CHASE bucket. Thin wrapper around `target_weights()` — keeps the
    backtest and live model wired to the same sizing function. Passing
    `chase_weight_fraction=None` defers to `PARAMS.chase_weight_fraction`.
    `promote_chase=True` folds CHASE into the confirmed set at full weight
    (regime-aware bull overlay).
    """
    return target_weights(refined, cash_buffer=cash_buffer,
                          chase_weight_fraction=chase_weight_fraction,
                          promote_chase=promote_chase)


@contextmanager
def temporarily_override_params(**overrides):
    """Mutate `config.settings.PARAMS` fields in-place for one backtest run.

    `PARAMS` is a frozen dataclass imported by name across the codebase, so
    rebinding `config.settings.PARAMS` doesn't help — already-bound `from
    config.settings import PARAMS` references would keep the old object.
    We instead mutate the existing instance via `object.__setattr__` (which
    bypasses the frozen check) and restore originals in `finally`.

    Intended for the walk-forward sweep ONLY. Do not call from app code.
    """
    if not overrides:
        yield
        return
    originals = {k: getattr(PARAMS, k) for k in overrides}
    try:
        for k, v in overrides.items():
            object.__setattr__(PARAMS, k, v)
        yield
    finally:
        for k, v in originals.items():
            object.__setattr__(PARAMS, k, v)


def _pick_price(fill_px: pd.Series, closes_ffill: pd.DataFrame,
                fill_date: pd.Timestamp, ticker: str) -> float | None:
    """Resolve a usable trade price for `ticker` at `fill_date`. Falls back
    to the ffilled close when the raw open is missing (rare for liquid ETFs
    but happens on holiday weeks). Returns None if neither is usable.
    """
    px = fill_px.get(ticker)
    if px is None or pd.isna(px):
        px = closes_ffill.loc[fill_date].get(ticker)
    if px is None or pd.isna(px) or float(px) <= 0:
        return None
    return float(px)


def _mark_to_market(shares: dict[str, float], cash: float,
                    prices: pd.Series) -> float:
    eq = cash
    for tkr, qty in shares.items():
        if qty == 0:
            continue
        px = prices.get(tkr)
        if px is None or pd.isna(px):
            # Use last known close — caller should ffill to avoid this.
            continue
        eq += qty * float(px)
    return eq


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _annualised_stats(equity: pd.Series) -> dict:
    """CAGR, vol, Sharpe (rf=0), MDD from a daily equity series."""
    if len(equity) < 2:
        return {"cagr": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "total_return": 0.0, "n_days": int(len(equity))}
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if years > 0 else 0.0
    ann_vol = float(rets.std(ddof=0) * np.sqrt(252))
    sharpe = float((rets.mean() * 252) / (rets.std(ddof=0) * np.sqrt(252))) \
        if rets.std(ddof=0) > 0 else 0.0
    return {
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(equity),
        "total_return": total,
        "n_days": int(len(equity)),
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_backtest(cfg: BacktestConfig | None = None,
                 closes: pd.DataFrame | None = None,
                 opens: pd.DataFrame | None = None) -> BacktestResult:
    """Run the mechanical-core backtest and return a `BacktestResult`.

    Pass in `closes`/`opens` to use a synthetic panel (tests do this); leave
    them as None to load from `prices.db`.
    """
    cfg = cfg or BacktestConfig()
    if closes is None or opens is None:
        closes, opens = load_price_panel()
    # Apply per-run PARAMS overrides for the duration of this backtest. The
    # context manager restores the originals in `finally` even on exception.
    with temporarily_override_params(**(cfg.param_overrides or {})):
        return _run_backtest_impl(cfg, closes, opens)


def _run_backtest_impl(cfg: BacktestConfig,
                       closes: pd.DataFrame,
                       opens: pd.DataFrame) -> BacktestResult:

    universe = _universe()
    have = [t for t in universe if t in closes.columns]
    if BENCHMARK not in closes.columns:
        raise RuntimeError(f"benchmark {BENCHMARK} not in price panel")

    # Need at least sma_window+1 bars before the first rebalance; clip start.
    earliest_idx = closes.dropna(how="all").index[PARAMS.sma_window + PARAMS.momentum_window]
    start_ts = max(pd.Timestamp(cfg.start), earliest_idx) if cfg.start else earliest_idx
    end_ts = pd.Timestamp(cfg.end) if cfg.end else closes.dropna(how="all").index[-1]
    rebal_dates = weekly_rebalance_dates(closes, start_ts.date(), end_ts.date())
    if not rebal_dates:
        raise RuntimeError("no rebalance dates available in the requested window")

    # ---- state ----
    shares: dict[str, float] = {t: 0.0 for t in have}
    cash = cfg.initial_capital
    equity_records: list[tuple[pd.Timestamp, float]] = []
    weights_records: list[dict] = []
    states_records: list[dict] = []
    trades: list[dict] = []
    raw_history: pd.DataFrame = pd.DataFrame(columns=have)  # for refine_signals

    # Pre-clip price frames to [first relevant date, end]. Refine still uses
    # `as_of=` slicing so adding extra rows is harmless, but keep it tidy.
    panel_idx = closes.index[closes.index <= end_ts]
    closes = closes.loc[panel_idx]
    opens = opens.loc[panel_idx]
    # Forward-fill within the universe + benchmark so per-day mark-to-market
    # doesn't NaN-out on a single missing print. Original bars used for signal
    # calc are NOT ffilled — `compute_sector_metrics` reads raw closes.
    closes_ffill = closes.ffill()

    # Regime-aware bull overlay (optional). All inputs are rolling windows on
    # SPY that read only bars <= each date, so precomputing over the whole
    # panel introduces no look-ahead — we index the boolean as-of below.
    strong_bull: pd.Series | None = None
    if cfg.regime_aware:
        spy_s = closes[BENCHMARK].dropna()
        sma = spy_s.rolling(cfg.bull_sma_window, min_periods=cfg.bull_sma_window).mean()
        sma_rising = sma > sma.shift(21)            # SMA higher than ~1 month ago
        roll_high = spy_s.rolling(252, min_periods=1).max()
        near_high = spy_s >= roll_high * (1.0 - cfg.bull_proximity_pct)
        strong_bull = (spy_s > sma) & sma_rising & near_high
        strong_bull = strong_bull.fillna(False)

    def _bull_overlay_at(rb_date: pd.Timestamp) -> tuple[bool, float]:
        """(promote_chase, effective_cash_buffer) for a rebalance date.

        Defensive defaults unless regime_aware AND the as-of tape is a
        confirmed STRONG uptrend (see BacktestConfig.regime_aware)."""
        if strong_bull is None:
            return False, cfg.cash_buffer
        pos = strong_bull.index.searchsorted(rb_date, side="right")
        is_bull = bool(strong_bull.iloc[pos - 1]) if pos > 0 else False
        if is_bull:
            return True, cfg.bull_cash_buffer
        return False, cfg.cash_buffer

    last_eq_date: pd.Timestamp | None = None

    def _record_equity_through(target_date: pd.Timestamp):
        """Append daily mark-to-market equity from (last_eq_date, target_date]."""
        nonlocal last_eq_date
        if last_eq_date is None:
            return
        rng = closes.index[(closes.index > last_eq_date) & (closes.index <= target_date)]
        for d in rng:
            eq = _mark_to_market(shares, cash, closes_ffill.loc[d])
            equity_records.append((d, eq))
        last_eq_date = target_date

    for rb_date in rebal_dates:
        refined = _signals_at(closes, rb_date, cfg.sentiment_gate, raw_history)
        if refined.empty:
            continue
        promote_chase, eff_cash_buffer = _bull_overlay_at(rb_date)
        tgt = _equal_weight_targets(refined, eff_cash_buffer,
                                    chase_weight_fraction=cfg.chase_weight_fraction,
                                    promote_chase=promote_chase)
        # Restrict targets to tickers we actually have prices for.
        tgt = tgt.reindex([t for t in tgt.index if t in have]).dropna()

        fill_date, fill_px = _execution_price(opens, closes, rb_date, cfg.execution)
        if fill_date is None or fill_px is None:
            # Last bar — still record the (unrebalanced) equity and stop.
            if last_eq_date is None:
                last_eq_date = rb_date
                equity_records.append((rb_date, _mark_to_market(shares, cash, closes_ffill.loc[rb_date])))
            else:
                _record_equity_through(rb_date)
            break

        # Walk daily equity up to (but not through) the fill date, marked at
        # last close — captures the holding period between rebalances.
        if last_eq_date is None:
            last_eq_date = rb_date
            equity_records.append((rb_date, _mark_to_market(shares, cash, closes_ffill.loc[rb_date])))
        else:
            _record_equity_through(rb_date)

        # Compute pre-trade NAV at the FILL bar's prices to size targets.
        nav = _mark_to_market(shares, cash, fill_px.fillna(closes_ffill.loc[fill_date]))

        cost_per_side = (cfg.cost_bps + cfg.slippage_bps) / 10_000.0
        target_set = set(tgt.index)
        held_set = {t for t, q in shares.items() if q > 1e-9}

        def _trade(t: str, delta_dollars: float, px: float, state: str):
            nonlocal cash
            delta_shares = delta_dollars / px
            cost = abs(delta_dollars) * cost_per_side
            shares[t] = shares.get(t, 0.0) + delta_shares
            cash -= delta_dollars
            cash -= cost
            trades.append({
                "rebalance_date": rb_date,
                "fill_date": fill_date,
                "ticker": t,
                "side": "BUY" if delta_shares > 0 else "SELL",
                "shares": float(delta_shares),
                "price": px,
                "notional": float(delta_dollars),
                "cost": float(cost),
                "state": state,
            })

        if cfg.trade_policy == "event_driven":
            # 1) Exit any held name no longer in target_set (state left BUY class).
            for t in sorted(held_set - target_set):
                px = _pick_price(fill_px, closes_ffill, fill_date, t)
                if px is None:
                    continue
                cur_dollars = shares[t] * px
                if abs(cur_dollars) < 1.0:
                    continue
                _trade(t, -cur_dollars, px, str(refined["state"].get(t, "")))
            # 2) Enter any name in target_set that we don't already hold.
            new_entries = sorted(target_set - held_set)
            for t in new_entries:
                px = _pick_price(fill_px, closes_ffill, fill_date, t)
                if px is None:
                    continue
                want = nav * float(tgt[t])
                if want < 1.0:
                    continue
                _trade(t, want, px, str(refined["state"].get(t, "")))
            # 3) Drift between held & still-in-target names is intentional —
            #    matches the live "no rebalance, just transitions" behaviour.
        else:  # rebalance_to_target
            touched = held_set | target_set
            target_dollars = {t: nav * float(w) for t, w in tgt.items()}
            for t in sorted(touched):
                px = _pick_price(fill_px, closes_ffill, fill_date, t)
                if px is None:
                    continue
                tgt_dollars = target_dollars.get(t, 0.0)
                cur_dollars = shares.get(t, 0.0) * px
                delta_dollars = tgt_dollars - cur_dollars
                if abs(delta_dollars) < 1.0:
                    continue
                _trade(t, delta_dollars, px, str(refined["state"].get(t, "")))

        # Drop fully-closed tickers from shares to keep the dict tidy.
        shares = {t: q for t, q in shares.items() if abs(q) > 1e-9}

        # Record equity at fill_date *after* the trade.
        equity_records.append((fill_date, _mark_to_market(shares, cash, closes_ffill.loc[fill_date])))
        last_eq_date = fill_date

        # Persist diagnostics for this rebalance.
        weights_records.append({
            "date": rb_date,
            "fill_date": fill_date,
            **{t: float(tgt.get(t, 0.0)) for t in have},
            "cash_buffer": float(1.0 - tgt.sum()) if not tgt.empty else 1.0,
        })
        for t, st in refined["state"].items():
            states_records.append({
                "date": rb_date,
                "ticker": t,
                "state": st,
                "signal": refined.loc[t].get("signal", ""),
                "above_sma": bool(refined.loc[t].get("above_sma", False)),
                "rs_rank": int(refined.loc[t].get("rs_rank", 0)),
                "extension_pct": float(refined.loc[t].get("extension_pct", 0.0) or 0.0),
                "relative_strength_3m": float(refined.loc[t].get("relative_strength_3m", 0.0) or 0.0),
            })

        # Append this snapshot's raw signal labels to the history window for
        # next iteration's staleness / REDUCE detection.
        new_row = refined["signal"].reindex(have)
        raw_history = pd.concat([raw_history, pd.DataFrame([new_row], index=[rb_date])])
        if len(raw_history) > PARAMS.history_weeks:
            raw_history = raw_history.iloc[-PARAMS.history_weeks:]

    # Final mark-to-market walk through end_ts.
    _record_equity_through(end_ts)

    # ---- assemble outputs ----
    if not equity_records:
        raise RuntimeError("backtest produced no equity records")
    equity = pd.Series(dict(equity_records)).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    equity.name = "strategy_equity"

    # SPY buy-and-hold over the same window, scaled to initial_capital.
    spy = closes_ffill[BENCHMARK].loc[equity.index[0]:equity.index[-1]]
    bench = (spy / spy.iloc[0]) * cfg.initial_capital
    bench.name = "spy_equity"

    trades_df = pd.DataFrame(trades)
    weights_df = pd.DataFrame(weights_records)
    states_df = pd.DataFrame(states_records)

    # Stats.
    s_strat = _annualised_stats(equity)
    s_bench = _annualised_stats(bench)
    # Turnover: sum |notional| / average NAV / years.
    if not trades_df.empty:
        gross_notional = trades_df["notional"].abs().sum()
        avg_nav = float(equity.mean())
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        ann_turnover = float(gross_notional / avg_nav / years) if years > 0 and avg_nav > 0 else 0.0
        # Hit rate of CLOSED positions: pair BUYs and SELLs per ticker FIFO.
        hit_rate = _closed_position_hit_rate(trades_df)
        total_costs = float(trades_df["cost"].sum())
    else:
        ann_turnover = 0.0
        hit_rate = 0.0
        total_costs = 0.0

    stats = {
        "window_start": str(equity.index[0].date()),
        "window_end": str(equity.index[-1].date()),
        "initial_capital": cfg.initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "strategy": s_strat,
        "spy": s_bench,
        "excess_cagr": s_strat["cagr"] - s_bench["cagr"],
        "excess_total_return": s_strat["total_return"] - s_bench["total_return"],
        "annualised_turnover": ann_turnover,
        "n_trades": int(0 if trades_df.empty else len(trades_df)),
        "total_costs": total_costs,
        "closed_position_hit_rate": hit_rate,
        "config": {
            "execution": cfg.execution,
            "cost_bps": cfg.cost_bps,
            "slippage_bps": cfg.slippage_bps,
            "cash_buffer": cfg.cash_buffer,
            "sentiment_gate": cfg.sentiment_gate,
        },
    }

    return BacktestResult(
        config=cfg, equity=equity, benchmark_equity=bench,
        trades=trades_df, stats=stats,
        weights_history=weights_df, states_history=states_df,
    )


def _closed_position_hit_rate(trades: pd.DataFrame) -> float:
    """FIFO-match BUYs and SELLs per ticker and report (closed PnL>0)/(closed)."""
    if trades.empty:
        return 0.0
    wins = total = 0
    for tkr, sub in trades.groupby("ticker"):
        sub = sub.sort_values("fill_date")
        lots: list[tuple[float, float]] = []  # (shares, price)
        for _, r in sub.iterrows():
            shares = r["shares"]
            price = r["price"]
            if shares > 0:
                lots.append((shares, price))
            else:
                qty_to_close = -shares
                while qty_to_close > 1e-9 and lots:
                    lot_shares, lot_price = lots[0]
                    take = min(lot_shares, qty_to_close)
                    pnl = (price - lot_price) * take
                    total += 1
                    if pnl > 0:
                        wins += 1
                    qty_to_close -= take
                    if lot_shares - take <= 1e-9:
                        lots.pop(0)
                    else:
                        lots[0] = (lot_shares - take, lot_price)
    if total == 0:
        return 0.0
    return wins / total


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_equity_csv(result: BacktestResult,
                    name: str = "backtest_equity") -> Path:
    """Write the strategy + SPY equity curves to ``data/<name>.csv``."""
    out = pd.DataFrame({
        "date": result.equity.index,
        "strategy": result.equity.values,
        "spy": result.benchmark_equity.reindex(result.equity.index).values,
    })
    path = Path(DATA_DIR) / f"{name}.csv"
    out.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Sentiment ablation over the (tiny) real-data window
# ---------------------------------------------------------------------------

def real_sentiment_ablation(
    closes: pd.DataFrame | None = None,
    weeks: int = 14,
    end: date | None = None,
) -> dict:
    """Forward-return comparison of NEW_BUY signals with vs without the
    sentiment gate over the trailing `weeks` weekly snapshots.

    HONEST CAVEAT: the project has only ~2 weeks of meaningful newsletter
    sentiment coverage and ~3 months of sparse coverage. Sample sizes here
    are TINY (typically < 30 per arm). This is a directional sanity check,
    not statistical evidence. Treat outputs accordingly.
    """
    if closes is None:
        closes, _ = load_price_panel()
    end = end or closes.dropna(how="all").index[-1].date()
    universe = _universe()

    rebal_dates = weekly_rebalance_dates(closes, None, end)[-weeks:]
    if not rebal_dates:
        return {"n_rebalances": 0, "off": {}, "on": {}, "caveat": "no rebalance dates"}

    def _arm(gate: str) -> tuple[int, float, float]:
        n = wins = 0
        excesses: list[float] = []
        prior_hist = pd.DataFrame(columns=universe)
        for rb in rebal_dates:
            refined = _signals_at(closes, rb, gate, prior_hist)
            if refined.empty:
                continue
            new_buys = [t for t in refined.index if refined.loc[t, "state"] == "NEW_BUY"]
            # Forward 1-week return vs SPY.
            entry_idx = closes.index.searchsorted(rb, side="left")
            exit_idx = closes.index.searchsorted(rb + pd.Timedelta(days=7), side="left")
            if entry_idx >= len(closes) or exit_idx >= len(closes) or exit_idx <= entry_idx:
                # Still update prior_hist before continuing.
                new_row = refined["signal"].reindex(universe)
                prior_hist = pd.concat([prior_hist, pd.DataFrame([new_row], index=[rb])]).iloc[-PARAMS.history_weeks:]
                continue
            spy_e = float(closes[BENCHMARK].iloc[entry_idx])
            spy_x = float(closes[BENCHMARK].iloc[exit_idx])
            spy_ret = spy_x / spy_e - 1.0 if spy_e else 0.0
            for t in new_buys:
                p_e = closes[t].iloc[entry_idx] if t in closes.columns else None
                p_x = closes[t].iloc[exit_idx] if t in closes.columns else None
                if p_e is None or p_x is None or pd.isna(p_e) or pd.isna(p_x) or p_e == 0:
                    continue
                ex = (float(p_x) / float(p_e) - 1.0) - spy_ret
                excesses.append(ex)
                n += 1
                if ex > 0:
                    wins += 1
            new_row = refined["signal"].reindex(universe)
            prior_hist = pd.concat([prior_hist, pd.DataFrame([new_row], index=[rb])]).iloc[-PARAMS.history_weeks:]
        if n == 0:
            return 0, 0.0, 0.0
        return n, float(np.mean(excesses)), wins / n

    n_off, m_off, h_off = _arm("off")
    n_on, m_on, h_on = _arm("on")
    return {
        "n_rebalances": len(rebal_dates),
        "window_weeks": weeks,
        "off": {"n_signals": n_off, "mean_excess_1w": m_off, "hit_rate": h_off},
        "on":  {"n_signals": n_on,  "mean_excess_1w": m_on,  "hit_rate": h_on},
        "caveat": ("Sample sizes are TINY (n typically < 30 per arm). Treat as a "
                   "directional sanity check, NOT statistical evidence. The "
                   "project has only ~2 weeks of meaningful newsletter coverage."),
    }
