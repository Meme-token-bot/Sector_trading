"""Module C: convergence decision matrix."""
from __future__ import annotations

import pandas as pd

from config.settings import PARAMS


def build_signals(metrics: pd.DataFrame,
                  sentiment: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    if sentiment.empty:
        df["sentiment_score"] = 0.0
        df["n_obs"] = 0
    else:
        df = df.join(sentiment[["score", "n_obs"]], how="left")
        df = df.rename(columns={"score": "sentiment_score"})
        df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
        df["n_obs"] = df["n_obs"].fillna(0).astype(int)

    # Sentiment-quality diagnostics (appended; never reordered). When the
    # sentiment frame doesn't carry these columns (older callers / empty
    # frame) we default to safe values: stdev 0, extremes NaN.
    import numpy as np
    for col, default in (("score_stdev", 0.0),
                         ("score_min",   np.nan),
                         ("score_max",   np.nan)):
        if not sentiment.empty and col in sentiment.columns:
            df[col] = sentiment[col].reindex(df.index)
            if col == "score_stdev":
                df[col] = df[col].fillna(0.0).astype(float)
        else:
            df[col] = default

    max_rank = df["rs_rank"].max()
    weak_threshold = max(max_rank - PARAMS.weak_rs_rank_cutoff + 1, 1)

    signals: list[str] = []
    reasons: list[str] = []
    for tkr, row in df.iterrows():
        sell_reasons = []
        if not row["above_sma"]:
            sell_reasons.append("price<SMA200")
        if row["rs_rank"] >= weak_threshold:
            sell_reasons.append(f"RS rank {int(row['rs_rank'])}/{int(max_rank)} (bottom {PARAMS.weak_rs_rank_cutoff})")
        if row["sentiment_score"] <= PARAMS.sell_sentiment_threshold:
            sell_reasons.append(f"sentiment {row['sentiment_score']:+.1f}<={PARAMS.sell_sentiment_threshold:+.0f}")

        if sell_reasons:
            signals.append("SELL")
            reasons.append("; ".join(sell_reasons))
            continue

        buy_conditions = [
            row["above_sma"],
            row["relative_strength_3m"] > 0,
            row["sentiment_score"] >= PARAMS.buy_sentiment_threshold,
        ]
        if all(buy_conditions):
            signals.append("BUY")
            reasons.append(
                f"above SMA200; RS {row['relative_strength_3m']*100:+.1f}%; "
                f"sentiment {row['sentiment_score']:+.1f}"
            )
        else:
            signals.append("HOLD")
            missing = []
            if not row["above_sma"]:
                missing.append("not above SMA200")
            if row["relative_strength_3m"] <= 0:
                missing.append(f"RS {row['relative_strength_3m']*100:+.1f}%<=0")
            if row["sentiment_score"] < PARAMS.buy_sentiment_threshold:
                missing.append(
                    f"sentiment {row['sentiment_score']:+.1f}<{PARAMS.buy_sentiment_threshold:+.0f}"
                    + (" (no coverage)" if row["n_obs"] == 0 else "")
                )
            reasons.append("; ".join(missing))

    df["signal"] = signals
    df["reasons"] = reasons
    return df


def _macro_net(macro_alignment: pd.DataFrame | None, tkr: str) -> int | None:
    """Net macro reading for a ticker = tailwinds - headwinds.

    Returns None when no macro frame is supplied or the sector has zero
    applicable readings (tailwinds + headwinds == 0) — callers must treat
    None as "no macro opinion" and leave conviction / state untouched.
    """
    if macro_alignment is None or macro_alignment.empty:
        return None
    if tkr not in macro_alignment.index:
        return None
    tw = int(macro_alignment.loc[tkr, "tailwinds"] or 0)
    hw = int(macro_alignment.loc[tkr, "headwinds"] or 0)
    if tw + hw == 0:
        return None
    return tw - hw


def refine_signals(signals: pd.DataFrame,
                   history: pd.DataFrame | None = None,
                   macro_alignment: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add a state-aware `state` column on top of the raw `signal`.

    States:
      NEW_BUY       — signal=BUY, not extended, fresh (< stale_buy_weeks consecutive BUYs)
      HOLD_IF_LONG  — signal=BUY but stale (BUY for >= stale_buy_weeks). Hold if owned, don't add.
      CHASE         — signal=BUY but extension_pct > cutoff. Don't enter; sector is parabolic.
      REDUCE        — signal=HOLD now, but was BUY in the recent history window. Trim if owned.
      HOLD          — signal=HOLD with no recent BUY history. Wait-and-see.
      SELL          — signal=SELL.

    Pure function. `history` may be None / empty — in that case we can't
    detect staleness or recent-BUY-now-HOLD, so HOLD stays HOLD and BUY
    becomes NEW_BUY (gated only by extension).
    """
    out = signals.copy()
    cutoff = PARAMS.extension_pct_cutoff
    stale_n = PARAMS.stale_buy_weeks

    if history is not None and not history.empty:
        from src.signal_history import consecutive_buy_weeks
        weeks = consecutive_buy_weeks(history).reindex(out.index).fillna(0).astype(int)
        # Was the sector BUY at any point in the recent history window?
        ever_buy = (history == "BUY").any(axis=0).reindex(out.index).fillna(False)
    else:
        weeks = pd.Series(0, index=out.index)
        ever_buy = pd.Series(False, index=out.index)

    out["consecutive_buy_weeks"] = weeks

    states: list[str] = []
    state_reasons: list[str] = []

    for tkr, row in out.iterrows():
        sig = row["signal"]
        ext = float(row.get("extension_pct", 0.0) or 0.0)
        n_buy = int(weeks.get(tkr, 0))

        if sig == "SELL":
            states.append("SELL")
            state_reasons.append(row["reasons"])
            continue

        if sig == "BUY":
            if ext > cutoff:
                states.append("CHASE")
                state_reasons.append(
                    f"price {ext*100:+.1f}% above SMA200 (cutoff {cutoff*100:.0f}%) "
                    f"— too extended for fresh entry"
                )
            elif n_buy >= stale_n:
                states.append("HOLD_IF_LONG")
                state_reasons.append(
                    f"BUY for {n_buy} consecutive weeks (cutoff {stale_n}) "
                    f"— hold if you own it, do not add fresh"
                )
            else:
                states.append("NEW_BUY")
                state_reasons.append(
                    f"fresh BUY (week {n_buy + 1}); ext {ext*100:+.1f}% "
                    f"vs SMA200 (cutoff {cutoff*100:.0f}%)"
                )
            continue

        # signal == HOLD
        if bool(ever_buy.get(tkr, False)):
            states.append("REDUCE")
            state_reasons.append(
                "was BUY in the last "
                f"{len(history) if history is not None else 0} weeks but no longer "
                f"qualifies — trim if owned"
            )
        else:
            states.append("HOLD")
            state_reasons.append(row["reasons"])

    out["state"] = states
    out["state_reason"] = state_reasons

    # ----- Macro veto / override pass -----------------------------------
    # Macro can CUT risk freely but cannot ADD unconfirmed price risk:
    #   NEW_BUY      + strong headwind -> HOLD   (veto: drops from target_weights)
    #   HOLD_IF_LONG + strong headwind -> REDUCE (trim a stale BUY)
    #   HOLD         + strong tailwind -> WATCH  (only if above_sma; no capital)
    # SELL/CHASE and any sector with no macro opinion are left untouched.
    # WATCH is deliberately excluded from target_weights — price hasn't
    # confirmed, so it's a visibility flag, not a position.
    thr = PARAMS.macro_strong_count
    new_states = list(out["state"])
    new_reasons = list(out["state_reason"])
    for i, (tkr, row) in enumerate(out.iterrows()):
        net = _macro_net(macro_alignment, tkr)
        if net is None:
            continue
        tw = int(macro_alignment.loc[tkr, "tailwinds"] or 0)
        hw = int(macro_alignment.loc[tkr, "headwinds"] or 0)
        st = new_states[i]
        if net <= -thr and st == "NEW_BUY":
            new_states[i] = "HOLD"
            new_reasons[i] = (f"macro veto: {hw} headwinds vs {tw} tailwinds "
                              f"— defer fresh entry")
        elif net <= -thr and st == "HOLD_IF_LONG":
            new_states[i] = "REDUCE"
            new_reasons[i] = (f"macro headwind ({hw} vs {tw}) on a stale BUY "
                              f"— trim if owned")
        elif net >= thr and st == "HOLD" and bool(row.get("above_sma", False)):
            new_states[i] = "WATCH"
            new_reasons[i] = (f"sentiment+macro support ({tw} tailwinds vs {hw} "
                              f"headwinds), price not yet confirmed — watch for RS turn")
    out["state"] = new_states
    out["state_reason"] = new_reasons

    # ----- Conviction score (0..5) -----
    # Each non-macro component contributes at most +1. The macro component is
    # graded and SYMMETRIC: a clear net tailwind adds +1, a clear net headwind
    # subtracts 1 (a sector fighting the macro tape loses conviction). The
    # final score is clamped to [0, 5] so the 5-dot display still renders;
    # the clamp can pull a sector below its trend/sentiment baseline but never
    # negative. When no macro frame is supplied the macro component is 0.
    convictions: list[int] = []
    for tkr, row in out.iterrows():
        score = 0
        rs3 = float(row.get("relative_strength_3m", 0.0) or 0.0)
        if rs3 > 0:
            score += 1
        if rs3 > PARAMS.strong_rs_margin:
            score += 1
        sent = float(row.get("sentiment_score", 0.0) or 0.0)
        if sent >= PARAMS.buy_sentiment_threshold + 1:
            score += 1
        n_buy = int(row.get("consecutive_buy_weeks", 0) or 0)
        if n_buy >= 2:
            score += 1
        # Conviction reacts to ANY clear macro lean (net >= +1 / <= -1) — a
        # finer bar than the state override below, which needs a STRONG lean
        # (macro_strong_count). This keeps conviction responsive while reserving
        # the disruptive veto/override for unambiguous macro signals.
        net = _macro_net(macro_alignment, tkr)
        if net is not None:
            if net >= 1:
                score += 1
            elif net <= -1:
                score -= 1
        convictions.append(max(0, min(5, score)))

    out["conviction"] = convictions
    return out


def target_weights(signals: pd.DataFrame, cash_buffer: float = 0.05) -> pd.Series:
    """Equal-weight target across BUY-class signals.

    If a `state` column is present (from refine_signals), uses the
    state-aware definition: NEW_BUY + HOLD_IF_LONG (i.e. all positions
    the model wants exposure to). CHASE is excluded — the model says
    don't enter from cash.

    Otherwise falls back to the raw `signal` column.

    Supplementary sectors (``config.settings.SUPPLEMENTARY_SECTORS``, e.g.
    UFO/Space) are excluded — they're tactical overlays the user sizes
    separately and should not dilute the equal-weight allocation across
    the 11 SPDR sectors.
    """
    from config.settings import SUPPLEMENTARY_SECTORS

    if "state" in signals.columns:
        active = signals.index[signals["state"].isin(["NEW_BUY", "HOLD_IF_LONG"])]
    else:
        active = signals.index[signals["signal"] == "BUY"]
    active = active.difference(SUPPLEMENTARY_SECTORS)
    if len(active) == 0:
        return pd.Series(dtype=float, name="target_weight")
    per_name = (1.0 - cash_buffer) / len(active)
    return pd.Series(per_name, index=active, name="target_weight")
