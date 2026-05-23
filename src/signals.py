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


def refine_signals(signals: pd.DataFrame,
                   history: pd.DataFrame | None = None) -> pd.DataFrame:
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
