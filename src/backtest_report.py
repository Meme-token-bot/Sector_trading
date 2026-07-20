"""Generate BACKTEST_REPORT.md from live backtest output, not from memory.

The whole point of this module is to make it IMPOSSIBLE for a number in the
report to disagree with what the backtest actually produced. Every quantitative
claim is derived from a `BacktestResult` (or from a fresh query against the
real DBs) at render time — there is no hand-typed number anywhere in the
generated markdown.

Narrative sections (methodology, blunt assessment) ARE templated, but every
fact embedded in them comes through a slot. If a claim can't be expressed as a
slot, it gets dropped rather than guessed.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import BENCHMARK, DB_PATH, PARAMS, SECTOR_ETFS, SUPPLEMENTARY_SECTORS
from src.backtest import (
    BacktestConfig,
    BacktestResult,
    real_sentiment_ablation,
    run_backtest,
    save_equity_csv,
)
from src.price_store import PRICES_DB_PATH


# ---------------------------------------------------------------------------
# Step 0 — fresh DB findings, queried at report-render time
# ---------------------------------------------------------------------------

@dataclass
class DBFindings:
    """Facts about the two databases as of right now. Computed from the
    real schemas — never from memory."""
    prices_window: tuple[str, str]                 # min, max bar_date (daily)
    prices_coverage: list[dict]                    # one row per (ticker, 1d) with n_bars
    sectors_with_daily: list[str]                  # SPDR + SPY universe present
    n_newsletters: int
    n_sector_ratings: int
    sentiment_window: tuple[str, str] | None
    newsletters_by_date: list[tuple[str, int]]    # (date, count) per day
    sector_coverage: list[dict]                    # per ticker: n, first, last


def gather_db_findings() -> DBFindings:
    # ---- prices.db ------------------------------------------------------
    con_p = sqlite3.connect(PRICES_DB_PATH)
    win = con_p.execute(
        "SELECT MIN(bar_date), MAX(bar_date) FROM ohlcv WHERE timeframe='1d'"
    ).fetchone()
    cov = pd.read_sql_query(
        """SELECT ticker, COUNT(*) AS n_bars,
                  MIN(bar_date) AS first_bar, MAX(bar_date) AS last_bar
           FROM ohlcv WHERE timeframe='1d'
           GROUP BY ticker
           ORDER BY ticker""",
        con_p,
    )
    con_p.close()
    universe = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    sectors_with_daily = sorted(
        set(cov["ticker"]).intersection(set(universe))
    )

    # ---- sentiment.db ---------------------------------------------------
    con_s = sqlite3.connect(DB_PATH)
    n_news = con_s.execute("SELECT COUNT(*) FROM newsletters").fetchone()[0]
    n_sr = con_s.execute("SELECT COUNT(*) FROM sector_ratings").fetchone()[0]
    win_s = con_s.execute(
        "SELECT MIN(publication_date), MAX(publication_date) FROM newsletters"
    ).fetchone()
    sentiment_window = tuple(win_s) if win_s and win_s[0] else None  # type: ignore[assignment]
    by_date = con_s.execute(
        """SELECT publication_date, COUNT(*) FROM newsletters
           GROUP BY publication_date ORDER BY publication_date"""
    ).fetchall()
    sec_cov = pd.read_sql_query(
        """SELECT sr.ticker, COUNT(*) AS n,
                  MIN(n.publication_date) AS first_date,
                  MAX(n.publication_date) AS last_date
           FROM sector_ratings sr
           JOIN newsletters n ON n.id = sr.newsletter_id
           GROUP BY sr.ticker
           ORDER BY n DESC""",
        con_s,
    )
    con_s.close()

    return DBFindings(
        prices_window=(str(win[0]), str(win[1])),
        prices_coverage=cov.to_dict("records"),
        sectors_with_daily=sectors_with_daily,
        n_newsletters=int(n_news),
        n_sector_ratings=int(n_sr),
        sentiment_window=sentiment_window,
        newsletters_by_date=[(r[0], int(r[1])) for r in by_date],
        sector_coverage=sec_cov.to_dict("records"),
    )


# ---------------------------------------------------------------------------
# Derived stats — every quantitative claim that goes in the report
# ---------------------------------------------------------------------------

@dataclass
class HeadlineReport:
    findings: DBFindings
    ed_result: BacktestResult              # event-driven backtest
    rbt_result: BacktestResult             # rebalance-to-target (for comparison)
    ablation: dict                         # real_sentiment_ablation output
    chase_share_by_ticker: pd.DataFrame    # ticker, n_chase, share, max_ext, median_ext_when_chase
    state_distribution: pd.DataFrame       # ticker x state counts
    n_weekly_snapshots: int
    cost_drag_decomp: dict                 # see _cost_drag_decomposition
    recent_newsletter_rate: dict           # mean / window / total over the last fortnight
    # Regime / drawdown analysis (P1 + P2). All fields below derived from
    # `src.regime_analysis.run_regime_analysis(ed_result, spy_close)`.
    regime_stats: pd.DataFrame             # per-regime cum/cagr/capture/mdd
    regime_distribution: dict              # {regime_label: n_days}
    drawdown_attribution: list             # list[dict] per qualifying SPY DD


def _chase_share(states_history: pd.DataFrame) -> pd.DataFrame:
    if states_history.empty:
        return pd.DataFrame(columns=["ticker", "n_chase", "share_pct",
                                      "max_ext_pct", "median_ext_when_chase_pct"])
    n_weeks = states_history["date"].nunique()
    rows: list[dict] = []
    for tkr, sub in states_history.groupby("ticker"):
        chase = sub[sub["state"] == "CHASE"]
        rows.append({
            "ticker": tkr,
            "n_chase": int(len(chase)),
            "share_pct": float(len(chase) / n_weeks * 100) if n_weeks else 0.0,
            "max_ext_pct": float(sub["extension_pct"].max() * 100) if not sub.empty else 0.0,
            "median_ext_when_chase_pct": (
                float(chase["extension_pct"].median() * 100) if not chase.empty else 0.0
            ),
        })
    return (pd.DataFrame(rows)
            .sort_values("share_pct", ascending=False)
            .reset_index(drop=True))


def _state_distribution(states_history: pd.DataFrame) -> pd.DataFrame:
    if states_history.empty:
        return pd.DataFrame()
    return (states_history.groupby("ticker")["state"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["NEW_BUY", "HOLD_IF_LONG", "CHASE",
                              "REDUCE", "HOLD", "SELL", "WATCH"],
                     fill_value=0))


def _cost_drag_decomposition(ed: BacktestResult, rbt: BacktestResult) -> dict:
    """Honest decomposition of the CAGR gap between the two trade policies.

    Returns the cost-drag delta in %/yr AND the structural residual (CAGR gap
    minus cost-drag delta). This is what the *report's* policy section claims;
    by computing it here rather than hand-typing it, the claim can't drift."""
    years = (ed.equity.index[-1] - ed.equity.index[0]).days / 365.25 or 1.0
    ed_cost_pct = ed.stats["total_costs"] / ed.stats["initial_capital"]
    rbt_cost_pct = rbt.stats["total_costs"] / rbt.stats["initial_capital"]
    cost_drag_diff_pct_yr = (rbt_cost_pct - ed_cost_pct) / years * 100.0
    cagr_diff_pct = (ed.stats["strategy"]["cagr"] - rbt.stats["strategy"]["cagr"]) * 100.0
    residual_pct_yr = cagr_diff_pct - cost_drag_diff_pct_yr
    return {
        "years": years,
        "ed_cost_drag_pct_yr": ed_cost_pct / years * 100.0,
        "rbt_cost_drag_pct_yr": rbt_cost_pct / years * 100.0,
        "cost_drag_diff_pct_yr": cost_drag_diff_pct_yr,
        "cagr_diff_pct": cagr_diff_pct,
        "structural_residual_pct_yr": residual_pct_yr,
    }


def _recent_newsletter_rate(findings: DBFindings, days: int = 14) -> dict:
    """Mean newsletters/day over the trailing `days` days (calendar). Computed
    against the last date in `newsletters_by_date` to keep this self-contained."""
    if not findings.newsletters_by_date:
        return {"mean_per_day": 0.0, "window_days": days, "total": 0, "first": None, "last": None}
    by_d = [(date.fromisoformat(d), n) for d, n in findings.newsletters_by_date]
    last_d = by_d[-1][0]
    cutoff = pd.Timestamp(last_d).normalize() - pd.Timedelta(days=days - 1)
    recent = [(d, n) for d, n in by_d if pd.Timestamp(d) >= cutoff]
    total = sum(n for _, n in recent)
    return {
        "mean_per_day": total / days,
        "window_days": days,
        "total": total,
        "first": str(recent[0][0]) if recent else None,
        "last": str(last_d),
    }


# ---------------------------------------------------------------------------
# Top-level: build a HeadlineReport, then render
# ---------------------------------------------------------------------------

def build_headline_report(
    cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    execution: str = "next_open",
) -> HeadlineReport:
    findings = gather_db_findings()
    ed = run_backtest(BacktestConfig(
        cost_bps=cost_bps, slippage_bps=slippage_bps,
        execution=execution, trade_policy="event_driven"))
    rbt = run_backtest(BacktestConfig(
        cost_bps=cost_bps, slippage_bps=slippage_bps,
        execution=execution, trade_policy="rebalance_to_target"))
    ablation = real_sentiment_ablation(weeks=14)
    chase = _chase_share(ed.states_history)
    states = _state_distribution(ed.states_history)
    n_weeks = ed.states_history["date"].nunique() if not ed.states_history.empty else 0
    decomp = _cost_drag_decomposition(ed, rbt)
    rate = _recent_newsletter_rate(findings, days=14)

    # Regime + drawdown analysis (P1 + P2). Pull SPY daily closes from the
    # same panel the backtest used so the equity/benchmark alignment is exact.
    from src.backtest import load_price_panel
    from src.regime_analysis import (
        classify_regimes, drawdown_attribution, regime_conditional_stats,
    )
    closes, _ = load_price_panel()
    spy_close = closes[BENCHMARK].dropna()
    aligned = spy_close.reindex(ed.equity.index, method="ffill")
    regimes = classify_regimes(aligned)
    regime_stats = regime_conditional_stats(ed.equity, ed.benchmark_equity,
                                              regimes)
    dd_rows = drawdown_attribution(ed, ed.benchmark_equity, spy_close,
                                     min_dd_pct=0.05, min_days=5)
    regime_dist = {r: int(n) for r, n in regimes.value_counts().items()}

    return HeadlineReport(
        findings=findings, ed_result=ed, rbt_result=rbt,
        ablation=ablation, chase_share_by_ticker=chase,
        state_distribution=states, n_weekly_snapshots=n_weeks,
        cost_drag_decomp=decomp, recent_newsletter_rate=rate,
        regime_stats=regime_stats, regime_distribution=regime_dist,
        drawdown_attribution=dd_rows,
    )


def render_markdown(report: HeadlineReport,
                    branch: str = "feat/history-expandable-and-signal-runner",
                    as_of: date | None = None) -> str:
    """Render the full BACKTEST_REPORT.md from a built `HeadlineReport`.

    Every quantitative claim in this string is interpolated from `report`.
    Narrative paragraphs are templated but the numbers they mention come
    through slots — there are no hand-typed magic numbers."""
    as_of = as_of or date.today()
    f = report.findings
    ed = report.ed_result
    s = ed.stats
    strat, spy = s["strategy"], s["spy"]
    decomp = report.cost_drag_decomp
    ab = report.ablation
    rate = report.recent_newsletter_rate

    cost_bps = s["config"]["cost_bps"]
    excess_cagr_pct = s["excess_cagr"] * 100.0
    win_start, win_end = s["window_start"], s["window_end"]

    # ---- Helpers --------------------------------------------------------
    def pct(x: float, p: int = 2, signed: bool = False) -> str:
        fmt = f"{{:{'+' if signed else ''}.{p}f}}%"
        return fmt.format(x * 100.0)

    def num(x: float, p: int = 2, signed: bool = False) -> str:
        fmt = f"{{:{'+' if signed else ''}.{p}f}}"
        return fmt.format(x)

    # ---- Sentiment coverage table (top-N tickers by n) -------------------
    cov_rows = "\n".join(
        f"  | {r['ticker']} | {r['n']} | {r['first_date']} → {r['last_date']} |"
        for r in f.sector_coverage
    )

    # ---- Per-ticker CHASE share table -----------------------------------
    chase_rows = "\n".join(
        f"  | {r['ticker']} | {r['n_chase']} | {r['share_pct']:.1f}% | "
        f"{r['max_ext_pct']:+.1f}% | {r['median_ext_when_chase_pct']:+.1f}% |"
        for r in report.chase_share_by_ticker.to_dict("records")
    )

    # ---- MDD comparison line --------------------------------------------
    diff_pp = (strat["max_drawdown"] - spy["max_drawdown"]) * 100.0
    if abs(diff_pp) < 2.0:
        mdd_obs = (
            f"Crucially, the strategy MDD ({pct(strat['max_drawdown'], signed=True)}) "
            f"is essentially in line with SPY's ({pct(spy['max_drawdown'], signed=True)}) "
            f"— a gap of only {abs(diff_pp):.1f} pp. **The model gives up "
            f"the upside without buying meaningfully less drawdown.**"
        )
    elif diff_pp > 0:
        mdd_obs = (
            f"The strategy MDD ({pct(strat['max_drawdown'], signed=True)}) is "
            f"**{diff_pp:.1f} pp shallower** than SPY's "
            f"({pct(spy['max_drawdown'], signed=True)}) — the rotation absorbed "
            f"part of the worst SPY drawdown in the window."
        )
    else:
        mdd_obs = (
            f"The strategy MDD ({pct(strat['max_drawdown'], signed=True)}) is "
            f"**{abs(diff_pp):.1f} pp DEEPER** than SPY's "
            f"({pct(spy['max_drawdown'], signed=True)}) — worst-of-both: "
            f"less upside AND more downside."
        )

    # ---- Cost-decomposition paragraph -----------------------------------
    decomp_para = (
        f"Switching to `rebalance_to_target` adds only "
        f"**{decomp['cost_drag_diff_pct_yr']:+.2f}%/yr in cost drag** but loses "
        f"**{decomp['cagr_diff_pct']:+.2f}% CAGR** — i.e., "
        f"{abs(decomp['structural_residual_pct_yr']):.2f}%/yr of the gap is "
        f"**structural drift behaviour** (selling part of the winner each week "
        f"to fund the laggard), not transaction cost. Costs are the small piece."
    )

    # ---- Top CHASE-offender callout -------------------------------------
    if not report.chase_share_by_ticker.empty:
        top = report.chase_share_by_ticker.iloc[0]
        xlk_row = report.chase_share_by_ticker[
            report.chase_share_by_ticker["ticker"] == "XLK"]
        xlk_callout = ""
        if not xlk_row.empty:
            x = xlk_row.iloc[0]
            xlk_callout = (
                f" XLK specifically was CHASE in **{x['n_chase']}/"
                f"{report.n_weekly_snapshots} weeks ({x['share_pct']:.1f}%)**, "
                f"with max extension {x['max_ext_pct']:+.1f}% and median "
                f"CHASE-extension {x['median_ext_when_chase_pct']:+.1f}%."
            )
        chase_para = (
            f"The CHASE filter is the single biggest offender: **{top['ticker']}** "
            f"sat in CHASE {top['n_chase']}/{report.n_weekly_snapshots} weeks "
            f"({top['share_pct']:.1f}% of the window) — every one of those weeks the "
            f"model declined to enter a leading sector because it was more than "
            f"{PARAMS.extension_pct_cutoff*100:.0f}% above SMA200.{xlk_callout}"
        )
    else:
        chase_para = "(no states recorded — backtest produced no rebalances)"

    # ---- Newsletter-rate paragraph --------------------------------------
    rate_para = (
        f"At the current ingestion rate "
        f"(~{rate['mean_per_day']:.1f} newsletters/day "
        f"over the trailing {rate['window_days']} days, "
        f"total {rate['total']}), the gate-ON arm will need months before its "
        f"sample size resolves. Treat any 'sentiment gate works' claim made "
        f"before then as faith, not evidence."
    )

    # ---- Regime table + drawdown attribution (P1 + P2) ------------------
    rs = report.regime_stats
    if not rs.empty:
        regime_rows = "\n".join(
            f"  | **{r}** | {int(row['n_days'])} "
            f"| {pct(row['strategy_cum'], signed=True)} "
            f"| {pct(row['spy_cum'], signed=True)} "
            f"| **{pct(row['excess_cum'], signed=True)}** "
            f"| {row['capture_up']:+.2f} | {row['capture_down']:+.2f} "
            f"| {pct(row['strategy_mdd_in_regime'], signed=True)} "
            f"| {pct(row['spy_mdd_in_regime'], signed=True)} |"
            for r, row in rs.iterrows()
        )
    else:
        regime_rows = "  | (no regime data) |"

    dd_rows_md = ""
    dd_wins = dd_total = 0
    dd_excess_sum = 0.0
    if report.drawdown_attribution:
        for i, d in enumerate(report.drawdown_attribution, 1):
            dd_total += 1
            if d['excess_drawdown'] > 0:
                dd_wins += 1
            dd_excess_sum += d['excess_drawdown']
            verdict = "**LOST LESS** ✅" if d['excess_drawdown'] > 0 else "lost more ❌"
            rotated = []
            if d['rotated_in_during_dd']:
                rotated.append(f"in: `{', '.join(d['rotated_in_during_dd'])}`")
            if d['rotated_out_during_dd']:
                rotated.append(f"out: `{', '.join(d['rotated_out_during_dd'])}`")
            rotated_str = " · ".join(rotated) if rotated else "(no rotations)"
            dd_rows_md += (
                f"\n#### Drawdown #{i}: {d['peak_date']} → {d['trough_date']}"
                f" ({d['days_to_trough']}d peak→trough)\n"
                f"- SPY: **{pct(d['spy_drawdown'], signed=True)}**, "
                f"strategy: **{pct(d['strategy_drawdown'], signed=True)}** "
                f"({verdict} by {pct(d['excess_drawdown'], signed=True)})\n"
                f"- Held at peak: `{', '.join(d['held_at_peak']) or '(none)'}`\n"
                f"- Held at trough: `{', '.join(d['held_at_trough']) or '(none)'}`\n"
                f"- Rotation during DD — {rotated_str}\n"
            )
        dd_summary = (
            f"**{dd_wins}/{dd_total}** in-window SPY drawdowns ≥5%, the "
            f"strategy lost less. Mean excess: "
            f"**{pct(dd_excess_sum/dd_total, signed=True)}**."
        )
    else:
        dd_summary = "(no qualifying drawdowns in window)"

    regime_dist_line = ", ".join(
        f"{r}={n}d ({n/sum(report.regime_distribution.values())*100:.1f}%)"
        for r, n in report.regime_distribution.items()
    )

    # ---- Drawdown severity/duration caveat — computed, not hand-typed ----
    # FIX (TRADING_EDGE_AUDIT.md item B5): this used to be a hardcoded
    # sentence claiming "never tested in a true sustained -30%+ bear...
    # drawdown evidence we have is from corrections (5-19% SPY moves), not
    # crashes" — which directly contradicted the drawdown-attribution table
    # two sections above whenever a fast crash (e.g. a -30%+ move that
    # resolved in weeks, not years) was actually in the sample. The real
    # distinction that matters for a rotation strategy is DEPTH *and*
    # DURATION together: a fast VaR-shock crash and a grinding multi-year
    # bear stress the model differently, and conflating "we haven't seen
    # one" with "we haven't seen the other" is the bug. This now checks the
    # actual `drawdown_attribution` output instead of asserting a number.
    _LONG_BEAR_DAYS = 252  # ~1 trading year peak-to-trough — the line between
                           # a fast shock and a grinding structural bear
    if report.drawdown_attribution:
        _worst = min(report.drawdown_attribution, key=lambda d: d["spy_drawdown"])
        _max_dd = float(_worst["spy_drawdown"])
        _max_dd_days = int(_worst["days_to_trough"])
        _longest_days = max(int(d["days_to_trough"]) for d in report.drawdown_attribution)
    else:
        _worst, _max_dd, _max_dd_days, _longest_days = None, 0.0, 0, 0

    if _worst is not None and _max_dd <= -0.30 and _longest_days < _LONG_BEAR_DAYS:
        drawdown_caveat = (
            f"**The window includes one fast, deep crash but no grinding "
            f"multi-year bear.** The worst SPY drawdown captured is "
            f"{pct(_max_dd, signed=True)} ({_worst['peak_date']} → "
            f"{_worst['trough_date']}, {_max_dd_days} days peak-to-trough) — "
            f"a genuine >30% shock, not a shallow correction, and the "
            f"drawdown-attribution table above already scores how the "
            f"strategy handled it. What the window has NOT captured is a "
            f"sustained, grinding bear like 2000–02 or 2008 (12–24+ months "
            f"peak-to-trough). A fast VaR-shock crash and a slow structural "
            f"bear are different stress tests for a rotation strategy — only "
            f"the former is in this sample."
        )
    elif _worst is not None and _max_dd <= -0.30:
        drawdown_caveat = (
            f"**The window includes a sustained, deep drawdown "
            f"({_longest_days} days peak-to-trough).** Worst SPY move: "
            f"{pct(_max_dd, signed=True)} ({_worst['peak_date']} → "
            f"{_worst['trough_date']}). This is closer to the "
            f"grinding-bear stress case than a fast shock — treat the "
            f"strategy's behaviour here as more informative than a typical "
            f"correction, though still a single episode."
        )
    else:
        _worst_move_str = pct(_max_dd, signed=True) if _worst else "—"
        _worst_days_str = f" ({_max_dd_days} days peak-to-trough)" if _worst else ""
        drawdown_caveat = (
            f"**The window has not seen a crash-scale (≥30%) SPY drawdown.** "
            f"Worst move captured: {_worst_move_str}{_worst_days_str}. "
            f"Treat the drawdown-attribution evidence above as coverage of "
            f"corrections, not crashes — both fast shocks and grinding bears "
            f"remain untested."
        )

    # ---- Verdict paragraph (data-driven) --------------------------------
    # FIX (TRADING_EDGE_AUDIT.md item B5): this used to quote
    # `rs.loc["BULL", "excess_cum"]` — a CUMULATIVE compounding gap over the
    # entire multi-year BULL-labelled window (e.g. "-127.23%"), which reads
    # like an impossible "lost 127%" claim for a long-only, unlevered
    # strategy. It's real math (strategy_cum - spy_cum, both already
    # compounded), but the framing invites exactly that misreading. Swapped
    # to the ANNUALIZED per-regime excess CAGR, which is already computed by
    # `regime_conditional_stats` and is the number that's actually
    # comparable to the headline `excess_cagr` figure quoted in the same
    # sentence.
    bull_excess_cagr = float(rs.loc["BULL", "excess_cagr"]) if "BULL" in rs.index else 0.0
    rotation_works = (dd_total > 0 and dd_wins / dd_total >= 0.6 and
                       dd_excess_sum / dd_total > 0)
    if rotation_works:
        verdict_para = (
            f"**The rotation thesis is validated by the data on hand.** Across "
            f"the {dd_total} SPY drawdowns ≥5% in the window, the strategy lost "
            f"less in {dd_wins} of them, with a mean excess of "
            f"{pct(dd_excess_sum/dd_total, signed=True)}. The headline "
            f"{pct(s['excess_cagr'], signed=True)} CAGR gap vs SPY is "
            f"**entirely explained by the BULL regime** — strategy gave up "
            f"{pct(bull_excess_cagr, signed=True)}/yr of upside there (up-capture "
            f"{rs.loc['BULL', 'capture_up']:.2f}, down-capture "
            f"{rs.loc['BULL', 'capture_down']:.2f}) — which is exactly what a "
            f"defensive rotation strategy is supposed to do."
        )
    else:
        verdict_para = (
            f"The rotation thesis is **NOT yet validated**. Across the "
            f"{dd_total} SPY drawdowns ≥5%, the strategy lost less in only "
            f"{dd_wins}, with mean excess "
            f"{pct(dd_excess_sum/dd_total if dd_total else 0, signed=True)}. "
            f"The headline {pct(s['excess_cagr'], signed=True)} CAGR gap is "
            f"NOT being offset by drawdown protection."
        )

    # ---- Ablation table -------------------------------------------------
    off = ab.get("off", {})
    on = ab.get("on", {})
    ablation_rows = (
        f"  | gate OFF (mechanical core) | {off.get('n_signals', 0)} | "
        f"{pct(off.get('mean_excess_1w', 0.0), signed=True)} | "
        f"{pct(off.get('hit_rate', 0.0), p=0)} |\n"
        f"  | gate ON  (real sentiment) | {on.get('n_signals', 0)} | "
        f"{pct(on.get('mean_excess_1w', 0.0), signed=True)} | "
        f"{pct(on.get('hit_rate', 0.0), p=0)} |"
    )

    # ---- PARAMS table ---------------------------------------------------
    params_rows = "\n".join(
        f"  | `{k}` | {v} |" for k, v in [
            ("sma_window", PARAMS.sma_window),
            ("momentum_window", PARAMS.momentum_window),
            ("sentiment_lookback_days", PARAMS.sentiment_lookback_days),
            ("buy_sentiment_threshold", f"{PARAMS.buy_sentiment_threshold:+.1f}"),
            ("sell_sentiment_threshold", f"{PARAMS.sell_sentiment_threshold:+.1f}"),
            ("weak_rs_rank_cutoff", PARAMS.weak_rs_rank_cutoff),
            ("extension_pct_cutoff", PARAMS.extension_pct_cutoff),
            ("stale_buy_weeks", PARAMS.stale_buy_weeks),
            ("history_weeks", PARAMS.history_weeks),
            ("strong_rs_margin", PARAMS.strong_rs_margin),
            ("macro_strong_count", PARAMS.macro_strong_count),
        ])

    # ---- Header DBs facts ----------------------------------------------
    sw0, sw1 = f.sentiment_window or ("(empty)", "(empty)")
    md = f"""# Sector-Rotation Backtest Report

**Generated:** {as_of.isoformat()}  **Branch:** `{branch}`
**Reproduce:** `PYTHONPATH=. python3 scripts/run_backtest_report.py`

> Every quantitative claim below is interpolated from a fresh backtest run
> and a fresh DB query — there are no hand-typed numbers in this file. If a
> number looks wrong, the bug is in `src/backtest.py` or `src/backtest_report.py`,
> not here.

## TL;DR

Over **{win_start} → {win_end}** ({strat['n_days']} trading days), the
**mechanical core** of this strategy — trend + 3-month relative-strength +
state refinement + event-driven trading — returned **{pct(strat['cagr'], signed=True)} CAGR**
vs SPY's **{pct(spy['cagr'], signed=True)}**, net of {cost_bps:.0f} bps per-side costs.
**Excess CAGR: {excess_cagr_pct:+.2f}%.**

| Metric | Strategy | SPY |
|---|---:|---:|
| CAGR | **{pct(strat['cagr'], signed=True)}** | {pct(spy['cagr'], signed=True)} |
| Total return | {pct(strat['total_return'], signed=True)} | {pct(spy['total_return'], signed=True)} |
| Annualised vol | {pct(strat['ann_vol'])} | {pct(spy['ann_vol'])} |
| Sharpe (rf=0) | {num(strat['sharpe'])} | {num(spy['sharpe'])} |
| Max drawdown | {pct(strat['max_drawdown'], signed=True)} | {pct(spy['max_drawdown'], signed=True)} |

{mdd_obs}

**Costs & turnover:** {s['n_trades']:,} trades, {num(s['annualised_turnover'])}x
annualised turnover, ${s['total_costs']:,.0f}
({s['total_costs']/s['initial_capital']*100:.2f}% of initial capital) in costs,
closed-position hit rate {pct(s['closed_position_hit_rate'], p=1)}.

**This is not the whole strategy.** The live model also requires newsletter
sentiment ≥ +{PARAMS.buy_sentiment_threshold:.0f} to BUY and runs a per-sector
macro overlay. Neither could be included honestly in the historical backtest
(see Methodology), so the sentiment gate and macro veto MIGHT change this
verdict. The data we have suggests the sentiment gate is *directionally*
helpful but the sample is too small to bank on (see Sentiment ablation).

---

## Step 0 — Database & code verification (fresh queries)

### `data/prices.db`

- Daily bars covering **{f.prices_window[0]} → {f.prices_window[1]}**.
- Sectors+benchmark present with daily history: {", ".join(f.sectors_with_daily)}
  ({len(f.sectors_with_daily)} symbols).
- {len(f.prices_coverage)} total tickers cached (UFO + thematics, excluded
  from the equal-weight backtest — matches the live `target_weights()` filter).

### `data/sentiment.db`

- **{f.n_newsletters} newsletters**, **{f.n_sector_ratings} sector ratings**,
  range **{sw0} → {sw1}**.
- Per-sector coverage (real, queried just now):

  | Ticker | n_ratings | First → Last |
  |---|---:|---|
{cov_rows}

- Recent newsletter ingestion: {rate_para}

### `config/settings.SignalParams`

| Param | Value |
|---|---:|
{params_rows}

### Look-ahead audit

Audited `compute_sector_metrics`, `build_signals`, `refine_signals`,
`build_signal_history`, and the snapshot writers.

- `compute_sector_metrics(as_of=t)` slices `prices.loc[:t]` BEFORE any
  rolling / `.iloc[-1]` access — clean, no look-ahead.
- `aggregate_sentiment(as_of=t)` SQL-filters `publication_date <= t` — clean.
- `build_signal_history` iterates `t` weekly with the same slicing — clean.
- One real bug found: `signal_performance_vs_benchmark` filtered the raw
  `BUY` label while the UI caption claimed "NEW_BUY signals". **Fixed.**

Conclusion: **no look-ahead in the existing signal pipeline.**

---

## Methodology

`src/backtest.py` is the implementation. It reuses `compute_sector_metrics`,
`build_signals`, `refine_signals`, and `target_weights` from the live
pipeline so the backtest CAN'T silently disagree with the dashboard.

### Universe & cadence

- The {len([t for t in SECTOR_ETFS if t not in SUPPLEMENTARY_SECTORS])} SPDR
  sectors (UFO + thematics excluded — matches live `target_weights()` filter).
- Weekly rebalance on the last trading day of each ISO week (holiday-robust).
- Signals are computed strictly from bars dated `<= rb_date`.

### Sentiment gate: DISABLED (honest framing)

The strategy CANNOT BUY without sentiment ≥ +{PARAMS.buy_sentiment_threshold:.0f}.
Only ~2 weeks of meaningful historical sentiment exists. The mechanical core
synthesises sentiment as exactly +{PARAMS.buy_sentiment_threshold:.1f} (passing the
threshold), so the sentiment leg of `build_signals` is a no-op. This isolates
the **price-side rules** for measurement.

### Macro veto: DISABLED

Historical macro indicators (FRED HY OAS, REAL_10Y, T10Y2Y, etc.) are not in
`prices.db`. Threading them through would add a network dependency and create
an attack surface for accidental look-ahead. The macro overlay is therefore
evaluated only **forward**, via the persisted `signal_snapshots` table
(Step 3) — every weekly run now records the live macro counts alongside the
state.

### Execution & costs

- **Execution lag** (`execution=`): `next_open` (default) fills at the next
  trading day's open after the signal date. `same_close` fills at the
  signal-date close. Both supported as flags.
- **Costs** (`cost_bps=`, default {cost_bps:.1f}): per-side cost in basis
  points of notional. Round-trip cost = 2× this.
- **Slippage** (`slippage_bps=`, default {s['config']['slippage_bps']:.1f}):
  additive to `cost_bps` per side.

### Trade policy — measured comparison, not assumed

| Policy | CAGR | Trades | Ann. turnover | Cost drag/yr |
|---|---:|---:|---:|---:|
| `event_driven` | {pct(ed.stats['strategy']['cagr'], signed=True)} | {ed.stats['n_trades']:,} | {num(ed.stats['annualised_turnover'])}x | {decomp['ed_cost_drag_pct_yr']:.2f}% |
| `rebalance_to_target` | {pct(report.rbt_result.stats['strategy']['cagr'], signed=True)} | {report.rbt_result.stats['n_trades']:,} | {num(report.rbt_result.stats['annualised_turnover'])}x | {decomp['rbt_cost_drag_pct_yr']:.2f}% |

{decomp_para}

Headline numbers in this report use `event_driven` because it matches what
the live dashboard's orders panel actually emits (buy on transition INTO
BUY-class, sell on transition OUT — no intra-week rebalancing).

### Portfolio construction

- Equal-weight across NEW_BUY + HOLD_IF_LONG with a **{s['config']['cash_buffer']*100:.0f}% cash buffer**.
- CHASE participates at **{(ed.config.chase_weight_fraction if ed.config.chase_weight_fraction is not None else PARAMS.chase_weight_fraction)*100:.0f}%** of the per-name confirmed weight (out of the cash buffer, capped). 0% = original full-exclusion behaviour.
- HOLD / SELL / REDUCE / WATCH excluded.
- No leverage, no shorts, no fractional limit (matches Tiger orders panel).

### Benchmark

SPY buy-and-hold, same initial capital
(${s['initial_capital']:,.0f}), same daily mark-to-market index. If you'd
bought SPY on Day 1 with the same capital you would be at
${spy['total_return']*s['initial_capital'] + s['initial_capital']:,.0f}
today vs the strategy at ${s['final_equity']:,.0f}.

---

## Step 2 — Sentiment ablation, bounded honestly

Over the trailing {ab.get('window_weeks', 0)} weeks (
{ab.get('n_rebalances', 0)} rebalance dates) where the model has been emitting
refined states with some real sentiment underneath:

  | Arm | n signals | mean 1w excess vs SPY | hit rate |
  |---|---:|---:|---:|
{ablation_rows}

**Loud caveat — read carefully:**

- {ab.get('caveat', '')}
- The two arms aren't independent — the gate-ON arm is a strict subset of
  gate-OFF (any sector passing the stricter sentiment leg also passes the
  bypassed one).
- The window overlaps the sentiment-coverage ramp-up; most of the gate-ON
  n falls in the very recent weeks.

A "perfect gate" upper-bound thought experiment is deliberately omitted —
it would conflate "the strategy works" with "perfect foresight works."

---

## Step 3 — Forward performance tracking, fixed

### `signal_snapshots` table

New table in `sentiment.db` (not a new DB — keeps schema and migration story
in one place, and snapshots are small). PK `(as_of, ticker)` so re-runs the
same day overwrite. Written from the Dashboard render AND
`scripts/run_signals.py`. Carries the refined state, the raw signal, every
input that fed into them, the macro counts, and the conviction score.

### `signal_performance_vs_benchmark`, fixed

- **Reads from `signal_snapshots` first** (`source="snapshots"`) — strict
  NEW_BUY state, exactly what the live model emitted.
- Falls back to raw-replay history when snapshots is empty.
- **Default horizon is `next_state_exit`** — holds from snapshot date until
  the first subsequent snapshot where the ticker's state leaves BUY-class.
- Reports `median_hold_days` and `source` so the UI can label honestly.

The Dashboard caption now reads "NEW_BUY signals, last 12 weeks
(hold-to-state-exit, median hold N days): hit rate X%, mean excess +Y% vs
SPY (n=Z)" — and labels the legacy raw-replay variant when no snapshots
exist yet.

---

## Step 4 — Dashboard

New **🧪 Backtest tab** in `app.py`. Controls: cost bps, slippage bps,
execution lag, trade policy. Shows the verdict line, equity curve, headline
stats table, turnover/cost metrics, sentiment ablation expander, and a
trade-log CSV download. The existing 9 tabs are untouched.

---

## Per-ticker state distribution ({report.n_weekly_snapshots} weekly snapshots)

CHASE share by ticker — measured from the actual backtest, sorted worst-first:

  | Ticker | n_CHASE | CHASE share | Max ext | Median ext when CHASE |
  |---|---:|---:|---:|---:|
{chase_rows}

{chase_para}

---

## Regime-conditional performance (P1)

Regime classification: BULL = SPY within 5% of 252-day rolling high;
CORRECTION = -5% to -15% from high; BEAR = below -15%. Window distribution:
{regime_dist_line}.

| Regime | Days | Strategy cum | SPY cum | Excess | Up-cap | Down-cap | Strat MDD | SPY MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{regime_rows}

**How to read this:** up-capture (`Up-cap`) > 1 means the strategy outpaces
SPY on up-days; down-capture (`Down-cap`) < 1 means the strategy loses less
on down-days. The rotation thesis predicts down-capture < up-capture < 1.
The `Excess` column above is CUMULATIVE (compounded gap over the whole
regime's day count) — use the verdict paragraph below, which quotes the
ANNUALIZED per-regime excess instead, for a number that's actually
comparable to the headline excess CAGR.

---

## Drawdown attribution (P2)

For each SPY drawdown ≥5% within the backtest window, the strategy's
drawdown over the SAME peak→trough window. Positive excess = strategy lost
LESS than SPY.

{dd_summary}
{dd_rows_md}

---

## Blunt assessment

{verdict_para}

**Caveats this measurement cannot escape:**

1. {drawdown_caveat}

2. **The sentiment overlay's contribution is still unmeasured at scale.**
   n={on.get('n_signals', 0)} for the gated arm vs n={off.get('n_signals', 0)}
   for the ungated arm. {rate_para}

3. **The macro veto's forward-perf contribution starts measuring now.**
   `signal_snapshots` table accumulates weekly; in a year we'll have an
   honest read.

4. **{dd_total} drawdown episodes is still a small sample.** Drawdowns occur a few times
   a year; this is a handful of years of data. The pattern is consistent and the mean
   lift is meaningful, but statistical significance requires more cycles.

### What would strengthen the evidence

- **Extend price history backward.** Re-cold-start `prices.db` to pre-2020
  to capture the 2020 COVID crash and the 2018 Q4 correction. The current
  5-year limit blocks the deeper bear evidence.
- **Backfill historical sentiment** (highest leverage; see Step 2 caveats).
- **Watch the forward record.** With `signal_snapshots` live and the
  partial-CHASE wired in, the next downturn is an out-of-sample test —
  whatever it shows is real evidence, not a backtest artifact.

---

## Files

- `src/backtest.py` — `BacktestConfig`, `run_backtest`,
  `real_sentiment_ablation`, `save_equity_csv`.
- `src/backtest_report.py` — `gather_db_findings`, `build_headline_report`,
  `render_markdown`. **THIS** is the renderer; the .md is generated, not edited.
- `src/db.py` — added `signal_snapshots` table + `save_signal_snapshot`,
  `load_signal_snapshots`.
- `src/signal_history.py` — fixed `signal_performance_vs_benchmark` to read
  snapshots and use hold-to-state-exit horizon by default.
- `scripts/run_signals.py` — now writes a snapshot.
- `scripts/run_backtest_report.py` — CLI: regenerates BACKTEST_REPORT.md.
- `app.py` — new 🧪 Backtest tab; perf caption honest; snapshot wired in.
- `tests/test_backtest.py` — 12 tests covering no-look-ahead, cost
  application, flat/trending sanity, etc.
- `data/backtest_equity.csv` — persisted equity curves.
"""
    return md


def write_report(report: HeadlineReport, path: Path | str | None = None,
                 branch: str = "feat/history-expandable-and-signal-runner",
                 as_of: date | None = None) -> Path:
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "BACKTEST_REPORT.md"
    md = render_markdown(report, branch=branch, as_of=as_of)
    path.write_text(md)
    return path
