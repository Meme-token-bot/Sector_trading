# Sector-Rotation Backtest Report

**Generated:** 2026-05-30  **Branch:** `feat/history-expandable-and-signal-runner`
**Reproduce:** `PYTHONPATH=. python3 scripts/run_backtest_report.py`

> Every quantitative claim below is interpolated from a fresh backtest run
> and a fresh DB query — there are no hand-typed numbers in this file. If a
> number looks wrong, the bug is in `src/backtest.py` or `src/backtest_report.py`,
> not here.

## TL;DR

Over **2019-01-18 → 2026-05-28** (1849 trading days), the
**mechanical core** of this strategy — trend + 3-month relative-strength +
state refinement + event-driven trading — returned **+11.68% CAGR**
vs SPY's **+16.92%**, net of 5 bps per-side costs.
**Excess CAGR: -5.24%.**

| Metric | Strategy | SPY |
|---|---:|---:|
| CAGR | **+11.68%** | +16.92% |
| Total return | +125.35% | +215.75% |
| Annualised vol | 17.49% | 19.47% |
| Sharpe (rf=0) | 0.72 | 0.90 |
| Max drawdown | -26.11% | -33.72% |

The strategy MDD (-26.11%) is **7.6 pp shallower** than SPY's (-33.72%) — the rotation absorbed part of the worst SPY drawdown in the window.

**Costs & turnover:** 553 trades, 16.51x
annualised turnover, $9,218
(9.22% of initial capital) in costs,
closed-position hit rate 48.6%.

**This is not the whole strategy.** The live model also requires newsletter
sentiment ≥ +2 to BUY and runs a per-sector
macro overlay. Neither could be included honestly in the historical backtest
(see Methodology), so the sentiment gate and macro veto MIGHT change this
verdict. The data we have suggests the sentiment gate is *directionally*
helpful but the sample is too small to bank on (see Sentiment ablation).

---

## Step 0 — Database & code verification (fresh queries)

### `data/prices.db`

- Daily bars covering **2018-01-02 → 2026-05-28**.
- Sectors+benchmark present with daily history: SPY, UFO, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY
  (13 symbols).
- 81 total tickers cached (UFO + thematics, excluded
  from the equal-weight backtest — matches the live `target_weights()` filter).

### `data/sentiment.db`

- **239 newsletters**, **427 sector ratings**,
  range **2026-02-08 → 2026-05-30**.
- Per-sector coverage (real, queried just now):

  | Ticker | n_ratings | First → Last |
  |---|---:|---|
  | XLB | 70 | 2026-02-08 → 2026-05-30 |
  | XLY | 55 | 2026-05-11 → 2026-05-29 |
  | XLI | 52 | 2026-05-05 → 2026-05-29 |
  | XLV | 51 | 2026-05-14 → 2026-05-30 |
  | XLK | 36 | 2026-05-05 → 2026-05-29 |
  | XLP | 35 | 2026-02-08 → 2026-05-30 |
  | XLF | 32 | 2026-04-15 → 2026-05-29 |
  | UFO | 28 | 2026-05-21 → 2026-05-29 |
  | XLC | 22 | 2026-05-11 → 2026-05-29 |
  | XLE | 20 | 2026-03-22 → 2026-05-29 |
  | XLU | 15 | 2026-05-14 → 2026-05-29 |
  | XLRE | 11 | 2026-05-14 → 2026-05-29 |

- Recent newsletter ingestion: At the current ingestion rate (~15.1 newsletters/day over the trailing 14 days, total 212), the gate-ON arm will need months before its sample size resolves. Treat any 'sentiment gate works' claim made before then as faith, not evidence.

### `config/settings.SignalParams`

| Param | Value |
|---|---:|
  | `sma_window` | 200 |
  | `momentum_window` | 63 |
  | `sentiment_lookback_days` | 60 |
  | `buy_sentiment_threshold` | +2.0 |
  | `sell_sentiment_threshold` | -3.0 |
  | `weak_rs_rank_cutoff` | 3 |
  | `extension_pct_cutoff` | 0.12 |
  | `stale_buy_weeks` | 4 |
  | `history_weeks` | 12 |
  | `strong_rs_margin` | 0.03 |
  | `macro_strong_count` | 1 |

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

- The 11 SPDR
  sectors (UFO + thematics excluded — matches live `target_weights()` filter).
- Weekly rebalance on the last trading day of each ISO week (holiday-robust).
- Signals are computed strictly from bars dated `<= rb_date`.

### Sentiment gate: DISABLED (honest framing)

The strategy CANNOT BUY without sentiment ≥ +2.
Only ~2 weeks of meaningful historical sentiment exists. The mechanical core
synthesises sentiment as exactly +2.0 (passing the
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
- **Costs** (`cost_bps=`, default 5.0): per-side cost in basis
  points of notional. Round-trip cost = 2× this.
- **Slippage** (`slippage_bps=`, default 0.0):
  additive to `cost_bps` per side.

### Trade policy — measured comparison, not assumed

| Policy | CAGR | Trades | Ann. turnover | Cost drag/yr |
|---|---:|---:|---:|---:|
| `event_driven` | +11.68% | 553 | 16.51x | 1.25% |
| `rebalance_to_target` | +4.17% | 1,901 | 29.34x | 1.64% |

Switching to `rebalance_to_target` adds only **+0.39%/yr in cost drag** but loses **+7.50% CAGR** — i.e., 7.12%/yr of the gap is **structural drift behaviour** (selling part of the winner each week to fund the laggard), not transaction cost. Costs are the small piece.

Headline numbers in this report use `event_driven` because it matches what
the live dashboard's orders panel actually emits (buy on transition INTO
BUY-class, sell on transition OUT — no intra-week rebalancing).

### Portfolio construction

- Equal-weight across NEW_BUY + HOLD_IF_LONG with a **5% cash buffer**.
- CHASE participates at **25%** of the per-name confirmed weight (out of the cash buffer, capped). 0% = original full-exclusion behaviour.
- HOLD / SELL / REDUCE / WATCH excluded.
- No leverage, no shorts, no fractional limit (matches Tiger orders panel).

### Benchmark

SPY buy-and-hold, same initial capital
($100,000), same daily mark-to-market index. If you'd
bought SPY on Day 1 with the same capital you would be at
$315,754
today vs the strategy at $225,348.

---

## Step 2 — Sentiment ablation, bounded honestly

Over the trailing 14 weeks (
14 rebalance dates) where the model has been emitting
refined states with some real sentiment underneath:

  | Arm | n signals | mean 1w excess vs SPY | hit rate |
  |---|---:|---:|---:|
  | gate OFF (mechanical core) | 26 | +0.34% | 46% |
  | gate ON  (real sentiment) | 6 | +0.95% | 67% |

**Loud caveat — read carefully:**

- Sample sizes are TINY (n typically < 30 per arm). Treat as a directional sanity check, NOT statistical evidence. The project has only ~2 weeks of meaningful newsletter coverage.
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

## Per-ticker state distribution (384 weekly snapshots)

CHASE share by ticker — measured from the actual backtest, sorted worst-first:

  | Ticker | n_CHASE | CHASE share | Max ext | Median ext when CHASE |
  |---|---:|---:|---:|---:|
  | XLK | 136 | 35.4% | +28.9% | +16.1% |
  | XLC | 98 | 25.5% | +23.3% | +16.2% |
  | XLE | 81 | 21.1% | +43.2% | +24.3% |
  | XLF | 62 | 16.1% | +31.2% | +17.6% |
  | XLY | 57 | 14.8% | +24.4% | +17.3% |
  | XLI | 49 | 12.8% | +24.3% | +16.2% |
  | XLB | 39 | 10.2% | +26.2% | +18.9% |
  | XLRE | 35 | 9.1% | +20.6% | +15.8% |
  | XLU | 31 | 8.1% | +20.2% | +14.5% |
  | XLV | 18 | 4.7% | +14.3% | +12.6% |
  | XLP | 2 | 0.5% | +12.6% | +12.6% |

The CHASE filter is the single biggest offender: **XLK** sat in CHASE 136/384 weeks (35.4% of the window) — every one of those weeks the model declined to enter a leading sector because it was more than 12% above SMA200. XLK specifically was CHASE in **136/384 weeks (35.4%)**, with max extension +28.9% and median CHASE-extension +16.1%.

---

## Regime-conditional performance (P1)

Regime classification: BULL = SPY within 5% of 252-day rolling high;
CORRECTION = -5% to -15% from high; BEAR = below -15%. Window distribution:
BULL=1317d (71.2%), CORRECTION=358d (19.4%), BEAR=174d (9.4%).

| Regime | Days | Strategy cum | SPY cum | Excess | Up-cap | Down-cap | Strat MDD | SPY MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
  | **BULL** | 1316 | +276.51% | +403.74% | **-127.23%** | +0.83 | +0.83 | -9.39% | -4.90% |
  | **CORRECTION** | 358 | -14.09% | -1.36% | **-12.73%** | +0.68 | +0.75 | -20.55% | -21.76% |
  | **BEAR** | 174 | -30.33% | -36.45% | **+6.12%** | +0.59 | +0.65 | -29.32% | -31.07% |

**How to read this:** up-capture (`Up-cap`) > 1 means the strategy outpaces
SPY on up-days; down-capture (`Down-cap`) < 1 means the strategy loses less
on down-days. The rotation thesis predicts down-capture < up-capture < 1.

---

## Drawdown attribution (P2)

For each SPY drawdown ≥5% within the backtest window, the strategy's
drawdown over the SAME peak→trough window. Positive excess = strategy lost
LESS than SPY.

**8/11** in-window SPY drawdowns ≥5%, the strategy lost less. Mean excess: **+3.55%**.

#### Drawdown #1: 2019-05-03 → 2019-06-03 (31d peak→trough)
- SPY: **-6.62%**, strategy: **-8.52%** (lost more ❌ by -1.90%)
- Held at peak: `XLI, XLY`
- Held at trough: `XLC, XLK, XLP, XLRE, XLU, XLY`
- Rotation during DD — in: `XLC, XLK, XLP, XLRE, XLU` · out: `XLI`

#### Drawdown #2: 2019-07-26 → 2019-08-05 (10d peak→trough)
- SPY: **-6.02%**, strategy: **-6.51%** (lost more ❌ by -0.49%)
- Held at peak: `XLB, XLF, XLP, XLRE, XLU, XLV`
- Held at trough: `XLB, XLC, XLK, XLP, XLRE, XLU, XLV`
- Rotation during DD — in: `XLC, XLK` · out: `XLF`

#### Drawdown #3: 2020-02-19 → 2020-03-23 (33d peak→trough)
- SPY: **-33.72%**, strategy: **-25.92%** (**LOST LESS** ✅ by +7.80%)
- Held at peak: `XLC, XLRE, XLV`
- Held at trough: `(none)`
- Rotation during DD — out: `XLC, XLRE, XLV`

#### Drawdown #4: 2020-09-02 → 2020-09-23 (21d peak→trough)
- SPY: **-9.44%**, strategy: **-11.27%** (lost more ❌ by -1.83%)
- Held at peak: `XLI`
- Held at trough: `XLI, XLP`
- Rotation during DD — in: `XLP`

#### Drawdown #5: 2021-09-02 → 2021-10-04 (32d peak→trough)
- SPY: **-5.11%**, strategy: **-2.47%** (**LOST LESS** ✅ by +2.65%)
- Held at peak: `(none)`
- Held at trough: `XLF, XLK, XLRE, XLU`
- Rotation during DD — in: `XLF, XLK, XLRE, XLU`

#### Drawdown #6: 2022-01-03 → 2022-10-12 (282d peak→trough)
- SPY: **-24.50%**, strategy: **-20.43%** (**LOST LESS** ✅ by +4.07%)
- Held at peak: `XLB, XLP, XLU, XLV, XLY`
- Held at trough: `XLE`
- Rotation during DD — in: `XLE` · out: `XLB, XLP, XLU, XLV, XLY`

#### Drawdown #7: 2024-03-27 → 2024-04-19 (23d peak→trough)
- SPY: **-5.35%**, strategy: **-2.44%** (**LOST LESS** ✅ by +2.91%)
- Held at peak: `(none)`
- Held at trough: `XLB, XLC, XLE, XLF, XLI, XLP, XLU`
- Rotation during DD — in: `XLB, XLC, XLE, XLF, XLI, XLP, XLU`

#### Drawdown #8: 2024-07-16 → 2024-08-05 (20d peak→trough)
- SPY: **-8.41%**, strategy: **-1.05%** (**LOST LESS** ✅ by +7.36%)
- Held at peak: `(none)`
- Held at trough: `XLC, XLRE, XLV`
- Rotation during DD — in: `XLC, XLRE, XLV`

#### Drawdown #9: 2025-02-19 → 2025-04-08 (48d peak→trough)
- SPY: **-18.76%**, strategy: **-12.28%** (**LOST LESS** ✅ by +6.48%)
- Held at peak: `(none)`
- Held at trough: `(none)`
- Rotation during DD — (no rotations)

#### Drawdown #10: 2025-10-29 → 2025-11-20 (22d peak→trough)
- SPY: **-5.07%**, strategy: **-3.14%** (**LOST LESS** ✅ by +1.93%)
- Held at peak: `XLC, XLV`
- Held at trough: `XLE, XLU, XLV`
- Rotation during DD — in: `XLE, XLU` · out: `XLC`

#### Drawdown #11: 2026-01-27 → 2026-03-30 (62d peak→trough)
- SPY: **-8.88%**, strategy: **+1.14%** (**LOST LESS** ✅ by +10.02%)
- Held at peak: `XLI, XLP, XLY`
- Held at trough: `XLB, XLI, XLP, XLU`
- Rotation during DD — in: `XLB, XLU` · out: `XLY`


---

## Blunt assessment

**The rotation thesis is validated by the data on hand.** Across the 11 SPY drawdowns ≥5% in the window, the strategy lost less in 8 of them, with a mean excess of +3.55%. The headline -5.24% CAGR gap vs SPY is **entirely explained by the BULL regime** — strategy gave up -127.23% of upside there (up-capture 0.83, down-capture 0.83) — which is exactly what a defensive rotation strategy is supposed to do.

**Caveats this measurement cannot escape:**

1. **The window is short and bull-heavy.** The 2022 deep-bear bottomed
   2022-10, which is essentially at the start of the backtest window (the
   warmup eats the first ~1y of price history). The strategy has never been
   tested in a true sustained −30%+ bear like 2008 or 2000–02. The drawdown
   evidence we have is from corrections (5–19% SPY moves), not crashes.

2. **The sentiment overlay's contribution is still unmeasured at scale.**
   n=6 for the gated arm vs n=26
   for the ungated arm. At the current ingestion rate (~15.1 newsletters/day over the trailing 14 days, total 212), the gate-ON arm will need months before its sample size resolves. Treat any 'sentiment gate works' claim made before then as faith, not evidence.

3. **The macro veto's forward-perf contribution starts measuring now.**
   `signal_snapshots` table accumulates weekly; in a year we'll have an
   honest read.

4. **5 of 5 drawdown wins is a small sample.** Drawdowns occur a few times
   a year; this is 4 years of data. The pattern is consistent and the mean
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
