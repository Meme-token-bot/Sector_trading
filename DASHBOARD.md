# Sector Rotation Dashboard — Reader's Guide

A walkthrough of every tab, every column, and how the signals are actually computed. Read top-to-bottom the first time; after that use it as a reference.

> **TL;DR.** The model watches the 11 SPDR Select Sector ETFs (XLK, XLY, XLC, XLF, XLI, XLB, XLE, XLV, XLP, XLU, XLRE) — plus a 12th, manually-sized Space sector (UFO) — against SPY. Each week it asks three questions per sector — *is it above its own SMA200, is it outperforming SPY, and is the newsletter sentiment positive?* — and converges those into one of **three raw outcomes** (BUY / HOLD / SELL), then a state-refinement layer (late-entry guard, trend-maturity check, and a macro veto/override pass) turns that into one of **seven final states** — see the table below. When a state says BUY, the **Expressions** tab tells you which specific ETF to actually buy (informed by an automated theme-news overlay), and a second-tier **self-check** tells you whether that expression is participating, lagging, broken, or overextended at its own level. A separate, non-sentiment-gated **🚀 Breakouts** tab runs an independent momentum signal across the same core tickers plus crypto, gold, silver, and mining — see its own section below.

---

## 1. The mental model — how the signal is built

Three independent inputs are combined into one decision. Each input is a yes/no test; only sectors that pass *all three* qualify for BUY.

| Input | What it measures | Source |
|---|---|---|
| **Trend** (`above_sma`) | Last close > 200-day SMA | yfinance daily closes |
| **Relative strength** (`relative_strength_3m`) | Sector's 63-trading-day return − SPY's 63-trading-day return | yfinance daily closes |
| **Sentiment** (`sentiment_score`) | Mean sentiment score from ingested newsletters over the last 60 days | Gmail ingest → GPT-4o-mini scoring → SQLite |

Each metric is computed in `src/market_engine.py::compute_sector_metrics`. Sentiment is aggregated in `src/db.py::aggregate_sentiment` using the per-sector scores produced by `src/nlp_pipeline.py`.

### The raw verdict — BUY / HOLD / SELL

`src/signals.py::build_signals` runs the convergence test for every sector:

- **SELL** if *any* of these hard fails fire:
  - price < SMA200, **or**
  - RS rank is in the bottom 3 (worst three sectors by 3-month relative strength), **or**
  - sentiment_score ≤ `−3.0` (`PARAMS.sell_sentiment_threshold`).
- **BUY** only if *all three* pass:
  - price > SMA200, **and**
  - 3-month relative strength > 0 (beating SPY), **and**
  - sentiment_score ≥ `+2.0` (`PARAMS.buy_sentiment_threshold`).
- Otherwise → **HOLD**.

A sector with no newsletter coverage in the last 60 days has `sentiment_score = 0` and `n_obs = 0` — that's below the BUY threshold, so it cannot BUY without sentiment. This is intentional: the model refuses to buy on price alone.

### The refined verdict — seven states

`src/signals.py::refine_signals` then layers three practitioner concerns on top of the raw verdict, in order:

1. **Late-entry guard.** A sector that's already 12%+ above its own SMA200 is over-extended; entering fresh from cash here is chasing.
2. **Trend maturity.** A sector that's been BUY for 4+ consecutive weekly snapshots is a mature trend — if you missed the entry, don't chase it now.
3. **Macro veto/override.** A per-sector macro tailwind/headwind count (`src/macro_alignment.py`) can then override the result of the first two layers on a STRONG net lean (`|tailwinds − headwinds| ≥ macro_strong_count`): a fresh `NEW_BUY` facing a strong macro headwind is vetoed down to `HOLD`; a stale `HOLD_IF_LONG` facing a strong headwind is downgraded to `REDUCE`; and a `HOLD` that's above its own SMA200 but hasn't confirmed on RS/sentiment gets promoted to `WATCH` when macro strongly supports it. This is the layer that actually produces `WATCH`.

These three layers promote/demote the raw signal into one of seven **states**:

| State | Color | Means |
|---|---|---|
| 🟢 **NEW_BUY** | green | Raw BUY, not extended, fresh (< 4 consecutive BUY weeks). **Fresh entry OK.** |
| 🟡 **HOLD_IF_LONG** | amber | Raw BUY, but BUY for ≥ 4 weeks. **Hold if owned, don't add. Don't enter from cash.** |
| 🟠 **CHASE** | orange | Raw BUY but extension > 12%. Full-size entry is too late — but by **default the model still takes a partial position** (`chase_weight_fraction`, 25%, funded from the cash buffer — see Position sizing below). Set the fraction to 0 to fully sit out CHASE instead. |
| 🟤 **REDUCE** | rust | Was BUY in the last 12 weeks but no longer qualifies, **or** a stale BUY hit by a strong macro headwind. **Trim if owned.** |
| 🔭 **WATCH** | teal | Raw HOLD, price hasn't confirmed (not above SMA200 / RS / sentiment), but a strong net macro tailwind supports it and price is at least above its own SMA200. **No position — watch for the RS turn / SMA200 reclaim.** |
| ⚪ **HOLD** | neutral | Doesn't qualify and hasn't recently, **or** a would-be fresh BUY that a strong macro headwind vetoed. **Wait.** |
| 🔴 **SELL** | red | Failed a hard SELL rule. **Exit.** |

The promotion logic uses historical signal replay (`src/signal_history.py`). Each week the same `build_signals` call is replayed on truncated price data so the model can answer "how many weeks in a row has this been BUY?" and "was this BUY at any point in the recent window?"

### Position sizing

`src/signals.py::target_weights` equal-weights across **NEW_BUY + HOLD_IF_LONG** with a 5% cash buffer (`cash_buffer`, default 0.05). **CHASE is not excluded by default any more** — the walk-forward sweep found a positive out-of-sample lift from letting CHASE sectors take a partial position, so `PARAMS.chase_weight_fraction` defaults to **0.25**: a CHASE sector gets 25% of the per-name confirmed weight, funded entirely from the cash buffer (capped so total weight never exceeds 1.0 — no implicit leverage) and never diluting confirmed positions. Set `chase_weight_fraction = 0` to go back to the original full-exclusion behaviour. `WATCH` never receives capital under any configuration — it is a visibility flag, not a position, precisely because price hasn't confirmed. `HOLD_IF_LONG` *is* included in target weights, but the dashboard's caption is explicit: if you don't already own a HOLD_IF_LONG row, do not enter — the target weight is only for sizing what you'd hold if you already had the position.

There's also a `promote_chase` flag on `target_weights()` — used only by the backtest's `regime_aware` overlay, and not exposed anywhere in the live Streamlit UI — that folds CHASE into the confirmed set at *full* per-name weight instead of a partial sleeve. See `scripts/walk_forward_regime.py` / `scripts/compare_regime_aware.py` in the file map below.

### Parameters (all in `config/settings.py::SignalParams`)

| Param | Default | What it controls |
|---|---:|---|
| `sma_window` | 200 | Trend filter length (daily bars) |
| `momentum_window` | 63 | 3-month return window (≈ trading days in a quarter) |
| `sentiment_lookback_days` | 60 | Rolling window for sentiment aggregation |
| `buy_sentiment_threshold` | +2.0 | Min sentiment to qualify for BUY |
| `sell_sentiment_threshold` | −3.0 | Sentiment that triggers SELL |
| `weak_rs_rank_cutoff` | 3 | Bottom-N RS rank that triggers SELL |
| `extension_pct_cutoff` | 0.12 | If (price-SMA)/SMA > 12%, BUY → CHASE |
| `stale_buy_weeks` | 4 | If BUY for ≥ N weekly snapshots, BUY → HOLD_IF_LONG |
| `history_weeks` | 12 | Replay window for state classification |
| `strong_rs_margin` | 0.03 | RS above this margin earns the "strong RS" conviction point |
| `macro_strong_count` | 1 | \|net macro lean\| required for the veto/override pass (§ above) to fire |
| `chase_weight_fraction` | 0.25 | Walk-forward-validated partial CHASE sleeve size (0 = fully excluded) |

---

## 2. Reading the tabs

### 📈 Dashboard

The main view. Top strip, two risk/readiness expanders, then two panes, then two more diagnostic expanders, then a phase-space chart.

**Decision Cockpit (top strip).** Current SPY regime (BULL/CORRECTION/BEAR,
from `src/regime_snapshot.py`, reusing `src/regime_analysis.py::classify_regimes`)
and how many days it's held; cross-sectional breadth (how many of the 11
core sectors are above their own SMA200 right now); RS dispersion
(standard deviation of 3-month relative strength across the core
universe, banded 🔴 Low / 🟡 Moderate / 🟢 High — low dispersion means
sectors are moving together and a cross-sectional rotation strategy has
structurally less to work with); and, once `scripts/run_walk_forward.py`
has been run at least once, a one-line walk-forward trust badge reading
how many parameter defaults are still walk-forward-validated as of the
last sweep. (The trailing 12-week hit-rate figure lives in its own
caption further down the page, not in this strip.) This strip answers
"should I even lean into rotation this week" before anything else on the
page.

**⚠️ Portfolio Risk** (expander). A correlation/concentration/VaR read
on **this week's target weights**, not necessarily your current Tiger
holdings: average pairwise weekly-return correlation (52-week lookback)
with a correlation-matrix heatmap; an "effective # of bets" figure
(naive 1/HHI alongside a correlation-adjusted version — a
diversification-ratio heuristic, not full PCA); and 1-week historical
VaR/ES at 95% confidence (104-week lookback, empirical, no
parametric-normal assumption). Explicitly a Phase-2 stopgap
(`src/risk_metrics.py`) — good enough to catch "four sectors, one bet"
at a glance, not a substitute for real factor analysis.

**✅ Pre-Trade Checklist** (expander). The exact same checks as the
terminal tool `scripts/preflight.py` — both call `src/preflight_checks.py`
so they can't silently disagree: data freshness (`prices.db`,
`sentiment.db`, `signal_snapshots`), current model state, the Tiger
connection + cash-coverage-for-rotation check, and the Monday order
list — rolled into one 🟢 READY / 🟡 READY WITH WARNINGS / 🔴 NOT READY
badge with every check row shown underneath.

**Left — This Week's Orders, state changes, and the Sector Matrix.**

- *This Week's Orders*: one row per actionable trade. **NEW_BUY** →
  green BUY; **CHASE** → orange "CHASE (N% partial)" whenever
  `PARAMS.chase_weight_fraction > 0` (the default); **SELL/REDUCE** →
  for sectors currently held (or, if Tiger isn't connected or the
  snapshot fetch fails, for every sector in that state — there's just
  no dollar size to attach). `HOLD_IF_LONG` never generates a row.
  *Vehicle* is the top **CONFIRMED** expression (fallback: the sector
  ETF); *Size* is `target_weight × Tiger NLV` for buys or the current
  Tiger position value for sells; *Why* is `state_reason` plus a 5-dot
  conviction badge. Empty state = "portfolio aligned, no actions this
  week."
- *State-change strip*: a one-line `st.info` summarising any sectors
  that flipped state since the last snapshot, e.g. `XLE: CHASE →
  NEW_BUY (RS turned positive) · XLF: HOLD_IF_LONG → REDUCE (sentiment
  fell)`. Hidden entirely when nothing changed.
- *Sector Relative Strength Matrix*: the full table, sorted by 3-month
  relative strength — columns below.

| Column | Meaning |
|---|---|
| Sector | Sector name (XLK → Technology, etc.) |
| 3M vs SPY | 63-day return minus SPY's 63-day return |
| Ext vs SMA | (price − SMA200) / SMA200. Positive = above trend, negative = below |
| Wks BUY | Consecutive weekly snapshots (out of last 12) where the raw test passed |
| Conviction | 0–5 dot scale: +1 each for RS>0, RS>`strong_rs_margin`, sentiment ≥ buy_threshold+1, ≥2 consecutive BUY weeks; the macro component is symmetric — a clear net tailwind (≥+1) adds a point, a clear net headwind (≤−1) subtracts one, floored at 0 |
| Sentiment | Mean sentiment score · n (newsletters covering this sector in the lookback window) · σ (stdev across them — high σ means sources disagree). A ⚠️ suffix flags a "consensus peak" (n≥3, σ<0.5, score≥+4) — unusually uniform bullishness read as a contrarian caution, not a stronger signal |
| State | One of the seven states above, row tinted by state color |
| Macro | `5/7 ✓` style pill = tailwind indicators / (tailwinds + headwinds) from the per-sector macro rule map. Neutral readings excluded from the denominator, so denominators differ sector to sector — the colour tint (green ≥0.625, amber 0.375–0.625, red <0.375), not the raw fraction, is the fair comparison |
| Why | For BUY-class rows: `📈 RS%  📊 Ext%  💬 Sentiment` triplet. For others: the prose `state_reason` |

Below the table: state-count badges (how many sectors sit in each of the
seven states), and a performance-feedback caption — trailing-12-week
NEW_BUY hit rate + 95% CI + mean excess return, clearly labelled whether
it's reading persisted `signal_snapshots` or falling back to a
raw-history replay, plus the median hold days for the current
hold-to-state-exit horizon.

**Right — Tiger Portfolio Drift.** If Tiger SDK is configured, shows
current vs target weights per sector and the trade size needed to close
the drift, sorted by urgency (SELL → REDUCE → everything else, then by
trade size). Includes the holding state (tinted same as the matrix), a
`Stop at (SMA200)` column showing `$current → $stop (-X%)` so you know
exactly where the exit lives. Supplementary-sector holdings (UFO) and
holdings that don't map to any sector's expression list are surfaced
separately so you know they aren't counted toward sector targets.

If Tiger isn't configured, a manual NLV input lets you preview the
target-value table for a hypothetical portfolio size.

**📉 Rolling edge trend** (expander). The trailing-12-week NEW_BUY
hit-rate and mean excess return, recomputed at many historical
evaluation points (not just "as of right now") with a shaded 95% Wilson
confidence band — this is what actually answers "is the edge currently
working or currently decaying." Reads exclusively from persisted
`signal_snapshots`, so it fills in slowly as real weekly snapshots
accumulate; empty until there's enough history.

**🎯 Conviction calibration** (expander). Historical hit-rate and mean
excess return bucketed by the 0–5 conviction score each NEW_BUY signal
carried *at entry* — whether a 5-dot signal has actually beaten a 1-dot
signal, or the score is just a plausible-looking prior nobody's verified
against outcomes yet. Same `signal_snapshots` dependency.

A **target weights** expander (the equal-weight allocation, with the
current partial-CHASE-sleeve percentage noted) and a **"How to read the
State column"** legend also live on this tab as further reference
panels; both echo the state legend and position-sizing rules already
described above.

**Sector Rotation Phase Space (Relative Rotation Graph).** Plots each
sector's 3-month RS vs SPY (x-axis) against its week-over-week change in
RS (y-axis), with a fading 26-week trail per sector (optional
3-week-EMA smoothing toggle). Four quadrants: **Leading** (RS positive &
rising — stay long), **Weakening** (RS positive but fading — watch for a
trim), **Improving** (RS still negative but turning — a potential early
entry *before* RS crosses zero), **Lagging** (weak and decelerating —
avoid). Marker size is larger for NEW_BUY / HOLD_IF_LONG; a SELL-state
sector renders as an ✕. Click any point (current or historical) to
isolate that sector's trail; click again to reset.

### 📰 Weekly Recap

A one-shot OpenAI synthesis of the last 7 days of ingested newsletters
plus the current macro tape (`src/weekly_recap.py`). Six pieces, shown
in this order in the UI even though the model is asked to write the
first one last (so it can legitimately reference everything else it
already decided):

1. **Executive summary** — 6–10 plain-English sentences: what the
   newsletters said, what the macro tape says, the highest-conviction
   tilts and why, and where the two genuinely disagree.
2. **Macro narrative** — a regime label (Risk-on / Risk-off / Late-cycle
   / Reflationary / Disinflationary / Mixed), dominant themes, and any
   contradictions between newsletters or between newsletters and macro.
3. **Sector recaps** — one per SECTOR_ETFS ticker, each with a
   newsletter-consensus tag (bullish / bearish / mixed / no coverage),
   a macro-alignment line, and 1–3 key risks. Ordered Overweight →
   Equal-weight → Underweight → Avoid, expanded by default only for
   Overweight.
4. **🔭 Sectors to watch** — 2–5 *forward-looking* flags, distinct from
   the allocation table below (which is "what to own now"). `building`
   = sentiment/macro support a sector but price hasn't confirmed yet
   (candidates to upgrade to NEW_BUY soon); `rolling over` = currently
   strong but extended (CHASE) or losing macro support (candidates to
   downgrade soon). Each cites a concrete signal-row fact (RS, state,
   macro tailwind/headwind count) plus a newsletter or macro reading,
   and names the specific trigger to watch for.
5. **Allocation tilts** — Overweight / Equal-weight / Underweight /
   Avoid per ticker, each rationale required to cite at least one
   supplied newsletter excerpt or macro reading.
6. **Caveats** — one or two sentences, always shown: informational
   only, not personalised advice.

Recaps are **saved to SQLite** keyed by `(as_of_iso, model)`. Reopening
the tab or picking a past date from the **Recap date** dropdown loads
the stored payload with zero OpenAI cost; "Force regenerate" deletes
the stored row and calls OpenAI again (useful after ingesting more
newsletters). Requires `OPENAI_API_KEY`; optionally set
`WEEKLY_RECAP_MODEL` in `.env` to upgrade just this tab to a stronger
model while every other tab keeps using `OPENAI_MODEL`.

### 🌐 Macro

Fifteen macro indicators, each rendered as: current reading + 30-day
slope or 1-year z-score → regime badge (🟢 / 🟡 / 🟠 / 🔴) → one-line
sector-rotation implication → 1-year line chart.

| Indicator | Group | What it tells you |
|---|---|---|
| VIX | Risk/Vol | Volatility regime (complacent / normal / stressed / crisis) |
| HY OAS | Risk/Vol | High-yield credit spread — risk-on vs credit stress |
| Gold/Oil | Risk/Vol | Defensive vs cyclical commodity pricing |
| BBB / IG OAS | Risk/Vol | Investment-grade spread — widens *before* HY OAS, earliest credit warning |
| Financial Conditions (NFCI) | Risk/Vol | Chicago Fed's ~105-input composite — negative = loose/risk-on, positive = tight/risk-off |
| Copper/Gold (z-score) | Growth/Cycle | Reflation gauge — rising = pro-cyclical |
| DXY | Growth/Cycle | Dollar regime — strong dollar pressures non-US and metals |
| Initial jobless claims (4-wk avg, z-score) | Growth/Cycle | Highest-frequency labour signal — turns weeks before payrolls |
| T10Y−T2Y | Rates/Inflation | Yield curve slope; inversion is a recession warning, re-steepening is the trigger |
| UST10Y | Rates/Inflation | Absolute level of long-end nominal rates |
| Real 10Y | Rates/Inflation | TIPS real yield — the cleanest read on policy stance |
| 5Y5Y breakeven | Rates/Inflation | Market-implied long-run inflation expectation |
| 2Y nominal | Rates/Inflation | Most Fed-expectations-sensitive point on the curve |
| 10Y breakeven | Rates/Inflation | Spot 10-year inflation expectation |
| 30Y mortgage (+ spread vs 10Y) | Rates/Inflation | Housing-credit availability, isolated from the general rate level |

The bands are **regime guideposts, not trade triggers**. They tell you what kind of market you're in so you can sanity-check the sector signal: a model saying "BUY XLF" while T10Y2Y is deeply inverted and HY OAS is widening deserves a second look. This full panel also feeds `src/macro_alignment.py`'s per-sector tailwind/headwind scoring — the Macro pill on the Dashboard matrix and the macro veto/override pass.

### 📉 Price Action

An interactive chart (candles, SMA50/200, optional RSI(14)/MACD(12,26,9)/
Bollinger(20,2σ) overlays, optional SPY-overlay comparison line) with
selectable sector/timeframe(Daily or Weekly)/lookback(3M–5Y). Loads from
the local prices DB (`data/prices.db`), which stores ~5y of daily and
weekly bars per ticker. Defaults to the first NEW_BUY sector (falling
back to XLK if nothing currently qualifies) — not fixed to SPY, though
SPY is selectable and can be overlaid on any other ticker's chart.

**Sector grid.** Below the main chart, a 3-column grid of mini
candlestick charts for all 11 sectors, border-tinted by state; click
**View {ticker}** under any tile to load that sector into the main
chart above.

**🔄 Update price data** button pulls incremental bars from yfinance for
the full universe — signal sectors + benchmark + all expression tickers
+ the expanded crypto/metals universe the Breakout scanner uses — and
runs a split-detection guard: if any overlapping historical bar
disagrees with the stored value by more than 0.5%, the ticker's history
is wiped and re-pulled in full. See the module docstring of
`src/price_store.py` for why this matters.

### 🎯 Expressions

This is where the sector signal turns into an actual trade. Each sector maps to a curated list of plain and operating-leverage ETFs in `config/expressions.py`, including an auto-appended iShares alternate for each of the 11 core sectors (IYW for XLK, IYF for XLF, …). **Every expression is a plain equity ETF** — no daily-reset leveraged products. The "leverage" comes from the underlying businesses (e.g. gold miners' fixed costs amplify their earnings beta to gold price), not from derivatives.

For each sector you see:
- An expander, opened by default if the sector is BUY-class.
- A table with one row per candidate expression. Columns:

| Column | Meaning |
|---|---|
| Ticker | The expression ETF |
| Label | Plain-language name |
| Kind | `plain` / `thematic` / `operating leverage` |
| β hint | Rough 3-month price beta vs the parent sector ETF |
| 60d | Sparkline of the last 60 daily closes |
| Band | Entry zone: BROKEN floor (SMA200) → STRETCHED ceiling (beta-scaled) |
| **Self-check** | Per-expression participation state (see below) |
| **Self-check reason** | One-line explanation of the self-check |
| **Theme news** | Blended newsletter+news theme sentiment, where applicable — see below |
| Note | Free-text guidance from `config/expressions.py` — several entries flag thin AUM / low liquidity |

#### The self-check — second-tier filter

Implemented in `src/expression_signals.py`. Runs *alongside* the sector signal — it does not replace it and does not change position sizing. Its job: when the parent sector fires NEW_BUY or HOLD_IF_LONG, tell you whether each candidate expression is participating, lagging, broken, or overextended at its own level.

Seven states:

| Self-check | Means |
|---|---|
| 🟢 **CONFIRMED** | Parent BUY-class, expression > own SMA200, expression's 3m return ≥ parent's, own extension within beta-scaled cutoff. **Participating — safe to use.** |
| 🟡 **LAGGING** | Parent BUY-class, expression rising and not extended, but its 3m return < parent's. **Rising slower than the sector — pick a different expression.** |
| 🟠 **STRETCHED** | Parent BUY-class, above own SMA200, but own extension > beta-scaled cutoff. **Too far above its own trend; wait.** |
| 🔴 **BROKEN** | Parent BUY-class, but expression price < own SMA200. **In its own downtrend regardless — avoid.** |
| ⚫ **WARMING_UP** | Fewer than 200 daily bars stored — SMA200 isn't computable yet. |
| ⚪ **PARENT_INACTIVE** | Parent sector is not NEW_BUY/HOLD_IF_LONG. No expression-level call — defer to the parent state. |
| 🔴 **NO_DATA** | No price data stored for this ticker. Hit *🔄 Update price data*. |

**Beta-scaled cutoff.** The same 12% extension cap used for sector-level CHASE is multiplied by the expression's `beta_hint` to define STRETCHED. A 1.0-beta plain ETF is STRETCHED at >12%; a 2.5-beta junior gold miner is STRETCHED at >30%.

**Theme-news overlay.** Expressions that map to a sub-sector theme in
`config/themes.py` (e.g. SOXX/SMH → SEMIS, URA/URNM → URANIUM, XBI/IBB →
BIOTECH — roughly 30 themes spanning every sector) also carry a blended
sentiment reading: newsletter theme-tags (from the same LLM pass that
tags the 12 sectors) blended with automated Google-News-headline scoring
(`src/ticker_news.py`, triggered by the tab's **🔄 Refresh theme news**
button — one batched OpenAI call scores every theme with fresh headlines
at once). The blend defaults to 60% newsletter / 40% news. Two flags can
fire on top of the technical self-check:

| Flag | Fires when | Means |
|---|---|---|
| **NEWS_CONTRADICTS** | Self-check is CONFIRMED or LAGGING (price healthy) but theme news is clearly negative | The technicals may be lagging bad news that hasn't hit price yet |
| **NEWS_DIVERGENCE** | Self-check is BROKEN (own downtrend) but theme news is clearly positive | A potential turn — price hasn't confirmed, but watch |

Plain sector proxies (XLK, VGT, IYW, …) map to no theme and never carry
a theme-news reading. Within a sector's expression table, rows are
ordered best-to-buy: technical state first (CONFIRMED → LAGGING →
STRETCHED → WARMING_UP → PARENT_INACTIVE → BROKEN → NO_DATA), higher
theme sentiment as the tiebreaker.

**Reading the table together.** A sector showing NEW_BUY on the Dashboard, with most of its expressions CONFIRMED, is the cleanest setup. NEW_BUY but everything STRETCHED means the move is real but the entry timing is bad — wait. NEW_BUY with one CONFIRMED and the rest BROKEN means the sector signal is being driven by a narrow leadership; you may want that one CONFIRMED ETF, not a diversified basket.

### ✨ Trend

Time-series view of sentiment. Per-sector line chart over the last `sentiment_lookback_days` window, plus a sectors × weeks heatmap (red-yellow-green). Useful for spotting sentiment regime shifts before they show up in the convergence test — e.g. a sector that has been BUY for 6 weeks but whose sentiment score has been monotonically declining is a HOLD_IF_LONG candidate the model hasn't yet flagged.

### 📧 Inbox

Pulls unread Gmail matching `GMAIL_FILTER_ADDRESS` via the Gmail REST
API (OAuth 2.0 — see `scripts/gmail_oauth_setup.py` and `SETUP.md`),
enriches each newsletter by:
1. Extracting whitelisted outbound links and PDF attachments,
2. Fetching their text content,
3. Pushing the assembled context through `gpt-4o-mini` for structured sector scoring.

Each successful ingest stamps the Gmail Message-ID into the DB, so re-running on the same inbox is a no-op. See `src/nlp_pipeline.py` for the prompt and schema.

### 📥 Ingest Newsletter

Manual paste-text entry point for newsletters that didn't arrive by email — same scoring pipeline as Inbox.

### 🗂 History

Browse past ingests. Each row is one newsletter: source, date, scored sectors with confidence, and the extracted-and-summarized text the model saw. Deletable if you want to remove a polluting source from the rolling sentiment window.

### 🧪 Backtest

Controls up top: cost per side (bps), slippage per side (bps), execution
lag (`next_open` fills at the next session's open — the default and the
closest match to how the live orders panel behaves; `same_close` fills
at the signal date's own close), and trade policy (`event_driven`, which
matches the live model — buy on transition into BUY-class, sell on
transition out, no intra-week rebalancing; or `rebalance_to_target`,
which drags every held name back to its target weight every week — more
turnover, more cost).

Leads with the current regime badge and a rotation verdict (how many
historical SPY drawdowns ≥5% the strategy lost less on, with mean
excess), plus up/down-capture ratios per regime. The raw CAGR-vs-SPY
comparison is demoted to an expander — a defensive rotation strategy
giving up some bull-market upside is the expected trade-off, not a
failure, so it shouldn't be the headline.

Further expanders: a full **drawdown-by-drawdown breakdown** (what was
held at the peak vs. the trough, what rotated in/out during the
drawdown, for every qualifying SPY drawdown in the window); the
**per-regime cumulative-return table**; **costs & turnover**; a
**real-sentiment ablation** over the (tiny) real-data window, always
shown with its own caveat text; and a **signal quality — information
coefficient** table (cross-sectional rank correlation between conviction
and forward return, by horizon — `n_periods` matters more than the
headline number this early). A **trade log CSV** download sits at the
bottom. See `BACKTEST_REPORT.md` for the full, regenerable methodology
writeup, and `WALK_FORWARD_REPORT.md` for the out-of-sample parameter
validation behind the current defaults (including `chase_weight_fraction`).

### 🚀 Breakouts

A second, **independent** signal system — deliberately NOT sentiment-
gated, because `src/signals.py::build_signals` can never emit BUY
without newsletter sentiment ≥ `buy_sentiment_threshold`, and there is
no newsletter coverage at all for most of the universe this tab covers.
Nothing here touches `build_signals`, `refine_signals`,
`target_weights`, or `signal_snapshots`.

**Universe.** Bitcoin / Ethereum / Solana (spot tickers + ETF wrappers),
crypto equities & infrastructure (COIN, BITQ, BKCH, WGMI), Gold, Silver,
broad Metals & Mining, and Copper Mining (`config/expanded_universe.py`)
— **plus** the plain-proxy ETF for each of the 11 core sectors and UFO
(the same tickers the sector model watches, via a second, independent
lens).

**The signal** (`src/breakout_signals.py`), per ticker:

1. **Consolidation/breakout** (`src/consolidation.py`) — a tight-range
   compression detector: by default, ATR% at or below 75% of its own
   1-year average, OR Bollinger-Band-width in the bottom quartile of its
   own 1-year history (`compression_require` can tighten this to require
   both legs, or pin to one). Held for at least `consolidation_min_days`
   (60). A breakout is a close above that range's high, and stays
   flagged "active" for a trailing `breakout_lookback_days` window (10)
   rather than vanishing the day after it fires — a weekly-reviewed
   dashboard needs that.
2. **Trend regime** — SMA50 / SMA150 slope (Rising / Flat / Falling)
   plus price vs SMA150. A falling SMA150, or price below it, is a hard
   **RISK_EXIT** that overrides any breakout signal that would
   otherwise fire, however clean it looks.
3. **Money flow** (`src/money_flow.py`) — Chaikin Money Flow + OBV
   trend slope (volume-normalized, so a thin name and a mega-cap ETF in
   the same pattern read the same), classified Strong Accumulation →
   Strong Distribution.

| Signal | Meaning |
|---|---|
| 🟢 **HIGH_CONVICTION_BUY** | Clean breakout + SMA50 rising + price above a rising SMA150 (full regime confirmation) |
| 🟢 **BUY** | Clean breakout + SMA50 rising, SMA150 regime not yet confirmed |
| 🔭 **WATCH** | Consolidating (≥60 days) but no breakout yet |
| ⚪ **NO_SIGNAL** | No qualifying setup right now |
| 🔴 **RISK_EXIT** | Price below a falling SMA150 — overrides any breakout |
| ⚫ **NOT_ENOUGH_DATA** | Fewer bars stored than `sma_long + slope_lookback + 5` needs |

**Reading the tab.** A heatmap up top (money-flow score, trend-regime
score, conviction, all centred at 0) gives a one-glance scan across the
whole universe. Below it, tickers are grouped (Bitcoin, Ethereum,
Solana, Crypto Equities & Infrastructure, Gold, Silver, Metals & Mining,
Copper Mining, then one group per SPDR sector plus UFO), each row
showing the signal badge, an **execution route** badge (🏦 Brokerage
Ticker for everything traded as a normal equity/ETF, 🔑 Direct Spot
Wallet for BTC-USD / ETH-USD / SOL-USD), the top reasons, and any risk
flags. A 0–5 conviction score sums: consolidation detected, breakout
active, SMA50 rising, price above SMA150, and accumulation-class money
flow.

**In-tab backtest.** The **"Run backtest"** button runs a separate,
simpler weekly-marked backtest (`src/breakout_backtest.py`) across the
whole expanded universe — equal-weight the top `max_positions` (default
8) BUY-class names, ranked by conviction with money-flow state as
tiebreaker, 5bps/side costs, mark-to-market at each weekly rebalance.
This is a validation harness for the signal in isolation, not a
production simulator (see the module docstring for what it deliberately
doesn't do, e.g. daily marking).

**Price data.** This tab shares `prices.db` with the rest of the app.
The in-app **🔄 Update price data** button (Price Action / Expressions
tabs) pulls the expanded universe too; the standalone CLI
(`scripts/update_prices.py`) currently does **not** — it only refreshes
the sector + expression universe. Use the in-app button, or extend the
CLI, to keep the Breakouts universe fresh via cron.

---

## 3. Worked examples

### Example A — Clean BUY

> Dashboard: **XLB — NEW_BUY**. State reason: "fresh BUY (week 2); ext +6.4% vs SMA200 (cutoff 12%)".
> Expressions tab → XLB expander:
> - XLB → **CONFIRMED**
> - XME → **CONFIRMED**
> - GDX → **STRETCHED** (extension +24% > cutoff 24% × β 2.0)
> - GDXJ → **CONFIRMED**

**Reading:** Materials is a fresh BUY with the sector itself and most miners participating. GDX is overextended at its own level — wait for a pullback before adding the senior miner, but GDXJ and the plain XLB / XME are fair game today. If your conviction is the gold-miner thesis specifically, GDXJ over GDX here.

### Example B — Mature trend, narrow leadership

> Dashboard: **XLK — HOLD_IF_LONG**. State reason: "BUY for 6 consecutive weeks (cutoff 4) — hold if you own it, do not add fresh".
> Expressions tab → XLK expander:
> - XLK → STRETCHED
> - SOXX, SMH → STRETCHED
> - IGV, WCLD → BROKEN
> - HACK, SKYY, BOTZ → LAGGING

**Reading:** The sector signal is mature and now narrowly led by semis, which are themselves extended. Software and cloud are already broken at the ETF level. Don't enter fresh. If you already own XLK or a semi ETF, hold; if you don't, sit it out — there is no clean expression to enter.

### Example C — Macro disagrees with the model

> Dashboard: **XLF — NEW_BUY** (RS positive, above SMA, sentiment +3.5).
> Macro tab: T10Y−T2Y = −0.3% (🔴 Inverted), HY OAS rising past 5%.

**Reading:** The model has done its job — the test passes. But the macro tab is telling you bank earnings beta to a steepener isn't there, and credit is wobbling. If the net macro lean is strong enough (§1, macro override layer), the model itself will already have downgraded this to `HOLD` or `REDUCE` rather than leaving it at `NEW_BUY` — a `NEW_BUY` that survives alongside a clearly hostile macro tape means the net lean wasn't strong enough to trip `macro_strong_count`. Either way, use the macro tab as a "size smaller than equal-weight" signal, or wait until at least one macro dial improves.

### Example D — WATCH: macro leads, price hasn't confirmed yet

> Dashboard: **XLI — WATCH**. State reason: "sentiment+macro support (2 tailwinds vs 0 headwinds), price not yet confirmed — watch for RS turn".

**Reading:** XLI is above its own SMA200 but hasn't cleared the RS/sentiment bar for a raw BUY, while the macro tape is leaning supportive enough to trip the override. This is a visibility flag, not a position — no capital is allocated. The concrete trigger to watch for is 3-month RS turning positive vs SPY; if that happens next week while macro support holds, expect a promotion to NEW_BUY.

### Example E — Breakout scanner: RISK_EXIT beats a clean-looking breakout

> Breakouts tab: **XYZ — RISK_EXIT**. Reasons prefixed "(prior setup)":
> tight 74-day consolidation, broke out 3 days ago, SMA50 rising.
> Risk flags: "Price $41.20 is below SMA150 ($44.80)".

**Reading:** the short-term setup (consolidation + breakout + rising SMA50) is real — but the ticker is still in a longer-term downtrend (price below a falling or barely-rising SMA150), and that overrides everything else. This is deliberate: a clean short-