# Sector Rotation Dashboard — Reader's Guide

A walkthrough of every tab, every column, and how the signals are actually computed. Read top-to-bottom the first time; after that use it as a reference.

> **TL;DR.** The model watches the 11 SPDR Select Sector ETFs (XLK, XLY, XLC, XLF, XLI, XLB, XLE, XLV, XLP, XLU, XLRE) against SPY. Each week it asks three questions per sector — *is it above its own SMA200, is it outperforming SPY, and is the newsletter sentiment positive?* — and converges those into one of six **states**. When a state says BUY, the **Expressions** tab tells you which specific ETF to actually buy, and a second-tier **self-check** tells you whether that expression is participating, lagging, broken, or overextended at its own level.

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

### The refined verdict — six states

`src/signals.py::refine_signals` then layers two practitioner concerns on top of the raw verdict:

1. **Late-entry guard.** A sector that's already 12%+ above its own SMA200 is over-extended; entering fresh from cash here is chasing.
2. **Trend maturity.** A sector that's been BUY for 4+ consecutive weekly snapshots is a mature trend — if you missed the entry, don't chase it now.

These promote/demote the raw signal into one of six **states**:

| State | Color | Means |
|---|---|---|
| 🟢 **NEW_BUY** | green | Raw BUY, not extended, fresh (< 4 consecutive BUY weeks). **Fresh entry OK.** |
| 🟡 **HOLD_IF_LONG** | amber | Raw BUY, but BUY for ≥ 4 weeks. **Hold if owned, don't add. Don't enter from cash.** |
| 🟠 **CHASE** | orange | Raw BUY but extension > 12%. **Too late; wait for a pullback to SMA200.** |
| 🟤 **REDUCE** | rust | Was BUY in the last 12 weeks but no longer qualifies. **Trim if owned.** |
| ⚪ **HOLD** | neutral | Doesn't qualify and hasn't recently. **Wait.** |
| 🔴 **SELL** | red | Failed a hard SELL rule. **Exit.** |

The promotion logic uses historical signal replay (`src/signal_history.py`). Each week the same `build_signals` call is replayed on truncated price data so the model can answer "how many weeks in a row has this been BUY?" and "was this BUY at any point in the recent window?"

### Position sizing

`src/signals.py::target_weights` equal-weights across **NEW_BUY + HOLD_IF_LONG** with a 5% cash buffer. `CHASE` is deliberately excluded — the model says don't enter from cash on a parabolic sector. `HOLD_IF_LONG` *is* included in target weights, but the dashboard's caption is explicit: if you don't already own a HOLD_IF_LONG row, do not enter — the target weight is only for sizing what you'd hold if you already had the position.

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

---

## 2. Reading the tabs

### 📈 Dashboard

The main view. Two panes.

**Left — Sector Relative Strength Matrix.** One row per sector, sorted by 3-month relative strength.

| Column | Meaning |
|---|---|
| Sector | Sector name (XLK → Technology, etc.) |
| 3M vs SPY | 63-day return minus SPY's 63-day return |
| Ext vs SMA | (price − SMA200) / SMA200. Positive = above trend, negative = below |
| Wks BUY | Consecutive weekly snapshots (out of last 12) where the raw test passed |
| Sentiment | Mean sentiment score + n (count of newsletters with coverage in the 60-day window) |
| State | One of the six states above, row tinted by state color |
| Action | Plain-English reason for the state |

Below the table: state counts (how many sectors are in each state right now), and an expandable **target-weights** table showing the equal-weight allocation.

**Right — Tiger Portfolio Drift.** If Tiger SDK is configured, shows current vs target weights per sector and the trade size needed to close the drift. Holdings that don't map to any sector's expression list are surfaced separately so you know they aren't counted toward sector targets.

If Tiger isn't configured, a manual NLV input lets you preview the target-value table for a hypothetical portfolio size.

### 🌐 Macro

Eight macro indicators, each rendered as: current reading + 30-day slope or 1-year z-score → regime badge (🟢 / 🟡 / 🟠 / 🔴) → one-line sector-rotation implication → 1-year line chart.

| Indicator | What it tells you |
|---|---|
| VIX | Volatility regime (complacent / normal / stressed / crisis) |
| HY OAS | High-yield credit spread — risk-on vs credit stress |
| Gold/Oil | Defensive vs cyclical commodity pricing |
| Copper/Gold (z) | Reflation gauge — rising = pro-cyclical |
| DXY | Dollar regime — strong dollar pressures non-US and metals |
| T10Y−T2Y | Yield curve slope; inverted is a recession warning |
| UST10Y | Absolute level of long-end nominal rates |
| Real 10Y | TIPS real yield — financial-conditions tightness |

The bands are **regime guideposts, not trade triggers**. They tell you what kind of market you're in so you can sanity-check the sector signal: a model saying "BUY XLF" while T10Y2Y is deeply inverted and HY OAS is widening deserves a second look.

### 📉 Price Action

Candles + SMA50/200, optional RSI/MACD/Bollinger overlays. Optional SPY comparison line. Loads from the local prices DB (`data/prices.db`), which stores ~5y of daily and weekly bars per ticker.

**Update price data** button pulls incremental bars from yfinance for the full universe (signal sectors + benchmark + all expression tickers) and runs a split-detection guard: if any overlapping historical bar disagrees with the stored value by more than 0.5%, the ticker's history is wiped and re-pulled in full. See the module docstring of `src/price_store.py` for why this matters.

### 🎯 Expressions

This is where the sector signal turns into an actual trade. Each sector maps to a curated list of plain and operating-leverage ETFs in `config/expressions.py`. **Every expression is a plain equity ETF** — no daily-reset leveraged products. The "leverage" comes from the underlying businesses (e.g. gold miners' fixed costs amplify their earnings beta to gold price), not from derivatives.

For each sector you see:
- An expander, opened by default if the sector is BUY.
- A table with one row per candidate expression. Columns:

| Column | Meaning |
|---|---|
| Ticker | The expression ETF |
| Label | Plain-language name |
| Kind | `plain` / `thematic` / `operating leverage` |
| β hint | Rough 3-month price beta vs the parent sector ETF |
| 60d | Sparkline of the last 60 daily closes |
| **Self-check** | Per-expression participation state (see below) |
| **Self-check reason** | One-line explanation of the self-check |
| Note | Free-text guidance from `config/expressions.py` |

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

**Beta-scaled cutoff.** The same 12% extension cap used for sector-level CHASE is multiplied by the expression's `beta_hint` to define STRETCHED. A 1.0-beta plain ETF is STRETCHED at >12%; a 2.5-beta junior gold miner is STRETCHED at >30%. The rationale: a high-beta vehicle naturally moves further from its SMA in a strong sector trend, so applying the parent's cutoff unmodified would always paint it as overextended.

**State priority.** Evaluated in this order — first match wins:

1. `NO_DATA` (no bars stored)
2. `WARMING_UP` (< 200 bars stored)
3. `PARENT_INACTIVE` (parent state not in NEW_BUY / HOLD_IF_LONG)
4. `BROKEN` (below own SMA200)
5. `STRETCHED` (extension > beta-scaled cutoff)
6. `LAGGING` (rs_vs_parent < 0)
7. `CONFIRMED` (everything else, including rs_vs_parent == 0)

The parent ETF inside its own expression list (e.g. XLK in XLK's list) mechanically gets `rs_vs_parent == 0` and falls through to CONFIRMED — that's the intended baseline.

**Reading the table together.** A sector showing NEW_BUY on the Dashboard, with most of its expressions CONFIRMED, is the cleanest setup. NEW_BUY but everything STRETCHED means the move is real but the entry timing is bad — wait. NEW_BUY with one CONFIRMED and the rest BROKEN means the sector signal is being driven by a narrow leadership; you may want that one CONFIRMED ETF, not a diversified basket.

Below the table, a **View full chart for** selectbox renders a 6-month candle chart for any expression you pick. Same data path as the Price Action tab; the 300-day warmup window keeps SMAs accurate at the left edge of the visible window.

### ✨ Trend

Time-series view of sentiment. Per-sector line chart over the last `sentiment_lookback_days` window, plus a sectors × weeks heatmap (red-yellow-green). Useful for spotting sentiment regime shifts before they show up in the convergence test — e.g. a sector that has been BUY for 6 weeks but whose sentiment score has been monotonically declining is a HOLD_IF_LONG candidate the model hasn't yet flagged.

### 📧 Inbox

Pulls unread Gmail matching `GMAIL_FILTER_ADDRESS`, enriches each newsletter by:
1. Extracting whitelisted outbound links and PDF attachments,
2. Fetching their text content,
3. Pushing the assembled context through `gpt-4o-mini` for structured sector scoring.

Each successful ingest stamps the Gmail Message-ID into the DB, so re-running on the same inbox is a no-op. See `src/nlp_pipeline.py` for the prompt and schema.

### 📥 Ingest Newsletter

Manual paste-text entry point for newsletters that didn't arrive by email — same scoring pipeline as Inbox.

### 🗂 History

Browse past ingests. Each row is one newsletter: source, date, scored sectors with confidence, and the extracted-and-summarized text the model saw. Deletable if you want to remove a polluting source from the rolling sentiment window.

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

**Reading:** The model has done its job — the test passes. But the macro tab is telling you bank earnings beta to a steepener isn't there, and credit is wobbling. Use this as a "size smaller than equal-weight" signal, or wait until at least one macro dial improves. The model's confidence is sector-level; the macro tab is the override layer.

---

## 4. Operational notes

- **Caches.** Most expensive computations are cached for 5–360 minutes (see `@st.cache_data(ttl=...)` decorators in `app.py`). The `🔄 Force refresh all caches` button at the bottom of the Dashboard clears everything.
- **Update cadence.** The convergence test is designed for **weekly** decisions, not intraday rebalancing. The price DB updates incrementally; sentiment updates as you ingest newsletters.
- **Sentiment requires coverage.** A sector with no recent newsletter mentions has `sentiment_score = 0`, which is below the BUY threshold. The model is intentionally conservative — no coverage means no BUY.
- **Split-adjustments.** yfinance returns split-adjusted history. A split that happens today rewrites every prior close. `src/price_store.py` re-checks a 60-day overlap on every incremental update and wipes-and-refetches the full history if it detects > 0.5% drift on any overlapping bar. Do not optimize this away.
- **What this is not.** This is not a backtester, not an execution engine, and not a risk model. It produces sector-level convergence verdicts; humans (or a small Tiger script) execute the trades.

---

## 5. File map for the curious

| Concern | File |
|---|---|
| Parameters & universe | `config/settings.py`, `config/expressions.py`, `config/whitelist.py` |
| Price storage (SQLite + yfinance) | `src/price_store.py` |
| Quant metrics (SMA, RS, momentum) | `src/market_engine.py` |
| Sector signal (BUY/HOLD/SELL → 6 states) | `src/signals.py` |
| Historical state replay | `src/signal_history.py` |
| Per-expression self-check | `src/expression_signals.py` |
| Sentiment ingest pipeline | `src/gmail_client.py`, `src/content_extractor.py`, `src/nlp_pipeline.py`, `src/db.py` |
| Sentiment trend reconstruction | `src/trend.py` |
| Chart builders | `src/charts.py`, `src/indicators.py` |
| Tiger broker integration | `src/tiger_client.py` |
| Dashboard UI | `app.py` |
