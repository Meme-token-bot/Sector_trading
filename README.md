# Sector Rotation — Macro-Filtered Convergence Model (+ Independent Breakout Scanner)

A systematic, macro-filtered ETF sector rotation dashboard, plus a second,
independent momentum/breakout scanner. The core model tells you **which
of the 11 US SPDR sector ETFs — plus a 12th, manually-sized Space sector
(UFO) — to buy, hold, or sell this week**, and why, by combining
quantitative price data (SMA200, 3-month relative strength vs SPY),
expert sentiment (LLM-extracted from newsletters you ingest), and the
live macro tape (FRED + yfinance). A separate **🚀 Breakouts** tab runs a
price/volume-only consolidation-breakout signal — no sentiment gate —
across a wider universe: Bitcoin, Ethereum, Solana, crypto equities,
gold, silver, broad/copper miners, and the plain proxy ETF for each core
sector, for setups the sentiment-gated model structurally can't reach
(there's no newsletter coverage for most of that universe). You place
trades manually — via the Tiger Brokers integration where configured, or
by hand from the tables the dashboard produces.

**On "beating SPY":** the core rotation model is deliberately defensive,
not return-maximizing. In a straight bull market it's *expected* to trail
SPY's raw return (see the **🧪 Backtest** tab) in exchange for smaller
drawdowns when the market breaks — that trade-off is measured, not
assumed, and `BACKTEST_REPORT.md` / `WALK_FORWARD_REPORT.md` are
generated to show the honest numbers, small samples and all. If raw
outperformance is the goal, see "Research & experimentation" under
**CLI scripts** below for the alternative, more concentrated strategies
being tested against this baseline.

---

## Reading the dashboard

### The state legend (memorise this — it's everything)

Every sector lands in one of **seven** states each week. The dashboard tints
rows by state across the matrix, the Tiger drift table, and the orders
panel so the colour always means the same thing.

| State | Colour | Meaning | What to do |
|---|---|---|---|
| **NEW_BUY** | 🟢 green | Fresh BUY trigger this week — price above SMA200, beating SPY by 3M, sentiment ≥ +2 | Open a position |
| **HOLD_IF_LONG** | 🟡 amber | Trend mature (≥ `stale_buy_weeks` of BUY in a row) | Keep if already held, but don't initiate |
| **CHASE** | 🟠 orange | Price > SMA200 by more than `extension_pct_cutoff` (×β for expressions) | Full entry zone is gone, but by **default the model still takes a partial position** — `chase_weight_fraction` (25%) of a full size, funded from the cash buffer. Set it to 0 in `config/settings.py` to fully sit out CHASE instead |
| **REDUCE** | 🟤 rust | Was BUY, now degraded (sentiment slipped, RS faded), **or** a stale BUY facing a strong macro headwind | Trim |
| **WATCH** | 🔭 teal | Not a BUY yet — price hasn't confirmed — but sentiment and the macro tape both support it (net macro tailwinds ≥ `macro_strong_count`) | No position — watch for the RS turn / SMA200 reclaim |
| **HOLD** | grey | Nothing convergent — no edge either way, **or** a would-be fresh BUY that a strong macro headwind vetoed | Do nothing |
| **SELL** | 🔴 red | Price < SMA200, OR bottom-3 RS rank, OR sentiment ≤ −3 | Exit |

The orders panel generates rows for **NEW_BUY** (BUY), **SELL / REDUCE**
(sell/trim), and — now that the walk-forward sweep validated a partial
sleeve — **CHASE** too, sized at `chase_weight_fraction` of a full
position and explicitly labelled `CHASE (25% partial)` so it's never
mistaken for a full-size entry. `HOLD_IF_LONG` is still omitted from the
orders panel: it means "hold if you already own it, do not add fresh," so
there's never an order to place for it either way. **WATCH** is a
visibility flag, not a position — it never receives target-weight capital.

---

### 📈 Dashboard — the only tab you need to ship orders

**Top: Decision Cockpit.** Four blocks: current SPY regime
(BULL/CORRECTION/BEAR) and how many days it's held; cross-sectional
breadth (how many of the 11 core sectors are above their own SMA200); RS
dispersion (how spread out relative strength is right now — low
dispersion means sectors are moving together and rotation has less to
work with); and, once `scripts/run_walk_forward.py` has been run at least
once, a one-line walk-forward trust badge showing whether current
defaults are still the validated choice. (The trailing hit-rate figure
lives in its own caption further down the page, not in this strip.)

**⚠️ Portfolio Risk** (expander). Correlation, concentration, and VaR on
**this week's target weights** (not necessarily your current Tiger
holdings): average pairwise weekly-return correlation with a correlation
heatmap (52-week lookback), an "effective # of bets" figure (naive
1/HHI alongside a correlation-adjusted version), and 1-week historical
VaR/ES at 95% confidence (104-week lookback, empirical). Explicitly a
lightweight stopgap, not a factor-risk system — good for catching "four
sectors, one bet" at a glance.

**✅ Pre-Trade Checklist** (expander). The same checks as
`scripts/preflight.py` — both read `src/preflight_checks.py`, so the CLI
and the in-app panel can't silently disagree: data freshness, current
model state, the Tiger connection + cash-coverage check, and the Monday
order list, rolled into one 🟢 READY / 🟡 READY WITH WARNINGS / 🔴 NOT
READY badge.

**This Week's Orders.** One row per actionable trade — **NEW_BUY** (green
BUY), **CHASE** whenever `chase_weight_fraction > 0` (orange, labelled
`CHASE (25% partial)`), and **SELL / REDUCE** for currently-held sectors.
When Tiger isn't connected (or the snapshot fetch fails), every sector is
treated as "potentially held" so SELL/REDUCE rows still render, just
without a dollar size. SELL/REDUCE size comes from your current Tiger
holding value; BUY/CHASE size is `target_weight × Tiger NLV`. The
*Vehicle* column picks the top **CONFIRMED** expression ETF for the
sector (falling back to the sector ETF itself) — check the
**Expressions** tab's Note column for any thin-AUM/low-liquidity caution
on that specific ticker before sizing a real order. *Why* shows the
`state_reason` plus a 5-dot conviction badge. Empty state = "portfolio
aligned, no actions this week."

**State-change strip.** A one-line `st.info` summarising any sectors
that flipped state since the last snapshot, e.g.
`XLE: CHASE → NEW_BUY (RS turned positive)  ·  XLF: HOLD_IF_LONG → REDUCE (sentiment fell)`.
Hidden entirely when nothing changed.

**Sector Relative Strength Matrix.** The full table of all 11 (12 with
UFO) sectors with the following columns:

| Column | What it tells you |
|---|---|
| **3M vs SPY** | Forward-looking edge — positive = sector is leading the index |
| **Ext vs SMA** | How far above (or below) the 200-day moving average — large positive = stretched, large negative = broken |
| **Wks BUY** | Consecutive weeks this sector has been BUY. ≥ `stale_buy_weeks` = trend is mature, expect HOLD_IF_LONG |
| **Conviction** | 0–5 dot scale (●●●○○). +1 each for: RS>0, RS>strong margin, sentiment ≥ buy_threshold+1, ≥2 weeks BUY. The macro component is SYMMETRIC — a clear net tailwind adds a point, a clear net headwind SUBTRACTS one (floored at 0) |
| **Sentiment** | `+2.1 · n=3 · σ=0.8` — mean newsletter score · how many newsletters covered this sector · stdev across them. High σ = newsletters disagree, take the mean with a grain of salt. A ⚠️ suffix flags a *consensus peak* (n≥3, σ<0.5, score≥+4) — unusually uniform bullishness read as a contrarian caution, not a stronger buy signal |
| **State** | The seven-state classification (see legend above), row-tinted |
| **Macro** | `5/7 ✓` style pill = tailwind indicators / (tailwinds + headwinds). Neutral indicators excluded from the denominator, so denominators differ sector to sector — the color tint, not the raw fraction, is the fair comparison. Green ≥ 0.625, amber 0.375–0.625, red < 0.375 |
| **Why** | For BUY-class rows: `📈 RS%  📊 Ext%  💬 Sentiment` triplet. For others: the prose `state_reason` |

**Performance feedback strip.** Below the state-distribution counters:
`{source} signals, last 12 weeks (hold-to-state-exit, median hold N
days): hit rate X% (95% CI L–H%), mean excess return +Y% vs SPY (n=Z)`.
`{source}` reads "NEW_BUY" when it's drawing on persisted
`signal_snapshots`, or "BUY/NEW_BUY (raw replay)" when it's falling back
to a raw historical replay because no snapshots exist yet — the label
tells you which. Shows "Performance stats unavailable" when you have
fewer than 4 weeks of history.

**📉 Rolling edge trend** and **🎯 Conviction calibration** (expanders).
The first plots the trailing-12-week hit-rate/mean-excess as a time
series across many historical evaluation points, with a 95% Wilson
confidence band, so you can see whether the edge is currently working or
decaying, not just its current snapshot. The second buckets hit-rate and
mean-excess by the 0–5 conviction score a signal carried *at entry* —
the check for whether the dots actually mean anything. Both read
exclusively from persisted `signal_snapshots`, so they fill in slowly
and are empty on a fresh install.

**Sector Rotation Phase Space (Relative Rotation Graph).** Plots each
sector's 3-month RS vs SPY against its week-over-week change in RS, with
a fading 26-week trail per sector (optional 3-week-EMA smoothing).
Leading / Weakening / Improving / Lagging quadrants; larger markers for
NEW_BUY/HOLD_IF_LONG; SELL renders as ✕. Click any point to isolate that
sector's trail, click again to reset.

**Right pane: Tiger Drift.** Per-sector comparison of `target_weight`
vs `current_weight` from your Tiger account. Includes the holding state
(tinted same as the matrix), a `Stop at (SMA200)` column showing
`$current → $stop (-X%)` so you know exactly where the exit lives, and
SELL rows sorted to the top because exits are time-sensitive.

---

### 📰 Weekly Recap — plain-language narrative, persisted

A one-shot OpenAI synthesis of the last 7 days of newsletters + the
current macro tape into, in reading order:

- **Executive summary** (6–10 sentences — the lede, written last so it
  can legitimately reference everything else below)
- **Macro narrative** + dominant themes + contradictions
- **Per-sector recaps** ordered by suggested tilt
- **🔭 Sectors to watch** — 2–5 *forward-looking* flags, distinct from
  the allocation table (which is "what to own now"): `building` =
  sentiment/macro support a sector but price hasn't confirmed yet;
  `rolling over` = currently strong but extended or losing macro
  support. Each cites a concrete signal-row fact plus a newsletter/macro
  reading, and names the specific trigger to watch for
- **Allocation tilts** (Overweight / Equal-weight / Underweight / Avoid)

Recaps are **saved to SQLite** keyed by `(date, model)`. Reopening the
tab or picking a past date from the **Recap date** dropdown loads the
stored payload with zero OpenAI cost. "Generate / load recap" hits the
cache; "Force regenerate" deletes the stored row and calls OpenAI again
(useful if you've ingested more newsletters since the last run).

Requires `OPENAI_API_KEY` in `.env`. Optionally set
`WEEKLY_RECAP_MODEL=gpt-4o` in `.env` to upgrade just this tab to a
stronger model — other tabs keep using `OPENAI_MODEL`.

---

### 🌐 Macro — the regime tape

Fifteen macro indicators, grouped the way the tab renders them, each
shown as: current level + 1-year z-score or 30-day slope → regime badge
(🟢/🟡/🟠/🔴) → one-line sector-rotation implication → 1-year line chart.

| Group | Indicators |
|---|---|
| 🛡️ Risk / Vol | VIX, HY OAS, Gold/Oil, BBB / IG OAS, Financial Conditions (NFCI) |
| 📈 Growth / Cycle | Copper/Gold (z-score banded), DXY, Initial jobless claims (4-wk avg, z-score banded) |
| 💵 Rates / Inflation | 10Y−2Y, 10Y nominal, 10Y real (TIPS), 5Y5Y breakeven, 2Y nominal, 10Y breakeven, 30Y mortgage (+ spread vs 10Y) |

The bands are **regime guideposts, not trade triggers** — they tell you
what kind of market you're in so you can sanity-check the sector signal:
a model saying "BUY XLF" while T10Y2Y is deeply inverted and HY OAS is
widening deserves a second look. This full panel also feeds
`src/macro_alignment.py`'s per-sector tailwind/headwind scoring — the
**Macro** pill on the Dashboard matrix and the macro veto/override pass —
so a reading here is doing real work upstream, not just decorating this
tab.

### 📉 Price Action — visual context

A full-featured interactive chart (candles, SMA50/200, optional
RSI/MACD/Bollinger overlays, optional SPY-overlay comparison line,
selectable ticker/timeframe/lookback) — defaults to your freshest
NEW_BUY sector, or XLK if nothing currently qualifies. Below it, a
clickable grid of mini-charts for all 11 sectors on the daily timeframe,
tinted by state; click any tile to load it into the main chart.
**Update price data** pulls incremental bars for the full universe —
signal sectors + benchmark + all expression tickers + the expanded
crypto/metals universe the **Breakouts** tab uses — and runs a
split-detection guard.

### 🎯 Expressions — narrower vehicles per sector

For each sector, candidate expression ETFs from the curated whitelist in
`config/expressions.py` (e.g. defence picks for XLI, biotech for XLV) —
including an auto-appended iShares alternate for each of the 11 core
sectors (IYW for tech, IYF for financials, …). Each row shows:

- **Self-check state**: CONFIRMED / LAGGING / STRETCHED / BROKEN /
  WARMING_UP / PARENT_INACTIVE / NO_DATA — the expression's own
  technical posture, independent of the parent sector
- **Band**: `$28.40 → $34.12` — the BROKEN floor (SMA200) → the
  STRETCHED ceiling (SMA200 × (1 + cutoff% × β)). The healthy entry
  zone lives between these two prices
- **Theme news** (where applicable): expressions mapped to a sub-sector
  theme in `config/themes.py` (semis, uranium, biotech, regional banks,
  …) also show a blended sentiment reading — newsletter theme-tags plus
  automated Google-News-headline scoring, refreshed with one batched
  OpenAI call via the tab's **🔄 Refresh theme news** button — and flag
  **NEWS_CONTRADICTS** when price is healthy but the news is clearly
  negative, or **NEWS_DIVERGENCE** when price is BROKEN but the news is
  clearly positive. Plain sector proxies carry no theme and no flag.

Only CONFIRMED expressions are offered as the Dashboard orders panel's
*Vehicle* pick (ties broken by whichever has the better theme-news
read); everything else still shows in this tab for context.

### ✨ Trend — sentiment over time

Weekly snapshots of the rolling-window aggregate sentiment for every
sector. Useful for spotting newsletter-sentiment momentum (or fading)
before it shows up in the weekly signal. Window length is
`sentiment_lookback_days` from `config/settings.py`.

### 📧 Inbox — Gmail auto-ingest

Pull whitelisted senders straight from Gmail via the OAuth 2.0 REST API
(`scripts/gmail_oauth_setup.py` mints a reusable, silently-refreshed
token), parse each newsletter through the LLM pipeline, and save to
SQLite. Idempotent on Gmail message ID, so re-runs only fetch new mail.

### 📥 Ingest Newsletter — manual paste

Drop in newsletter text from a paywalled or non-email source. Same
parser as the Gmail path. Idempotent on SHA256 of `(author, date,
content)` — re-ingesting the same text is a no-op.

### 🗂 History — what's in the database

Browse and (carefully) delete past newsletters. Deletion cascades to
the per-sector ratings.

### 🧪 Backtest — is the rotation thesis working

Controls: cost bps, slippage bps, execution lag (`next_open` default, or
`same_close`), trade policy (`event_driven`, matching what the live
orders panel does, or `rebalance_to_target`, more turnover). Leads with
the current regime badge, the rotation verdict (how many historical SPY
drawdowns ≥5% the strategy lost less on, with mean excess), and
up/down-capture ratios per regime — the metrics that actually matter for
a defensive rotation strategy. The raw CAGR-vs-SPY comparison is demoted
to an expander further down, since a rotation strategy giving up some
bull-market upside is the expected trade-off, not a failure. Further
expanders: a full drawdown-by-drawdown breakdown (what was held at the
peak vs. the trough, what rotated during it), the per-regime cumulative-
return table, costs & turnover, a real-sentiment ablation (tiny sample,
always caveated), a signal-quality information-coefficient table, and a
trade-log CSV download. See `BACKTEST_REPORT.md` for the full,
regenerable methodology writeup, and `WALK_FORWARD_REPORT.md` for the
out-of-sample parameter validation behind the current defaults.

### 🚀 Breakouts — an independent, non-sentiment-gated momentum scanner

A second signal system, deliberately separate from the sector
convergence model above. `src/signals.py::build_signals` can never emit
BUY without newsletter sentiment ≥ +2 — fine for the 11 SPDR sectors
where you're actively ingesting coverage, useless for an asset with no
newsletter coverage at all. This tab runs a **price/volume-only** signal
instead, across a wider universe:

- **Bitcoin, Ethereum, Solana** (spot tickers + ETF wrappers)
- **Crypto equities & infrastructure** (COIN, BITQ, BKCH, WGMI)
- **Gold, Silver, broad Metals & Mining, Copper Mining**
- **The plain-proxy ETF for each of the 11 core sectors and UFO** — the
  same tickers the sector model watches, scored a second, independent way

For each ticker: a consolidation/breakout detector (tight-range
compression via ATR% and/or Bollinger-Band-width percentile, then a
close-above-the-range breakout that stays "active" for a configurable
trailing window instead of vanishing the day after it fires), a
trend-regime filter (SMA50/SMA150 slope + price vs SMA150 — a falling
SMA150 is a hard **RISK_EXIT** that overrides any breakout, however
clean it looks), and a money-flow read (Chaikin Money Flow + OBV trend
slope, classified Strong Accumulation → Strong Distribution).

| Signal | Meaning |
|---|---|
| 🟢 **HIGH_CONVICTION_BUY** | Clean breakout, SMA50 rising, AND price above a rising SMA150 |
| 🟢 **BUY** | Clean breakout + SMA50 rising, SMA150 regime not yet confirmed |
| 🔭 **WATCH** | Consolidating (≥60 days) but no breakout yet |
| ⚪ **NO_SIGNAL** | No qualifying setup right now |
| 🔴 **RISK_EXIT** | Price below a falling SMA150 — overrides any breakout |
| ⚫ **NOT_ENOUGH_DATA** | Fewer bars stored than the SMA150 + slope lookback needs |

A 0–5 conviction score and an execution-route badge (🏦 Brokerage Ticker
vs 🔑 Direct Spot Wallet for BTC-USD/ETH-USD/SOL-USD) accompany each row.
An in-tab **"Run backtest"** button runs a separate, simpler
weekly-marked backtest (`src/breakout_backtest.py`) over the whole
expanded universe for a quick sanity check on this signal alone.

This tab shares `prices.db` with the rest of the dashboard — the
**🔄 Update price data** button (Price Action / Expressions tabs) pulls
the expanded universe too. The standalone CLI
(`scripts/update_prices.py`) does **not** — it only refreshes the sector
+ expression universe, so if you rely on it for cron-style updates, add
a scheduled call that also touches the expanded universe, or use the
in-app button.

---

## Weekly workflow

| When | Action |
|---|---|
| **Friday / Saturday** | Open **📧 Inbox** to auto-ingest newsletters as they arrive, or paste into **📥 Ingest Newsletter** for paywalled sources |
| **Sunday** | Open **📈 Dashboard** → forces fresh price + sentiment compute. Start with the Decision Cockpit (regime / breadth / RS dispersion / walk-forward badge), check **✅ Pre-Trade Checklist** for any FAIL/WARN, then skim the orders panel, the state-change strip, and the matrix. Open **📰 Weekly Recap** for narrative context, and **🚀 Breakouts** for a second, non-sentiment-gated read on the same tickers plus crypto/metals |
| **Monday morning** | Read the right-pane Tiger Drift table (or the Orders panel if Tiger isn't connected), place orders manually. SELL/REDUCE rows first, then BUY and partial-CHASE rows |

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # edit .env — at minimum set OPENAI_API_KEY
python -c "from src.db import init_db; init_db()"
streamlit run app.py
```

`FRED_API_KEY` (optional but recommended): `src/market_engine.py` tries
the authenticated FRED JSON API first and transparently falls back to
the anonymous CSV endpoint if the key is missing or the call fails, so
macro data still works without one — just with tighter rate limits.

CLI scripts under `scripts/` need the project root on `PYTHONPATH`:

```bash
PYTHONPATH=. python scripts/refresh_weekly.py
PYTHONPATH=. python scripts/ingest_newsletter.py --file path/to/newsletter.txt
```

---

## Architecture

```
config/settings.py           Universe, signal params, breakout params, env vars
config/expressions.py        Sector → curated expression ETFs (+ iShares alternates)
config/expanded_universe.py  Crypto / metals universe for the Breakout scanner
config/themes.py             Sub-sector theme taxonomy (newsletters/news → expression tickers)
config/whitelist.py          Domain whitelist for following newsletter links/PDFs
src/schemas.py                Pydantic models — also drive OpenAI Structured Outputs
src/db.py                     SQLite: sentiment, weekly