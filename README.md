# Sector Rotation — Macro-Filtered Convergence Model

A systematic, macro-filtered ETF sector rotation dashboard. Aggregates
quantitative price data (SMA200, 3M relative strength vs SPY) and
qualitative expert sentiment (LLM-extracted from newsletters) to generate
weekly BUY / HOLD / SELL signals on the 11 US SPDR Select Sector ETFs.
The user executes trades manually via Tiger Brokers on Monday mornings.

## Architecture

```
config/settings.py       Universe, signal params, env vars
config/expressions.py    Sector → curated expression ETFs (Phase 2)
src/schemas.py           Pydantic models — also drive OpenAI Structured Outputs
src/db.py                SQLite store for parsed newsletter sentiment
src/nlp_pipeline.py      Newsletter → NewsletterAnalysis → DB
src/market_engine.py     yfinance prices, FRED yield curve, gold/oil
src/signals.py           Pure-function convergence decision matrix
src/tiger_client.py      Tiger positions + sector-aware drift
app.py                   Streamlit dashboard
scripts/                 CLI helpers (refresh, ingest)
data/sentiment.db        SQLite (auto-created)
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — at minimum set OPENAI_API_KEY
python -c "from src.db import init_db; init_db()"
streamlit run app.py
```

The CLI scripts under `scripts/` need the project root on `PYTHONPATH`:

```bash
PYTHONPATH=. python scripts/refresh_weekly.py
PYTHONPATH=. python scripts/ingest_newsletter.py --file path/to/newsletter.txt
```

## Weekly Workflow

| Day                 | Action                                                      |
|---------------------|-------------------------------------------------------------|
| Friday / Saturday   | Paste newsletters into the Ingest tab as they arrive        |
| Sunday              | Open Dashboard tab → forces fresh price + sentiment compute |
| Monday morning      | Read Drift table, place orders manually in Tiger app        |

## Design Notes

- **Caching TTLs**: prices 6h, Tiger snapshot 5min, sentiment aggregate 10min.
  Override with the *Force refresh all caches* button.
- **Idempotency**: each newsletter is keyed by SHA256(content+author+date).
  Re-ingesting the same text is a no-op.
- **Pure signal logic**: `src/signals.py` has no IO, no globals, no caching.
  Same inputs always give same outputs — preconditioned for backtesting.
- **Lazy Tiger import**: `tigeropen` is imported inside functions only, so
  the dashboard runs without the SDK installed.
- **Convergence rules** (raw signal): BUY needs price > SMA200 *and* RS > 0
  *and* sentiment ≥ +2. SELL fires on any of: price < SMA200, bottom-3 RS rank,
  or sentiment ≤ −3. Anything else is HOLD.
- **State classification** (refined signal — late-entry guard): the raw BUY/HOLD/SELL
  is then enriched with weekly history + extension data into one of:
  `NEW_BUY` (fresh entry OK), `HOLD_IF_LONG` (trend mature, don't chase),
  `CHASE` (price too far above SMA200), `REDUCE` (was BUY, now degraded),
  `HOLD`, `SELL`. Thresholds in `config/settings.py` (`extension_pct_cutoff`,
  `stale_buy_weeks`, `history_weeks`).
- **No leveraged ETFs**: expression list (Phase 2) is plain equity only.
  Operating leverage from underlying business cost structures, not derivatives.
