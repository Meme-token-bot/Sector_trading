# Setup & Recommended Newsletter Seeds

## First-time setup

```bash
cd sector_rotation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then edit and set OPENAI_API_KEY
python -c "from src.db import init_db; init_db()"
streamlit run app.py
```

Tiger credentials are optional. The dashboard runs without them; the
drift panel falls back to a manual NLV entry.

## Seeding the sentiment database

The model needs enough sentiment coverage that the rolling 21-day window
contains at least 2-3 newsletters touching each sector. Suggested sources:

### Tier 1 — highest priority (free or cheap, deep sector coverage)
- **Lyn Alden — Strategic Investment Newsletter** (free, ~6 week cadence)
- **The Macro Compass — Alfonso Peccatiello** (free posts + paid)
- **Doomberg** (paid Substack, sharp energy/materials views)
- **SSGA Sector & Industry Insights** (free, maps directly to SPDR tickers)

### Tier 2 — daily / weekly macro briefs
- **Hedgeye Daily Market Brief**
- **Apollo Daily Spark** — Torsten Sløk (free, one chart per day)
- **BlackRock Investment Institute Weekly Commentary**

### Tier 3 — sell-side (if you have access)
- **Goldman Top of Mind**
- **JPM Cross-Asset Insights**

### Space sector seeds (new — supplementary 12th sector)
Space is a thematic overlay that cuts across XLI/XLK/XLC. These three
have enough cadence and ticker-specificity to score reliably:

- **Payload** (payloadspace.com) — daily, business-of-space, names tickers
- **Space.biz** by Ian Vorbach (spacedotbiz.substack.com) — weekly, investor angle, interviews Quilty Analytics
- **Cyclop SpaceTech** (Substack) — weekly, research-driven, data-backed

## Operational tip — Gmail filter address

Create a dedicated Gmail filter address such as `<you>+macro@gmail.com`
and subscribe every newsletter there. Two benefits:

1. Your main inbox stays clean.
2. The 📧 **Inbox** tab pulls only mail addressed to that alias, so you
   won't accidentally feed personal email into gpt-4o-mini.

## Connecting Gmail (📧 Inbox tab)

The Inbox tab uses the official Gmail REST API with OAuth 2.0. Setup:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) →
   create a project → enable the **Gmail API** (APIs & Services → Library).
2. Create **OAuth 2.0 credentials** of type **Desktop app**
   (APIs & Services → Credentials → Create credentials → OAuth client ID).
   Download the client_secret JSON and save it to
   `credentials/gmail_credentials.json`.
3. Run the one-time setup script (opens a browser once to authorize):
   ```bash
   PYTHONPATH=. python scripts/gmail_oauth_setup.py
   ```
   This writes the reusable token to `credentials/gmail_token.json`.
   The runtime client refreshes it silently — it never opens a browser again.
4. Set `GMAIL_ADDRESS` and `GMAIL_FILTER_ADDRESS` in `.env` as before:
   ```
   GMAIL_ADDRESS=you@gmail.com
   GMAIL_FILTER_ADDRESS=you+macro@gmail.com
   ```
5. Restart streamlit. Open the Inbox tab → **Test connection** →
   then **Fetch & parse all**. It works exactly as before.

> `credentials/` is git-ignored, so the client secret and token are never
> committed.

### What happens on fetch

For each unread message addressed to your filter address, the pipeline:

1. Strips HTML to text (drops nav/footer/scripts).
2. Extracts hyperlinks; keeps only those whose host is in
   `config/whitelist.py` (Substack, author roots, FRED/BLS/etc.).
3. Fetches up to 5 whitelisted links (`trafilatura` for HTML, `pypdf`
   for PDFs) and appends the extracted text.
4. Extracts text from any PDF *attachments* on the email itself.
5. Caps the assembled context at 40k characters.
6. Sends it through gpt-4o-mini → structured `NewsletterAnalysis`.
7. Persists, stamping the Gmail Message-ID so re-runs are no-ops.

### CLI equivalent

```bash
PYTHONPATH=. python scripts/fetch_inbox.py
PYTHONPATH=. python scripts/fetch_inbox.py --no-mark-seen --no-follow-links --json
```

## Weekly cadence

| Day               | What to do                                                  |
|-------------------|-------------------------------------------------------------|
| Fri / Sat         | Paste each new newsletter into the **Ingest** tab           |
| Sunday            | Open **Dashboard** tab. Review signals + macro panel        |
| Monday morning    | Read **Drift** table. Place trades manually in Tiger app    |
