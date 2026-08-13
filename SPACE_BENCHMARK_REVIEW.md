# Space Sector Benchmark Review — UFO vs. the theme's own candidates

**As of:** 2026-08 · **Scope:** `config.settings.SECTOR_ETFS["UFO"]` (the
parent/thermometer for `config.themes.THEMES["SPACE"]`) vs. the five
expression-basket alternatives already tracked under it in
`config.expressions.EXPRESSIONS["UFO"]`: ARKX, ROKT, NASA, ORBX, XOVR.

**Recommendation: keep UFO as the Space parent for now. Re-run this review
in Q1 2027**, once NASA exits `WARMING_UP` and the SpaceX lockup-expiry
window opens — see "What changes the answer" below. No code change is
included with this note; nothing here clears the bar for one yet.

---

## 1. Why this matters

Everything downstream keys off the parent: `src/signals.py::build_signals`
computes `above_sma` / `relative_strength_3m` / `rs_rank` for UFO itself as
the *sector-level* signal; `src/expression_signals.py` then judges every
expression in the basket (ARKX, ROKT, NASA, ORBX, XOVR) **relative to
whichever ticker holds the parent slot**. Swap UFO for something with less
history, weaker liquidity, or a diluted "space" thesis, and every downstream
self-check degrades along with it — not just the UI label.

## 2. Candidates, judged on the three requested axes

| Ticker | History / SMA200 viable today? | Purity of space-pure exposure | Liquidity (inferred) |
|---|---|---|---|
| **UFO**  | Yes — the config note calls it the "OG" pure-play, i.e. the longest-tenured of the six. Fully SMA200-viable. | **High.** Rocket Lab / MDA / Viasat top weights — actual space-industry operating companies, not a broader thematic wrapper. | Best of the six (most established, oldest, presumably deepest order books — see §5 for the live query to confirm). |
| **ARKX** | Presumably yes — no `WARMING_UP` flag in its config note (unlike NASA/ORBX, which are explicitly flagged). | **Medium-low.** Config note: "includes A&D — overlaps XLI's ITA/XAR." An active fund spanning aerospace *and* defense dilutes the space-specific read, and duplicates exposure the model already gets from XLI's own defense expressions. | Unknown, presumed reasonable (established fund) — confirm via query. |
| **ROKT** | Presumably yes — no `WARMING_UP` flag. | **Low.** Config note: "Space + deep-sea exploration; broader taxonomy" — an unrelated second theme is baked into the same fund. | Unknown — confirm via query. |
| **NASA** | **No, not today.** Config note: "WARMING_UP until ~Feb 2027." Self-check for this ticker cannot leave `WARMING_UP` (no computable SMA200) until then. | Mixed. The fund's overall mandate is "space innovators," but its headline hook — "~10% SpaceX via SPV" — is a *minority* sleeve, not the fund's identity. | Unknown, likely thin pre-2027 (newer, smaller fund) — confirm via query once it has a full quarter of post-WARMING_UP trading. |
| **ORBX** | **No, not today.** Config note: "Launched Apr 2026 — WARMING_UP." Roughly 4 months of history as of this review — well short of 200 trading days. | **High** in principle — config note: "Tight space-exploration focus." But untested at scale (too new to have a liquidity or tracking-quality track record). | Unknown, very likely thin (newly launched) — confirm via query, re-check again once it clears WARMING_UP. |
| **XOVR** | Presumably yes — no `WARMING_UP` flag. | **Low.** Config note is explicit: "~10% SpaceX SPV; not pure space but a SpaceX vehicle." It's a private/public crossover fund first, a space fund a distant second. | Unknown — confirm via query. |

**Bottom line today:** UFO is the only candidate that clears all three bars
simultaneously. NASA and ORBX are mechanically disqualified right now — the
state-refinement pipeline can't even compute a self-check for them yet
(`WARMING_UP` has no SMA200 to test extension against). ARKX, ROKT, and
XOVR are technically usable but each dilutes the "Space" thesis with a
second, unrelated basket (defense, deep-sea, or broad private-equity
crossover) — worse purity than UFO with no offsetting advantage.

## 3. A caveat against UFO, for balance

UFO being the *oldest* fund cuts both ways. Its holdings methodology may
not reflect how the space industry has evolved since its inception — it
was constructed before several of the newer public space names existed,
and (like every candidate here) it cannot hold SpaceX at all, since SpaceX
is not — yet — a public listing UFO's index could include. That's a real,
standing limitation of using UFO as "the" space benchmark. It just isn't,
today, outweighed by any alternative's advantages: nothing else on the list
combines UFO's history, purity, *and* presumed liquidity.

## 4. The SpaceX lockup-expiry trigger (~early 2027)

The already-identified catalyst: once SpaceX's post-IPO lockup expires
(~early 2027), whatever exposure NASA carries via its SpaceX SPV
presumably becomes more liquid and easier to value. Two things are worth
flagging about the timing:

- **NASA's technical and thesis catalysts land in roughly the same window.**
  The config note times NASA's `WARMING_UP` exit at ~Feb 2027 — the same
  general window as the lockup expiry. NASA goes, in one quarter, from
  "can't even self-check" to "self-check works *and* just picked up a
  freshly-liquid SpaceX sleeve." That coincidence is exactly why this
  review is timed for Q1 2027, not later.
- **10% is still a minority stake even post-unlock.** NASA would remain a
  diversified "space innovators" basket with a real, but partial, SpaceX
  leg — not a SpaceX-tracking vehicle. Whether that makes it MORE
  representative of the sector (diversified, like UFO) or just redundant
  with UFO's existing basket depends entirely on what the other ~90% of
  NASA's holdings look like — data this review doesn't have and the next
  one should pull directly (holdings breakdown, not just the SPV note).

**This is a real trigger, not a foregone conclusion.** The lockup expiring
makes NASA *worth re-underwriting* in Q1 2027; it does not by itself make
NASA the better parent. Purity and liquidity both need a fresh, live read
at that time — see below.

## 5. What to check at the Q1 2027 review (concrete queries)

Run directly against `data/prices.db`:

```sql
-- Bar count + date range per Space-theme ticker
SELECT ticker, COUNT(*) AS n_bars, MIN(bar_date) AS first_bar, MAX(bar_date) AS last_bar
FROM ohlcv
WHERE timeframe = '1d' AND ticker IN ('UFO','ARKX','ROKT','NASA','ORBX','XOVR')
GROUP BY ticker
ORDER BY n_bars DESC;

-- Liquidity proxy: trailing-60-day average daily dollar volume
SELECT ticker, AVG(close * volume) AS avg_dollar_volume_60d
FROM ohlcv
WHERE timeframe = '1d' AND ticker IN ('UFO','ARKX','ROKT','NASA','ORBX','XOVR')
  AND bar_date >= date('now', '-60 day')
GROUP BY ticker
ORDER BY avg_dollar_volume_60d DESC;
```

Alongside the query results, confirm at that time:

1. NASA has actually cleared `WARMING_UP` in the live Expressions tab (its
   Self-check state, not just the calendar date, since price gaps or a
   late cold-start pull could push this later than the config comment's
   estimate).
2. NASA's full holdings breakdown (not just the SpaceX SPV %) — is the
   other ~90% genuinely diversified space exposure, or heavily overlapping
   with UFO's own top names?
3. Whether the SpaceX lockup actually resolved as expected, and whether
   NASA's SPV stake became more liquid/valuable as a result, or whether the
   unlock event changed nothing observable in NASA's own price/volume
   behavior.
4. A fresh average-dollar-volume read for every candidate, not just NASA —
   UFO's own liquidity should be re-confirmed too, not assumed static.

**Do not auto-promote NASA (or anything) the moment it exits `WARMING_UP`.**
Technical viability is necessary, not sufficient — purity and liquidity
must be re-confirmed with live data at that time, per §2's framework above.

## 6. If a future review concludes a change is warranted

For reference only (not applied now): the one-line edit would be swapping
the `parent_sector` on `config.themes.THEMES["SPACE"]` from `"UFO"` to the
chosen ticker, plus re-pointing `config.settings.SECTOR_ETFS["UFO"]`'s key
(or introducing a new key) and the affected drift test
(`tests/test_config_layer.py`) that currently asserts on the UFO basket
shape. That's a single, isolated change — deliberately scoped separately
from this note, per the task brief.