#!/usr/bin/env python
"""CLI: backfill historical newsletter sentiment from Gmail.

Default `fetch_inbox.py` only pulls UNREAD mail — useful for the daily/weekly
refresh, useless for filling the multi-year gap before sentiment.db started.
This script accepts an explicit date range, drops the UNREAD constraint, and
DOES NOT modify any Gmail labels (so your inbox state is preserved).

Recommended workflow:

  1. Cheap exploration — count emails matching the range:
       PYTHONPATH=. python3 scripts/backfill_inbox.py --from 2024-01-01 --count-only

  2. Small validation batch — ingest the most recent 20 to confirm the
     pipeline works on real historical mail:
       PYTHONPATH=. python3 scripts/backfill_inbox.py --from 2024-06-01 --max 20

  3. Real backfill — commit to ingesting everything in the chosen window.
     Cost ≈ $0.001 × N (gpt-4o-mini). Time ≈ 3 sec/message:
       PYTHONPATH=. python3 scripts/backfill_inbox.py --from 2024-01-01

  4. Resume: dedup is on (gmail_message_id, content_hash) so re-running with
     the same window is idempotent — already-ingested mail is SKIPPED, only
     new gaps get LLM-processed.

Notes:
  - Backfill runs sequentially and prints per-message status. Ctrl-C is safe;
    progress so far is committed.
  - `--from` is INCLUSIVE, `--to` is EXCLUSIVE (matching Gmail's convention).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gmail_client import build_backfill_query, count_messages  # noqa: E402
from src.nlp_pipeline import fetch_and_ingest  # noqa: E402


# Rough cost-per-message at gpt-4o-mini ($0.15/M input, $0.60/M output tokens)
# for a typical newsletter (~6k input tokens, ~400 output tokens) = ~$0.0012.
# Conservative budget figure for the "cost estimate" line.
_COST_PER_MSG_USD = 0.0015


def _parse_date(raw: str) -> date:
    return datetime.fromisoformat(raw).date()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--from", dest="start", required=True, type=_parse_date,
        help="Backfill window start (YYYY-MM-DD), INCLUSIVE.",
    )
    ap.add_argument(
        "--to", dest="end", default=None, type=_parse_date,
        help="Backfill window end (YYYY-MM-DD), EXCLUSIVE. Default: today.",
    )
    ap.add_argument(
        "--count-only", action="store_true",
        help="Just count messages in the window and exit. No LLM calls.",
    )
    ap.add_argument(
        "--max", type=int, default=None, dest="max_n",
        help="Cap on messages to fetch+ingest. Useful for validation batches.",
    )
    ap.add_argument(
        "--no-follow-links", action="store_true",
        help="Skip whitelist link/PDF enrichment (faster, cheaper, less coverage).",
    )
    ap.add_argument(
        "--yes", action="store_true",
        help="Skip the cost-estimate confirmation prompt.",
    )
    args = ap.parse_args()

    end = args.end or date.today()
    if args.start >= end:
        print(f"error: --from ({args.start}) must be before --to ({end})",
              file=sys.stderr)
        return 2

    query = build_backfill_query(args.start, end)
    print(f"Gmail query: {query!r}")

    # Always count first — it's cheap and informative.
    print(f"Counting matching messages…", flush=True)
    try:
        n_match = count_messages(query)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    n_planned = min(n_match, args.max_n) if args.max_n else n_match
    est_cost = n_planned * _COST_PER_MSG_USD
    est_minutes = n_planned * 3.0 / 60.0
    print(f"  Matched: {n_match} messages in window "
          f"{args.start} → {end} ({(end - args.start).days} days)")
    if args.max_n and args.max_n < n_match:
        print(f"  Capped:  {args.max_n} (will ingest most recent)")
    print(f"  Estimated cost (OpenAI):  ${est_cost:.2f} "
          f"(~${_COST_PER_MSG_USD*1000:.2f} per 1000 messages)")
    print(f"  Estimated time:           ~{est_minutes:.0f} minutes")
    print()

    if args.count_only:
        return 0
    if n_planned == 0:
        print("(nothing to do)")
        return 0
    if not args.yes:
        try:
            resp = input(f"Proceed with ingestion of {n_planned} messages? [y/N] ")
        except EOFError:
            resp = ""
        if resp.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 0

    counters = {"ingested": 0, "skipped_dupe_message_id": 0,
                "skipped_dupe_content_hash": 0, "error": 0}

    def _on_message(entry: dict) -> None:
        status = entry.get("status", "?")
        counters[status] = counters.get(status, 0) + 1
        total = sum(counters.values())
        flag = {"ingested": "✓", "skipped_dupe_message_id": "·",
                "skipped_dupe_content_hash": "·", "error": "✗"}.get(status, "?")
        sectors = ""
        if entry.get("sectors"):
            sectors = f"  sectors={entry['sectors']}"
        err = f"  err={entry.get('error')}" if status == "error" else ""
        print(f"  [{total:4d}/{n_planned}] {flag} {entry.get('date', '?')} "
              f"{(entry.get('from') or '?')[:35]:35s}  "
              f"{(entry.get('subject') or '')[:50]:50s}{sectors}{err}",
              flush=True)

    print(f"Ingesting {n_planned} messages…\n")
    fetch_and_ingest(
        mark_seen=False,                    # never touch the user's inbox state
        follow_links=not args.no_follow_links,
        query=query,
        limit=args.max_n,
        on_message=_on_message,
    )

    print("\n" + "=" * 70)
    print(f"Summary:")
    for k, v in counters.items():
        print(f"  {k:30s} {v:5d}")
    if counters.get("error", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
