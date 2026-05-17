"""CLI: fetch unread Gmail and ingest each message."""
from __future__ import annotations

import argparse
import json

from src.nlp_pipeline import fetch_and_ingest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mark-seen", action="store_true",
                    help="Do not flip the IMAP \\Seen flag (safe for testing).")
    ap.add_argument("--no-follow-links", action="store_true",
                    help="Skip whitelist link/PDF enrichment.")
    ap.add_argument("--json", action="store_true",
                    help="Emit the per-message report as JSON.")
    args = ap.parse_args()

    report = fetch_and_ingest(
        mark_seen=not args.no_mark_seen,
        follow_links=not args.no_follow_links,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    if not report:
        print("No unread messages matched the filter.")
        return 0

    for r in report:
        status = r["status"]
        line = f"[{status:>26}] {r.get('date', '?')}  {r.get('from', '?'):<40} {r.get('subject', '')[:60]}"
        if status == "ingested":
            line += f"  sectors={r.get('sectors', [])}"
        if status == "error":
            line += f"  err={r.get('error', '')}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
