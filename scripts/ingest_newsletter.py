"""CLI: ingest a newsletter."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from src.nlp_pipeline import ingest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, help="Path to newsletter text file.")
    ap.add_argument("--author", type=str, default=None, help="Author hint.")
    ap.add_argument("--date", type=str, default=None,
                    help="Publication date YYYY-MM-DD (otherwise today).")
    args = ap.parse_args()

    if args.file:
        raw = args.file.read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            ap.error("No input. Provide --file or pipe text to stdin.")
        raw = sys.stdin.read()

    if not raw.strip():
        ap.error("Empty input.")

    date_hint = date.fromisoformat(args.date) if args.date else date.today()

    analysis, nid = ingest(raw, author_hint=args.author, date_hint=date_hint)
    if nid is None:
        print(f"[skip] Already ingested: {analysis.author} {analysis.publication_date}")
        return 0
    print(f"[ok] Ingested newsletter id={nid}: {analysis.author} {analysis.publication_date}")
    print(f"     Bias: {analysis.overall_macro_bias.value}")
    for r in analysis.sector_ratings:
        print(f"     {r.ticker:5} {r.sentiment_score:+d}  {r.reasoning[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
