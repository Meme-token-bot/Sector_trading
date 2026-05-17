"""Pydantic schemas — also drive OpenAI Structured Outputs.

The schemas are deliberately rigid: enum macro bias, fixed-domain ticker list,
integer -5..+5 sentiment. This forces gpt-4o-mini to commit rather than emit
mealy-mouthed adjectives we can't aggregate.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SectorTicker = Literal[
    "XLK", "XLY", "XLC", "XLF", "XLI", "XLB",
    "XLE", "XLV", "XLP", "XLU", "XLRE",
]


class MacroBias(str, Enum):
    DEFENSIVE = "Defensive"
    EXPANSIONARY = "Expansionary"
    NEUTRAL = "Neutral"


class SectorRating(BaseModel):
    ticker: SectorTicker
    sentiment_score: int = Field(
        ..., ge=-5, le=5,
        description="Author's view on this sector. -5 strongly bearish, "
                    "0 neutral/no clear view, +5 strongly bullish.",
    )
    reasoning: str = Field(
        ..., max_length=400,
        description="One or two sentences capturing the author's actual stated "
                    "reasoning. Do NOT invent — if the author did not address "
                    "the sector, do not include it.",
    )


class NewsletterAnalysis(BaseModel):
    """One newsletter -> one of these. Persisted to SQLite."""
    author: str = Field(..., description="Author or publication name.")
    publication_date: date = Field(..., description="Date the piece was published.")
    overall_macro_bias: MacroBias
    sector_ratings: list[SectorRating] = Field(
        default_factory=list,
        description="Only include sectors the author explicitly discusses. "
                    "Omit sectors that are not addressed — do not fill with zeros.",
    )
    summary: str = Field(
        ..., max_length=600,
        description="2-3 sentence neutral summary of the macro thesis.",
    )
