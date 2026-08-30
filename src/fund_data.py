"""Fund metadata fetcher — expense ratio, AUM, display name via yfinance.

This is the ONE network-touching module behind the Fund Screener. Returns
computed FROM PRICE HISTORY (which this codebase already caches reliably in
prices.db) are handled by src.fund_screener directly, given a `prices`
frame the caller loads via src.price_store — this module supplies ONLY the
things that genuinely can't come from a price series: expense ratio, AUM,
and a display name.

Coverage is inconsistent across fund families and has drifted across
yfinance versions, so every field is read defensively with multiple
candidate keys, and a single ticker's failure never aborts the batch (each
call is independently wrapped) -- never crash the whole screener over one
bad ticker.

`fetch_one` is a swappable seam (mirrors src/ticker_news.py's `http_get`
parameter) so tests never hit the network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FundRecord:
    ticker: str
    name: str | None = None
    expense_ratio: float | None = None   # decimal, e.g. 0.0035 = 0.35%
    aum: float | None = None             # dollars
    error: str | None = None


def _coerce_expense_ratio(raw) -> float | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    # Some yfinance/fund-family payloads report this as a percent (0.35)
    # rather than a decimal (0.0035) -- an expense ratio > 1.0 (100%) is
    # never real, so treat that as the percent form and rescale.
    if v > 1.0:
        v = v / 100.0
    if v < 0:
        return None
    return v


def _default_fetch_one(ticker: str) -> FundRecord:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception as e:  # noqa: BLE001
        return FundRecord(ticker=ticker, error=f"{type(e).__name__}: {e}")

    name = info.get("longName") or info.get("shortName") or None

    expense_ratio = None
    for key in ("annualReportExpenseRatio", "netExpenseRatio", "expenseRatio"):
        v = info.get(key)
        if v is not None:
            expense_ratio = _coerce_expense_ratio(v)
            if expense_ratio is not None:
                break

    aum = None
    for key in ("totalAssets", "netAssets"):
        v = info.get(key)
        if v is not None:
            try:
                aum = float(v)
                break
            except (TypeError, ValueError):
                continue

    return FundRecord(ticker=ticker, name=name, expense_ratio=expense_ratio, aum=aum)


def fetch_fund_metadata(
    tickers: list[str],
    fetch_one: Callable[[str], FundRecord] = _default_fetch_one,
) -> dict[str, FundRecord]:
    """One FundRecord per ticker.

    A single ticker's exception never aborts the batch -- wrapped again
    here even though `_default_fetch_one` already catches its own errors,
    so a custom `fetch_one` passed by a caller/test doesn't have to repeat
    that discipline to get the same safety guarantee.
    """
    out: dict[str, FundRecord] = {}
    for t in tickers:
        try:
            out[t] = fetch_one(t)
        except Exception as e:  # noqa: BLE001
            out[t] = FundRecord(ticker=t, error=f"{type(e).__name__}: {e}")
    return out