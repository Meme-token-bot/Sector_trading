"""Weekly refresh job."""
from __future__ import annotations

from config.settings import BENCHMARK, SECTOR_ETFS
from src.db import aggregate_sentiment
from src.market_engine import (
    compute_sector_metrics, fetch_macro_prices, fetch_prices,
    gold_oil_ratio, yield_curve_spread,
)
from src.signals import build_signals, target_weights


def main() -> int:
    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    prices = fetch_prices(tickers)
    metrics = compute_sector_metrics(prices)
    sentiment = aggregate_sentiment()
    sigs = build_signals(metrics, sentiment)

    print("=" * 90)
    print("SECTOR SIGNALS")
    print("=" * 90)
    print(sigs[["name", "relative_strength_3m", "above_sma",
                "sentiment_score", "n_obs", "signal", "reasons"]].to_string(
        formatters={"relative_strength_3m": "{:+.2%}".format,
                    "sentiment_score": "{:+.1f}".format}))

    print("\nTarget weights (equal-weight BUYs, 5% cash):")
    print(target_weights(sigs).to_string(float_format="{:.1%}".format) or "  (no BUY signals)")

    macro = fetch_macro_prices()
    gor = gold_oil_ratio(macro)
    yc = yield_curve_spread()
    print(f"\nMacro: Gold/Oil = {gor['current']:.2f}  (z={gor['z_score_1y']:+.2f})")
    print(f"       10Y-2Y   = {yc['current']:+.2f}%  (30d slope {yc['slope_30d']:+.4f}/day)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
