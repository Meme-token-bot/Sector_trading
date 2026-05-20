"""Weekly refresh job."""
from __future__ import annotations

from config.settings import BENCHMARK, SECTOR_ETFS
from src.db import aggregate_sentiment
from src.market_engine import (
    compute_sector_metrics, copper_gold_ratio, dxy_level,
    fetch_fred_indicators, fetch_macro_prices, fetch_prices,
    gold_oil_ratio, vix_level, yield_curve_spread,
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
    cgr = copper_gold_ratio(macro)
    dxy = dxy_level(macro)
    vix = vix_level(macro)
    yc = yield_curve_spread()
    fred = fetch_fred_indicators()

    def _val(d, key, fmt="+.2f"):
        v = d.get(key)
        return format(v, fmt) if v is not None and v == v else "—"  # nan != nan

    print(f"\nMacro: Gold/Oil       = {_val(gor, 'current')}  (z={_val(gor, 'z_score_1y')})")
    print(f"       Copper/Gold    = {_val(cgr, 'current', '+.4f')}  (z={_val(cgr, 'z_score_1y')})")
    print(f"       DXY            = {_val(dxy, 'current')}  (z={_val(dxy, 'z_score_1y')})")
    print(f"       VIX            = {_val(vix, 'current')}  (z={_val(vix, 'z_score_1y')})")
    print(f"       10Y-2Y         = {_val(yc, 'current')}%  (30d slope {_val(yc, 'slope_30d', '+.4f')}/day)")
    hy = fred.get("HY_OAS", {})
    ust10 = fred.get("UST10", {})
    real10 = fred.get("REAL_10Y", {})
    be = fred.get("BREAKEVEN_5Y5Y", {})
    print(f"       HY OAS         = {_val(hy, 'current')}%  (z={_val(hy, 'z_score_1y')})")
    print(f"       10Y nominal    = {_val(ust10, 'current')}%  (30d slope {_val(ust10, 'slope_30d', '+.4f')}/day)")
    print(f"       10Y real       = {_val(real10, 'current')}%  (30d slope {_val(real10, 'slope_30d', '+.4f')}/day)")
    print(f"       5Y5Y breakeven = {_val(be, 'current')}%  (z={_val(be, 'z_score_1y')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
