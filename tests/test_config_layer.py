from config.expressions import EXPRESSIONS, Expression, all_expression_tickers
from config.expanded_universe import all_expanded_tickers


def test_ishares_alternates_landed_in_every_core_sector():
    for sector in ["XLK", "XLV", "XLF", "XLY", "XLC", "XLI", "XLP", "XLE", "XLU", "XLRE", "XLB"]:
        tickers = [e.ticker for e in EXPRESSIONS[sector]]
        assert len(tickers) >= 2, f"{sector} missing its iShares alternate: {tickers}"


def test_original_positional_style_expression_calls_still_work():
    """The exact call pattern already used ~70 times in the real
    config/expressions.py -- 5 positional args, no keywords."""
    e = Expression("SOXX", "iShares Semiconductors", "operating_leverage", 1.6,
                   "Semi earnings cycle amplifies tech beta.")
    assert e.execution_route == "BROKERAGE_TICKER"   # new field defaults correctly
    assert e.execution_ticker == ""


def test_expanded_universe_tickers_are_disjoint_from_expression_tickers_by_role():
    """GDX intentionally appears in BOTH systems (judged relative to XLB
    there, judged on its own setup here) -- confirm that's the ONLY kind
    of overlap, i.e. nothing in the expanded universe is a a plain SECTOR
    proxy ticker (XLK, XLF, etc.) that would collide with the sentiment-
    gated model's own identity."""
    core_proxies = {"XLK", "XLY", "XLC", "XLF", "XLI", "XLB", "XLE", "XLV",
                    "XLP", "XLU", "XLRE", "UFO"}
    assert set(all_expanded_tickers()).isdisjoint(core_proxies)


def test_no_duplicate_tickers_within_the_expanded_universe_itself():
    tickers = all_expanded_tickers()
    assert len(tickers) == len(set(tickers))


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
