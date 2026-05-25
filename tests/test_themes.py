"""Theme taxonomy integrity + drift guard against schemas.ThemeKey."""
from __future__ import annotations

import typing

from config.expressions import EXPRESSIONS, all_expression_tickers
from config.settings import SECTOR_ETFS
from config.themes import (
    THEMES, THEME_KEYS, all_theme_keys, theme_for_ticker, themes_for_sector,
)
from src.schemas import ThemeKey


# Plain sector proxies — these intentionally map to NO theme.
_PLAIN_PROXIES = {
    "XLK", "VGT", "XLY", "XLC", "VOX", "XLF", "XLI", "XLB", "XLE",
    "XLV", "XLP", "VDC", "XLU", "XLRE", "VNQ", "UFO",
}


def test_every_theme_parent_is_a_sector():
    for t in THEMES.values():
        assert t.parent_sector in SECTOR_ETFS, t.key


def test_each_expression_ticker_maps_to_at_most_one_theme():
    seen: dict[str, str] = {}
    for t in THEMES.values():
        for tk in t.expression_tickers:
            assert tk not in seen, f"{tk} in both {seen.get(tk)} and {t.key}"
            seen[tk] = t.key


def test_plain_proxies_have_no_theme():
    for tk in _PLAIN_PROXIES:
        assert theme_for_ticker(tk) is None, tk


def test_only_plain_proxies_are_unmapped():
    unmapped = [tk for tk in all_expression_tickers() if theme_for_ticker(tk) is None]
    assert set(unmapped) == _PLAIN_PROXIES


def test_theme_tickers_exist_in_expressions():
    all_tickers = set(all_expression_tickers())
    for t in THEMES.values():
        for tk in t.expression_tickers:
            assert tk in all_tickers, f"{t.key} references unknown ticker {tk}"


def test_themes_for_sector_matches_parent():
    for sector in SECTOR_ETFS:
        for t in themes_for_sector(sector):
            assert t.parent_sector == sector


def test_schema_themekey_matches_taxonomy():
    """schemas.ThemeKey Literal must list exactly the THEME_KEYS (drift guard)."""
    literal_values = set(typing.get_args(ThemeKey))
    assert literal_values == set(THEME_KEYS)
    assert len(literal_values) == len(THEME_KEYS) == len(set(all_theme_keys()))
