from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetInstrument:
    ticker: str
    label: str
    group: str
    asset_class: str
    execution_ticker: str
    execution_route: str
    is_primary: bool = False


EXPANDED_UNIVERSE: dict[str, list[AssetInstrument]] = {
    "Bitcoin": [
        AssetInstrument("BTC-USD", "Bitcoin (spot)", "Bitcoin", "Crypto",
                        "BTC", "DIRECT_SPOT_WALLET", is_primary=True),
        AssetInstrument("IBIT", "iShares Bitcoin Trust", "Bitcoin", "Crypto",
                        "BTC", "DIRECT_SPOT_WALLET"),
        AssetInstrument("FBTC", "Fidelity Wise Origin Bitcoin Fund", "Bitcoin", "Crypto",
                        "BTC", "DIRECT_SPOT_WALLET"),
    ],
    "Ethereum": [
        AssetInstrument("ETH-USD", "Ethereum (spot)", "Ethereum", "Crypto",
                        "ETH", "DIRECT_SPOT_WALLET", is_primary=True),
        AssetInstrument("ETHA", "iShares Ethereum Trust", "Ethereum", "Crypto",
                        "ETH", "DIRECT_SPOT_WALLET"),
        AssetInstrument("FETH", "Fidelity Ethereum Fund", "Ethereum", "Crypto",
                        "ETH", "DIRECT_SPOT_WALLET"),
    ],
    "Solana": [
        AssetInstrument("SOL-USD", "Solana (spot)", "Solana", "Crypto",
                        "SOL", "DIRECT_SPOT_WALLET", is_primary=True),
    ],
    "Crypto Equities & Infrastructure": [
        AssetInstrument("COIN", "Coinbase Global", "Crypto Equities", "Crypto",
                        "COIN", "BROKERAGE_TICKER", is_primary=True),
        AssetInstrument("BITQ", "Bitwise Crypto Innovators ETF", "Crypto Equities", "Crypto",
                        "BITQ", "BROKERAGE_TICKER"),
        AssetInstrument("BKCH", "Global X Blockchain ETF", "Crypto Equities", "Crypto",
                        "BKCH", "BROKERAGE_TICKER"),
        AssetInstrument("WGMI", "Valkyrie Bitcoin Miners ETF", "Crypto Equities", "Crypto",
                        "WGMI", "BROKERAGE_TICKER"),
    ],
    "Gold": [
        AssetInstrument("GLD", "SPDR Gold Shares", "Gold", "Metals",
                        "GLD", "BROKERAGE_TICKER", is_primary=True),
        AssetInstrument("IAU", "iShares Gold Trust", "Gold", "Metals",
                        "IAU", "BROKERAGE_TICKER"),
        AssetInstrument("GDX", "VanEck Gold Miners ETF", "Gold", "Metals",
                        "GDX", "BROKERAGE_TICKER"),
    ],
    "Silver": [
        AssetInstrument("SLV", "iShares Silver Trust", "Silver", "Metals",
                        "SLV", "BROKERAGE_TICKER", is_primary=True),
        AssetInstrument("SIL", "Global X Silver Miners ETF", "Silver", "Metals",
                        "SIL", "BROKERAGE_TICKER"),
    ],
    "Metals & Mining (broad)": [
        AssetInstrument("XME", "SPDR S&P Metals & Mining ETF", "Metals & Mining (broad)",
                        "Metals", "XME", "BROKERAGE_TICKER", is_primary=True),
        AssetInstrument("PICK", "iShares MSCI Global Metals & Mining Producers",
                        "Metals & Mining (broad)", "Metals", "PICK", "BROKERAGE_TICKER"),
    ],
    "Copper Mining": [
        AssetInstrument("COPX", "Global X Copper Miners ETF", "Copper Mining", "Metals",
                        "COPX", "BROKERAGE_TICKER", is_primary=True),
    ],
}


def all_expanded_tickers() -> list[str]:
    seen, out = set(), []
    for group in EXPANDED_UNIVERSE.values():
        for a in group:
            if a.ticker not in seen:
                seen.add(a.ticker)
                out.append(a.ticker)
    return out


def primary_instrument(group: str) -> AssetInstrument | None:
    for a in EXPANDED_UNIVERSE.get(group, []):
        if a.is_primary:
            return a
    items = EXPANDED_UNIVERSE.get(group, [])
    return items[0] if items else None
