"""Thin wrapper around tigeropen — positions + drift calc."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import (
    SECTOR_ETFS, TIGER_ACCOUNT, TIGER_ID,
    TIGER_PRIVATE_KEY_PATH, TIGER_SANDBOX, tiger_configured,
)


@dataclass
class AccountSnapshot:
    net_liquidation: float
    cash: float
    positions: pd.DataFrame


def _build_client():
    if not tiger_configured():
        raise RuntimeError(
            "Tiger credentials incomplete. Check TIGER_ID, TIGER_ACCOUNT, "
            "TIGER_PRIVATE_KEY_PATH in .env (and that the .pem file exists)."
        )
    from tigeropen.common.util.signature_utils import read_private_key
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.trade.trade_client import TradeClient

    cfg = TigerOpenClientConfig(sandbox_debug=TIGER_SANDBOX)
    cfg.private_key = read_private_key(TIGER_PRIVATE_KEY_PATH)
    cfg.tiger_id = TIGER_ID
    cfg.account = TIGER_ACCOUNT
    return TradeClient(cfg)


def fetch_account_snapshot() -> AccountSnapshot:
    client = _build_client()

    portfolio = client.get_assets(account=TIGER_ACCOUNT, segment=True, market_value=True)
    p0 = portfolio[0]
    nlv = float(p0.summary.net_liquidation or 0)
    cash = float(p0.summary.cash or 0)

    positions = client.get_positions(account=TIGER_ACCOUNT)
    rows = []
    for pos in positions or []:
        sym = pos.contract.symbol
        mv = float(pos.market_value or 0)
        rows.append({
            "symbol": sym,
            "quantity": float(pos.quantity or 0),
            "avg_cost": float(pos.average_cost or 0),
            "market_value": mv,
            "weight": mv / nlv if nlv else 0.0,
        })

    df = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame(
        columns=["quantity", "avg_cost", "market_value", "weight"]
    ).rename_axis("symbol")

    return AccountSnapshot(net_liquidation=nlv, cash=cash, positions=df)


def compute_drift(snapshot: AccountSnapshot,
                  targets: pd.Series) -> pd.DataFrame:
    universe = list(SECTOR_ETFS.keys())
    current = snapshot.positions["weight"].reindex(universe).fillna(0.0)
    current_val = snapshot.positions["market_value"].reindex(universe).fillna(0.0)

    tgt = targets.reindex(universe).fillna(0.0)
    target_val = tgt * snapshot.net_liquidation
    trade_val = target_val - current_val

    return pd.DataFrame({
        "target_weight": tgt,
        "current_weight": current,
        "drift": current - tgt,
        "target_value": target_val,
        "current_value": current_val,
        "trade_value": trade_val,
    }).sort_values("trade_value", ascending=False)


def compute_drift_by_sector(snapshot: AccountSnapshot,
                            targets: pd.Series) -> pd.DataFrame:
    """Drift accounting that rolls expression holdings up to their signal sector.

    A position in GDX is counted toward the XLB sector target. Cash and
    any holding that isn't in any expression list is ignored from the
    sector totals (but shown in a separate 'unmapped' row in the UI).
    """
    from config.expressions import sector_for_ticker
    from config.settings import SECTOR_ETFS

    pos = snapshot.positions
    sector_value: dict[str, float] = {s: 0.0 for s in SECTOR_ETFS}
    unmapped: dict[str, float] = {}

    for symbol, row in pos.iterrows():
        mv = float(row["market_value"])
        sector = sector_for_ticker(symbol)
        if sector is None:
            unmapped[symbol] = mv
        else:
            sector_value[sector] += mv

    nlv = snapshot.net_liquidation or 1.0
    current_weight = pd.Series({s: v / nlv for s, v in sector_value.items()})
    current_val = pd.Series(sector_value)

    tgt = targets.reindex(SECTOR_ETFS.keys()).fillna(0.0)
    target_val = tgt * snapshot.net_liquidation

    df = pd.DataFrame({
        "target_weight": tgt,
        "current_weight": current_weight,
        "drift": current_weight - tgt,
        "target_value": target_val,
        "current_value": current_val,
        "trade_value": target_val - current_val,
    }).sort_values("trade_value", ascending=False)

    df.attrs["unmapped"] = unmapped
    return df
