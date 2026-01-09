"""
Definition of the asset universe usable by Quant A (and reusable by Quant B).

Structure:
- grouped by category (indices, forex, equities, commodities, ...),
- each asset contains at least a readable name and a Yahoo Finance ticker.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Asset:
    name: str        # Readable name ("CAC 40", "EUR/USD", "Apple")
    ticker: str      # Yahoo Finance ticker ("^FCHI", "EURUSD=X", "AAPL")
    asset_class: str # Asset class ("Indices", "Forex", ...)


# Base asset universe: extend as needed
ASSET_UNIVERSE: Dict[str, List[Asset]] = {
    "Indices": [
        Asset(name="CAC 40", ticker="^FCHI", asset_class="Indices"),
        Asset(name="DAX 40", ticker="^GDAXI", asset_class="Indices"),
        Asset(name="S&P 500", ticker="^GSPC", asset_class="Indices"),
    ],
    "Forex": [
        Asset(name="EUR/USD", ticker="EURUSD=X", asset_class="Forex"),
        Asset(name="GBP/USD", ticker="GBPUSD=X", asset_class="Forex"),
        Asset(name="USD/JPY", ticker="JPY=X", asset_class="Forex"),
    ],
    "Equities": [
        Asset(name="TotalEnergies (Paris)", ticker="TTE.PA", asset_class="Equities"),
        Asset(name="LVMH (Paris)", ticker="MC.PA", asset_class="Equities"),
        Asset(name="Apple (US)", ticker="AAPL", asset_class="Equities"),
        Asset(name="Microsoft (US)", ticker="MSFT", asset_class="Equities"),
    ],
    "Commodities": [
        Asset(name="Gold (futures)", ticker="GC=F", asset_class="Commodities"),
        Asset(name="WTI Crude Oil", ticker="CL=F", asset_class="Commodities"),
    ],
}


def get_asset_classes() -> List[str]:
    """
    Return the list of available asset classes (Indices, Forex, Equities, ...).
    """
    return list(ASSET_UNIVERSE.keys())


def get_assets_by_class(asset_class: str) -> List[Asset]:
    """
    Return the list of assets for a given asset class.
    """
    return ASSET_UNIVERSE.get(asset_class, [])


def get_default_asset() -> Asset:
    """
    Default asset (can be used to initialize the UI).
    Default: CAC 40.
    """
    return ASSET_UNIVERSE["Indices"][0]
