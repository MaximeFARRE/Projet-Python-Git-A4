"""
Définition de l'univers d'actifs utilisable par le Compte A (et réutilisable par le Compte B).

Structure :
- par catégorie (indices, forex, actions, matières premières, ...),
- chaque actif contient au minimum un nom lisible et un ticker Yahoo Finance.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Asset:
    name: str        # Nom lisible ("CAC 40", "EUR/USD", "Apple")
    ticker: str      # Ticker Yahoo Finance ("^FCHI", "EURUSD=X", "AAPL")
    asset_class: str # Classe d'actif ("Indices", "Forex", ...)


# Univers d'actifs de base : à étendre au besoin
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
    "Actions": [
        Asset(name="TotalEnergies (Paris)", ticker="TTE.PA", asset_class="Actions"),
        Asset(name="LVMH (Paris)", ticker="MC.PA", asset_class="Actions"),
        Asset(name="Apple (US)", ticker="AAPL", asset_class="Actions"),
        Asset(name="Microsoft (US)", ticker="MSFT", asset_class="Actions"),
    ],
    "Matières premières": [
        Asset(name="Or (Gold futures)", ticker="GC=F", asset_class="Matières premières"),
        Asset(name="Pétrole WTI", ticker="CL=F", asset_class="Matières premières"),
    ],
}


def get_asset_classes() -> List[str]:
    """
    Renvoie la liste des classes d'actifs (Indices, Forex, Actions, ...).
    """
    return list(ASSET_UNIVERSE.keys())



