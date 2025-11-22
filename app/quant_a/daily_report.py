import datetime as dt
from pathlib import Path

import pandas as pd

from app.quant_a.data_loader import load_cac40_history
from app.quant_a.strategies import buy_and_hold
from app.quant_a.metrics import compute_all_metrics


def _compute_period_return(prices: pd.Series, start_date: dt.date) -> float:
    """
    Retour % entre le premier et le dernier prix à partir de start_date.
    Si pas assez de données, renvoie NaN.
    """
    if prices.empty:
        return float("nan")

    # Filtre à partir de start_date (en comparant les dates sans l'heure)
    mask = prices.index.date >= start_date
    sub = prices[mask]

    if len(sub) < 2:
        return float("nan")

    start_price = float(sub.iloc[0])
    end_price = float(sub.iloc[-1])

    if start_price == 0:
        return float("nan")

    return (end_price / start_price - 1) * 100.0