# app/quant_a/strategies.py
import pandas as pd
import numpy as np


def _compute_equity_curve_from_position(prices: pd.Series, position: pd.Series) -> pd.DataFrame:
    """
    Utilitaire : à partir d'une série de prix et d'une série de positions (0/1),
    calcule les rendements et la courbe d’équité.
    """
    prices = prices.sort_index()
    returns = prices.pct_change().fillna(0.0)

    # On applique la position décalée d'un jour (on prend la position connue au début de la journée)
    strategy_returns = position.shift(1).fillna(0.0) * returns

    equity_curve = (1 + strategy_returns).cumprod()

    df = pd.DataFrame(
        {
            "price": prices,
            "position": position,
            "strategy_returns": strategy_returns,
            "equity_curve": equity_curve,
        }
    )

    return df


def buy_and_hold(prices: pd.Series) -> pd.DataFrame:
    """
    Stratégie Buy & Hold : on est investi à 100% dès le début et jusqu'à la fin.
    """
    position = pd.Series(1.0, index=prices.index, name="position")
    return _compute_equity_curve_from_position(prices, position)


def moving_average_crossover(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.DataFrame:
    """
    Stratégie Moving Average Crossover :
    - On achète (position = 1) quand la moyenne courte > moyenne longue
    - On sort du marché (position = 0) sinon.
    """
    prices = prices.sort_index()

    if short_window >= long_window:
        raise ValueError("short_window doit être strictement inférieur à long_window")

    ma_short = prices.rolling(window=short_window).mean()
    ma_long = prices.rolling(window=long_window).mean()

    signal = (ma_short > ma_long).astype(float)
    signal.name = "position"

    df = _compute_equity_curve_from_position(prices, signal)
    df["ma_short"] = ma_short
    df["ma_long"] = ma_long

    return df
