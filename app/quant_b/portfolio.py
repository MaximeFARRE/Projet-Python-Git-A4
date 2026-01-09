import pandas as pd
import numpy as np


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements logarithmiques à partir des prix.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()


def normalize_weights(weights: dict) -> dict:
    """
    Normalise les poids pour que la somme soit égale à 1.
    """
    total = sum(weights.values())
    if total == 0:
        raise ValueError("La somme des poids ne peut pas être nulle.")
    return {k: v / total for k, v in weights.items()}


def compute_portfolio_value(
    prices: pd.DataFrame,
    weights: dict,
    rebalance: str = "M",
    base: float = 100.0,
) -> pd.Series:
    """
    Calcule la valeur cumulée d'un portefeuille multi-actifs.
    rebalance : 'D', 'W', 'M' ou None
    """
    weights = normalize_weights(weights)

    returns = compute_returns(prices)

    weights_df = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        data=np.nan,
    )

    if rebalance is None:
        weights_df.iloc[0] = list(weights.values())
        weights_df = weights_df.ffill()
    else:
        # Normalisation de la fréquence pour pandas (évite le warning)
        pandas_rebalance = rebalance
        if rebalance == "M":
            pandas_rebalance = "ME"
        rebalance_dates = returns.resample(pandas_rebalance).first().index
        for date in rebalance_dates:
            if date in weights_df.index:
                weights_df.loc[date] = list(weights.values())
        weights_df = weights_df.ffill()


    portfolio_returns = (weights_df * returns).sum(axis=1)
    portfolio_value = base * np.exp(portfolio_returns.cumsum())

    return portfolio_value
