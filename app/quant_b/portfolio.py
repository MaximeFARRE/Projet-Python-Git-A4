import pandas as pd
import numpy as np


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute logarithmic returns from price data.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()


def normalize_weights(weights: dict) -> dict:
    """
    Normalize weights so that their sum equals 1.
    """
    total = sum(weights.values())
    if total == 0:
        raise ValueError("The sum of weights cannot be zero.")
    return {k: v / total for k, v in weights.items()}


def compute_portfolio_value(
    prices: pd.DataFrame,
    weights: dict,
    rebalance: str = "M",
    base: float = 100.0,
) -> pd.Series:
    """
    Compute the cumulative value of a multi-asset portfolio.
    rebalance: 'D', 'W', 'M', or None
    """
    weights = normalize_weights(weights)

    returns = compute_returns(prices)

    weights_df = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        data=np.nan,
    )

    if rebalance is None:
        # Always initialize at the first observation
        weights_df.iloc[0] = [weights[c] for c in weights_df.columns]
        weights_df = weights_df.ffill()
    else:
        pandas_rebalance = rebalance
        if rebalance == "M":
            pandas_rebalance = "ME"

        # ✅ rebalance dates = first actual dates present in the index (not missing calendar dates)
        rebal_idx = returns.groupby(pd.Grouper(freq=pandas_rebalance)).head(1).index

        # ✅ Always initialize at the first observation
        weights_df.iloc[0] = [weights[c] for c in weights_df.columns]

        for dt in rebal_idx:
            weights_df.loc[dt] = [weights[c] for c in weights_df.columns]

        weights_df = weights_df.ffill()

    portfolio_returns = (weights_df * returns).sum(axis=1)
    portfolio_value = base * np.exp(portfolio_returns.cumsum())

    return portfolio_value
