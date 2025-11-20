# app/quant_a/strategies.py
import pandas as pd
import numpy as np


def _to_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Force un objet pandas 1D (Series). Si DataFrame (n,1), on squeeze la première colonne.
    """
    if isinstance(x, pd.DataFrame):
        # on prend la première colonne
        return x.iloc[:, 0]
    return x


def _compute_equity_curve_from_position(prices: pd.Series, position: pd.Series) -> pd.DataFrame:
    """
    Utilitaire : à partir d'une série de prix et d'une série de positions (0/1),
    calcule les rendements et la courbe d’équité.
    """

    prices = _to_series(prices).sort_index()
    position = _to_series(position)

    # On s'assure que l'index est aligné
    position = position.reindex(prices.index).fillna(0.0)

    returns = prices.pct_change().fillna(0.0)

    # On applique la position connue au début de la période
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
    prices = _to_series(prices)
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
    prices = _to_series(prices).sort_index()

    if short_window >= long_window:
        raise ValueError("short_window doit être strictement inférieur à long_window")

    ma_short = prices.rolling(window=short_window).mean()
    ma_long = prices.rolling(window=long_window).mean()

    # ma_short et ma_long sont des Series ici
    signal = (ma_short > ma_long).astype(float)
    signal = _to_series(signal)
    signal.name = "position"

    df = _compute_equity_curve_from_position(prices, signal)
    df["ma_short"] = ma_short
    df["ma_long"] = ma_long

    return df

# =====================================================================
# 3) STRATÉGIE REGIME SWITCHING : Trend-Following + Mean-Reversion
# =====================================================================

# ---------------------------------------------------------------------
# A. CALCULS UTILITAIRES : volatilité, MA, z-score
# ---------------------------------------------------------------------

def _rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Volatilité mobile simple (écart-type des rendements).
    """
    return returns.rolling(window=window).std()


def _moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Moyenne mobile simple.
    """
    return prices.rolling(window=window).mean()


def _z_score(prices: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Retourne (mu, sigma, zscore) pour le mean reversion.
    """
    mu = prices.rolling(window).mean()
    sigma = prices.rolling(window).std()
    z = (prices - mu) / sigma
    return mu, sigma, z