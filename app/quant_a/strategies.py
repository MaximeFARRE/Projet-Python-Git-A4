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

# ---------------------------------------------------------------------
# B. SIGNALS : Trend-Following et Mean-Reversion
# ---------------------------------------------------------------------

def _trend_signal(prices: pd.Series, ma_window: int) -> tuple[pd.Series, pd.Series]:
    """
    Trend-following : long si prix > MA, short si prix < MA.
    """
    ma = _moving_average(prices, ma_window)
    sig = pd.Series(0, index=prices.index, dtype=float)
    sig[prices > ma] = 1.0
    sig[prices < ma] = -1.0
    return sig, ma


def _mean_reversion_signal(prices: pd.Series, mr_window: int, z_threshold: float):
    """
    Mean-Reversion : long si zscore < -seuil, short si zscore > +seuil.
    """
    mu, sigma, z = _z_score(prices, mr_window)
    sig = pd.Series(0, index=prices.index, dtype=float)
    sig[z < -z_threshold] = 1.0
    sig[z > +z_threshold] = -1.0
    return sig, mu, sigma, z

# ---------------------------------------------------------------------
# C. DÉTECTION DE RÉGIME (Volatilité court terme vs long terme)
# ---------------------------------------------------------------------

def _detect_regime(returns: pd.Series,
                   vol_short_window: int,
                   vol_long_window: int,
                   alpha: float) -> pd.Series:
    """
    Régime = 'TREND' si vol_short > alpha * vol_long
           = 'MR' sinon.
    """
    vol_short = _rolling_volatility(returns, vol_short_window)
    vol_long = _rolling_volatility(returns, vol_long_window)

    regime = pd.Series("MR", index=returns.index, dtype=object)
    regime[vol_short > alpha * vol_long] = "TREND"

    return regime, vol_short, vol_long


# ---------------------------------------------------------------------
# D. COMBINAISON DES RÉGIMES
# ---------------------------------------------------------------------

def _combine_signals(regime: pd.Series,
                     trend_sig: pd.Series,
                     mr_sig: pd.Series) -> pd.Series:
    """
    Combine les deux signaux selon le régime.
    """
    pos = pd.Series(0.0, index=regime.index)
    pos[regime == "TREND"] = trend_sig[regime == "TREND"]
    pos[regime == "MR"] = mr_sig[regime == "MR"]
    return pos

# ---------------------------------------------------------------------
# E. STRATÉGIE FINALE : Trend + Mean Reversion (Regime Switching)
# ---------------------------------------------------------------------

def regime_switch_trend_meanrev(
    prices: pd.Series,
    vol_short_window: int = 20,     # volatilité court terme
    vol_long_window: int = 100,     # volatilité long terme
    alpha: float = 1.0,              # seuil de changement de régime
    trend_ma_window: int = 50,       # fenêtre trend-following
    mr_window: int = 20,             # fenêtre MR
    z_threshold: float = 1.0,        # seuil MR
) -> pd.DataFrame:
    """
    Stratégie Regime Switching :
    - Régime TREND : on suit la tendance (long ou short)
    - Régime MR    : mean reversion (contrariant)
    """

    prices = _to_series(prices).sort_index()
    returns = prices.pct_change().fillna(0.0)

    # 1. Régime
    regime, vol_short, vol_long = _detect_regime(
        returns,
        vol_short_window,
        vol_long_window,
        alpha,
    )

    # 2. Trend-following signal
    trend_sig, ma_trend = _trend_signal(prices, trend_ma_window)

    # 3. Mean-reversion signal
    mr_sig, mr_mu, mr_sigma, z = _mean_reversion_signal(prices, mr_window, z_threshold)

    # 4. Position finale selon régime
    position = _combine_signals(regime, trend_sig, mr_sig)

    # 5. Calcul equity curve (via utilitaire existant)
    df = _compute_equity_curve_from_position(prices, position)

    # 6. Colonnes de debug / analyse
    df["regime"] = regime
    df["vol_short"] = vol_short
    df["vol_long"] = vol_long
    df["ma_trend"] = ma_trend
    df["mr_mu"] = mr_mu
    df["mr_sigma"] = mr_sigma
    df["zscore"] = z
    df["trend_signal"] = trend_sig
    df["mr_signal"] = mr_sig

    return df