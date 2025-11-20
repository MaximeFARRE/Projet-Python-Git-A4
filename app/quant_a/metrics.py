# app/quant_a/metrics.py
import pandas as pd
import numpy as np


def total_return(equity_curve: pd.Series) -> float:
    """
    Rendement total sur la période, à partir de la courbe d’équité (base 1.0).
    """
    if equity_curve.empty:
        return np.nan
    return float(equity_curve.iloc[-1] - 1.0)


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Rendement annualisé à partir de la courbe d’équité.
    periods_per_year : 252 pour du daily, 12 pour du monthly, etc.
    """
    if equity_curve.empty:
        return np.nan

    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return np.nan

    total_ret = equity_curve.iloc[-1] - 1.0
    return float((1.0 + total_ret) ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Volatilité annualisée (écart-type des rendements * sqrt(periods_per_year)).
    """
    if returns.empty:
        return np.nan

    vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return float(vol)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratio annualisé.
    risk_free_rate : taux sans risque annualisé (ex: 0.02 pour 2%).
    """
    if returns.empty:
        return np.nan

    avg_return = returns.mean()
    vol = returns.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan

    # Rendement en excès par période
    excess_return_per_period = avg_return - (risk_free_rate / periods_per_year)
    sharpe = (excess_return_per_period / vol) * np.sqrt(periods_per_year)
    return float(sharpe)


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown à partir de la courbe d’équité.
    Retourne une valeur négative (ex: -0.35 pour -35%).
    """
    if equity_curve.empty:
        return np.nan

    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    return float(drawdowns.min())


def compute_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Helper qui renvoie toutes les métriques dans un dict.
    """
    tr = total_return(equity_curve)
    ar = annualized_return(equity_curve, periods_per_year=periods_per_year)
    vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    sharpe = sharpe_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    mdd = max_drawdown(equity_curve)

    return {
        "total_return": tr,
        "annualized_return": ar,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
    }

def compute_trade_metrics(trades: pd.DataFrame) -> dict:
    """
    Calcule des métriques de trading à partir d'un DataFrame de trades
    produit par extract_trades_from_position, contenant au moins :
    - direction ('LONG' / 'SHORT')
    - trade_return (rendement en décimal, ex: 0.05 pour +5%)
    - holding_period_bars (durée du trade en nombre de barres)

    Retourne un dict avec par exemple :
    - n_trades
    - win_rate
    - pct_longs
    - pct_shorts
    - avg_trade_return
    - avg_win_return
    - avg_loss_return
    - avg_holding_period
    """

    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "pct_longs": np.nan,
            "pct_shorts": np.nan,
            "avg_trade_return": np.nan,
            "avg_win_return": np.nan,
            "avg_loss_return": np.nan,
            "avg_holding_period": np.nan,
        }

    n_trades = len(trades)
    n_winning = (trades["trade_return"] > 0).sum()
    win_rate = n_winning / n_trades if n_trades > 0 else np.nan

    n_longs = (trades["direction"] == "LONG").sum()
    n_shorts = (trades["direction"] == "SHORT").sum()

    pct_longs = n_longs / n_trades if n_trades > 0 else np.nan
    pct_shorts = n_shorts / n_trades if n_trades > 0 else np.nan

    avg_trade_return = trades["trade_return"].mean() if n_trades > 0 else np.nan

    wins = trades.loc[trades["trade_return"] > 0, "trade_return"]
    losses = trades.loc[trades["trade_return"] <= 0, "trade_return"]

    avg_win_return = wins.mean() if not wins.empty else np.nan
    avg_loss_return = losses.mean() if not losses.empty else np.nan

    if "holding_period_bars" in trades.columns:
        avg_holding_period = trades["holding_period_bars"].mean()
    else:
        avg_holding_period = np.nan

    return {
        "n_trades": int(n_trades),
        "win_rate": float(win_rate) if win_rate == win_rate else np.nan,
        "pct_longs": float(pct_longs) if pct_longs == pct_longs else np.nan,
        "pct_shorts": float(pct_shorts) if pct_shorts == pct_shorts else np.nan,
        "avg_trade_return": float(avg_trade_return) if avg_trade_return == avg_trade_return else np.nan,
        "avg_win_return": float(avg_win_return) if avg_win_return == avg_win_return else np.nan,
        "avg_loss_return": float(avg_loss_return) if avg_loss_return == avg_loss_return else np.nan,
        "avg_holding_period": float(avg_holding_period) if avg_holding_period == avg_holding_period else np.nan,
    }