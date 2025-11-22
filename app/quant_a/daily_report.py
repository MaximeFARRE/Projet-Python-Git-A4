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


def compute_report_stats(df: pd.DataFrame, vol_window_days: int = 90) -> dict:
    """
    Calcule les statistiques nécessaires au rapport à partir d'un DataFrame de prix.

    df doit contenir au moins les colonnes 'Open' et 'Close' et un DatetimeIndex.
    Cette fonction est GENERIQUE : le Quant B peut l'utiliser pour n'importe quel actif
    en lui passant un DataFrame de même structure.
    """

    if df is None or df.empty:
        raise ValueError("DataFrame de prix vide dans compute_report_stats.")

    df = df.sort_index()
    df = df.dropna(subset=["Open", "Close"])

    if df.empty:
        raise ValueError("DataFrame de prix vide après nettoyage dans compute_report_stats.")

    # Date de référence = dernier point disponible (dernier jour de marché)
    last_ts = df.index[-1]
    as_of_date = last_ts.date()

    # Infos du jour
    last_row = df.iloc[-1]
    open_price = float(last_row["Open"])
    close_price = float(last_row["Close"])

    # Rendement journalier (vs veille)
    closes = df["Close"].astype(float)
    if len(closes) > 1:
        prev_close = float(closes.iloc[-2])
        if prev_close != 0:
            daily_return = (close_price / prev_close - 1) * 100.0
        else:
            daily_return = float("nan")
    else:
        daily_return = float("nan")

    # Périodes pour semaine / mois / année en cours
    week_start_date = as_of_date - dt.timedelta(days=as_of_date.weekday())  # lundi de la semaine
    month_start_date = as_of_date.replace(day=1)
    year_start_date = as_of_date.replace(month=1, day=1)

    week_return = _compute_period_return(closes, week_start_date)
    month_return = _compute_period_return(closes, month_start_date)
    ytd_return = _compute_period_return(closes, year_start_date)

    # Volatilité & max drawdown sur vol_window_days (ex: 90 derniers jours)
    vol_start_date = as_of_date - dt.timedelta(days=vol_window_days)
    mask_vol = df.index.date >= vol_start_date
    df_vol = df[mask_vol]

    if len(df_vol) < 2:
        vol_annual = float("nan")
        max_drawdown = float("nan")
    else:
        prices_window = df_vol["Close"].astype(float)
        bh_df = buy_and_hold(prices_window)
        metrics = compute_all_metrics(
            equity_curve=bh_df["equity_curve"],
            returns=bh_df["strategy_returns"],
            risk_free_rate=0.0,
            periods_per_year=252,
        )
        vol_annual = metrics["annualized_volatility"] * 100.0
        max_drawdown = metrics["max_drawdown"] * 100.0

    return {
        "as_of_date": as_of_date,
        "open_price": open_price,
        "close_price": close_price,
        "daily_return_pct": daily_return,
        "week_return_pct": week_return,
        "month_return_pct": month_return,
        "ytd_return_pct": ytd_return,
        "vol_annual_pct": vol_annual,
        "max_drawdown_pct": max_drawdown,
    }