import os
import datetime as dt
from pathlib import Path

import pandas as pd

from .data_loader import load_cac40_history
from .strategies import buy_and_hold
from .metrics import compute_all_metrics


def generate_daily_report():
    """
    Génère un rapport quotidien simple sur le CAC 40.
    - données daily sur les 90 derniers jours
    - prix open/close du jour
    - volatilité annualisée
    - max drawdown
    """

    # Préparation des dates
    today = dt.date.today()
    start_date = today - dt.timedelta(days=90)

    # Charger les données daily
    df = load_cac40_history(
        start=start_date.strftime("%Y-%m-%d"),
        end=today.strftime("%Y-%m-%d"),
        interval="1d",
    )

    if df is None or df.empty:
        raise ValueError("Aucune donnée téléchargée pour le CAC 40.")

    df = df.sort_index()

    # Récupération infos du jour
    last_row = df.iloc[-1]
    open_price = last_row["Open"]
    close_price = last_row["Close"]

    # Calcul du rendement journalier
    if len(df) > 1:
        prev_close = df["Close"].iloc[-2]
        daily_return = (close_price / prev_close - 1) * 100
    else:
        daily_return = float("nan")

    # Calcul performance (via Buy & Hold)
    prices = df["Close"]
    bh_df = buy_and_hold(prices)
    metrics = compute_all_metrics(
        equity_curve=bh_df["equity_curve"],
        returns=bh_df["strategy_returns"],
        risk_free_rate=0.0,
        periods_per_year=252,
    )

    vol = metrics["annualized_volatility"] * 100
    mdd = metrics["max_drawdown"] * 100

    # Création dossier reports/
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Nom du fichier
    filename = reports_dir / f"daily_report_{today.isoformat()}.txt"

    # Contenu du rapport
    contenu = (
        f"===== RAPPORT QUOTIDIEN DU {today.isoformat()} =====\n"
        f"\n"
        f"Actif : CAC 40 (^FCHI)\n"
        f"\n"
        f"Prix d'ouverture : {open_price:.2f}\n"
        f"Prix de clôture : {close_price:.2f}\n"
        f"Rendement du jour : {daily_return:.2f} %\n"
        f"\n"
        f"Volatilité annualisée (90 jours) : {vol:.2f} %\n"
        f"Max drawdown (90 jours) : {mdd:.2f} %\n"
        f"\n"
        f"Source des données : Yahoo Finance via yfinance\n"
        f"Generé automatiquement par daily_report.py\n"
    )

    # Sauvegarde fichier
    with open(filename, "w", encoding="utf-8") as f:
        f.write(contenu)

    return filename


if __name__ == "__main__":
    path = generate_daily_report()
    print(f"Rapport généré : {path}")
