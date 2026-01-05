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

    df doit contenir au moins une série de prix, avec idéalement 'Open' et 'Close'.
    Cette fonction est GENERIQUE : le Quant B peut l'utiliser pour n'importe quel actif
    en lui passant un DataFrame de même structure.

    Elle est robuste :
    - si df est une Series, on la convertit en DataFrame avec une colonne 'Close'
    - si 'Close' est absent mais 'Adj Close' est présent, on utilise 'Adj Close'
    - si 'Close' est absent et qu'il n'y a pas 'Adj Close', on prend la première colonne
    - si 'Open' est absent, on le reconstruit à partir de 'Close'
    """

    if df is None or len(df) == 0:
        raise ValueError("DataFrame de prix vide dans compute_report_stats.")

    # S'assurer que df est bien un DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name="Close")

    df = df.copy()
    df = df.sort_index()

    # Normalisation de la colonne Close
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            # On prend la première colonne numérique disponible comme proxy
            first_col = df.columns[0]
            df["Close"] = df[first_col]

    # Normalisation de la colonne Open
    if "Open" not in df.columns:
        df["Open"] = df["Close"]

    # On se débarrasse des lignes sans Close
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError("DataFrame de prix vide après normalisation dans compute_report_stats.")

    # Date de référence = dernier point disponible (dernier jour/point de marché)
    last_ts = df.index[-1]
    as_of_date = last_ts.date()

    # Infos du dernier point
    last_row = df.iloc[-1]
    open_price = float(last_row["Open"]) if "Open" in df.columns else float(last_row["Close"])
    close_price = float(last_row["Close"])

    # Rendement journalier (vs point précédent)
    closes = df["Close"].astype(float)
    if len(closes) > 1:
        prev_close = float(closes.iloc[-2])
        if prev_close != 0:
            daily_return = (close_price / prev_close - 1) * 100.0
        else:
            daily_return = float("nan")
    else:
        daily_return = float("nan")

    # Périodes pour semaine / mois / année en cours (basées sur les dates, pas sur la fréquence)
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
    

def generate_daily_report():
    """
    Génère un rapport quotidien sur le CAC 40.

    - Données daily sur les ~365 derniers jours
    - Infos du jour (open, close, rendement journalier)
    - Infos sur la semaine, le mois et l'année en cours
    - Volatilité annualisée et max drawdown sur les 90 derniers jours
    - Heure d'export du rapport
    """

    # Préparation des dates (1 an en arrière pour avoir YTD + mois + semaine)
    today = dt.date.today()
    start_date = today - dt.timedelta(days=365)

    # Charger les données daily du CAC 40
    df = load_cac40_history(
        start=start_date.strftime("%Y-%m-%d"),
        end=today.strftime("%Y-%m-%d"),
        interval="1d",
    )

    if df is None or df.empty:
        raise ValueError("Aucune donnée téléchargée pour le CAC 40.")

    # Calcul des stats avec la fonction générique (réutilisable par Quant B)
    stats = compute_report_stats(df, vol_window_days=90)

    # Création dossier reports/
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Nom du fichier basé sur la date de marché (as_of_date)
    as_of_date = stats["as_of_date"]
    filename = reports_dir / f"daily_report_{as_of_date.isoformat()}.txt"

    # Heure d'export (heure locale de génération du rapport)
    export_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt_pct(x: float) -> str:
        return "N/A" if pd.isna(x) else f"{x:.2f} %"

    # Contenu du rapport
    contenu = (
        f"===== RAPPORT QUOTIDIEN DU {as_of_date.isoformat()} =====\n"
        f"Heure de génération du rapport : {export_time}\n"
        f"\n"
        f"Actif : CAC 40 (^FCHI)\n"
        f"\n"
        f"--- JOURNÉE ---\n"
        f"Prix d'ouverture : {stats['open_price']:.2f}\n"
        f"Prix de clôture : {stats['close_price']:.2f}\n"
        f"Rendement du jour : {fmt_pct(stats['daily_return_pct'])}\n"
        f"\n"
        f"--- PÉRIODES EN COURS ---\n"
        f"Performance semaine en cours : {fmt_pct(stats['week_return_pct'])}\n"
        f"Performance mois en cours   : {fmt_pct(stats['month_return_pct'])}\n"
        f"Performance YTD (année en cours) : {fmt_pct(stats['ytd_return_pct'])}\n"
        f"\n"
        f"--- RISQUE (fenêtre 90 jours) ---\n"
        f"Volatilité annualisée : {fmt_pct(stats['vol_annual_pct'])}\n"
        f"Max drawdown          : {fmt_pct(stats['max_drawdown_pct'])}\n"
        f"\n"
        f"Source des données : Yahoo Finance via yfinance\n"
        f"Généré automatiquement par app.quant_a.daily_report\n"
    )

    # Sauvegarde fichier
    with open(filename, "w", encoding="utf-8") as f:
        f.write(contenu)

    return filename


if __name__ == "__main__":
    path = generate_daily_report()
    print(f"Rapport généré : {path}")
