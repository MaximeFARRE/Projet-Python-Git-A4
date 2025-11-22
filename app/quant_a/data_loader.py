# app/quant_a/data_loader.py
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional

CAC40_TICKER = "^FCHI"


def load_cac40_history(
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    intraday_days: int = 5,
) -> pd.DataFrame:
    """
    Charge l'historique du CAC 40 via Yahoo Finance.

    - Pour les intervalles journaliers / hebdo / mensuels ("1d", "1wk", "1mo"),
      on utilise start / end comme avant.
    - Pour les intervalles intraday ("1m", "2m", "5m", ...),
      Yahoo impose d'utiliser un paramètre `period` plutôt que start/end.
      On récupère alors les `intraday_days` derniers jours.

    Remarque :
    - Le "rafraîchissement toutes les 5 minutes" ne dépend pas de cette fonction :
      c'est l'application (Streamlit) qui doit rappeler cette fonction
      au moins toutes les 5 minutes pour obtenir des données mises à jour.
    """

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # Intervalles intraday supportés par yfinance
    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}

    if interval in intraday_intervals:
        # Pour l'intraday, on doit utiliser period=...
        data = yf.download(
            CAC40_TICKER,
            period=f"{intraday_days}d",
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    else:
        # Comportement historique pour 1d / 1wk / 1mo
        data = yf.download(
            CAC40_TICKER,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

    if data.empty:
        return data

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    return data



def get_last_cac40_close() -> float:
    data = load_cac40_history(interval="1d")
    if data.empty:
        raise ValueError("Aucune donnée retournée pour le CAC40.")
    return float(data["Close"].iloc[-1])
