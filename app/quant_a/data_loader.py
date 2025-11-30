import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional

# Ticker historique pour le CAC 40 (compatibilité avec l'ancienne API)
CAC40_TICKER = "^FCHI"


def load_history(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    intraday_days: int = 5,
) -> pd.DataFrame:
    """
    Charge l'historique d'un actif générique via Yahoo Finance.

    - ticker : symbole Yahoo Finance (ex: "^FCHI", "AAPL", "EURUSD=X").
    - start / end : dates au format "YYYY-MM-DD" (pour daily/weekly/monthly).
    - interval : "1d", "1wk", "1mo" pour les données de clôture, ou intraday ("5m", "15m", "60m"...).
    - intraday_days : pour les intervalles intraday, on utilise 'period=X d' plutôt que start/end.

    Remarque :
    - Pour les intervalles intraday, Yahoo limite la profondeur historique accessible.
    """

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}

    if interval in intraday_intervals:
        # Pour l'intraday, on doit utiliser period=...
        data = yf.download(
            ticker,
            period=f"{intraday_days}d",
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    else:
        # Comportement classique pour 1d / 1wk / 1mo
        data = yf.download(
            ticker,
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


def load_cac40_history(
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    intraday_days: int = 5,
) -> pd.DataFrame:
    """
    Wrapper historique pour le CAC 40, conservé pour compatibilité.
    Utilise désormais la fonction générique load_history().
    """
    return load_history(
        ticker=CAC40_TICKER,
        start=start,
        end=end,
        interval=interval,
        intraday_days=intraday_days,
    )


def get_last_cac40_close() -> float:
    """
    Renvoie le dernier cours de clôture du CAC 40.
    Utilise load_cac40_history() avec interval '1d'.
    """
    data = load_cac40_history(interval="1d")
    if data.empty:
        raise ValueError("Aucune donnée retournée pour le CAC 40.")
    return float(data["Close"].iloc[-1])
