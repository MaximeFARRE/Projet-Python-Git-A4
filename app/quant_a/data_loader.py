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
) -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

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
