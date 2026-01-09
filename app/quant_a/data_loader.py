import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional

# Historical ticker for the CAC 40 (legacy API compatibility)
CAC40_TICKER = "^FCHI"


def load_history(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    intraday_days: int = 5,
) -> pd.DataFrame:
    """
    Load historical data for a generic asset via Yahoo Finance.

    - ticker: Yahoo Finance symbol (e.g. "^FCHI", "AAPL", "EURUSD=X").
    - start / end: dates in "YYYY-MM-DD" format (for daily/weekly/monthly data).
    - interval: "1d", "1wk", "1mo" for closing data, or intraday ("5m", "15m", "60m", etc.).
    - intraday_days: for intraday intervals, use 'period=X d' instead of start/end.

    Note:
    - For intraday intervals, Yahoo Finance limits the available historical depth.
    """

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}

    if interval in intraday_intervals:
        # For intraday data, Yahoo requires the use of period=...
        data = yf.download(
            ticker,
            period=f"{intraday_days}d",
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    else:
        # Standard behavior for 1d / 1wk / 1mo
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

    if data is None or data.empty:
        raise ValueError(
            f"No data returned by Yahoo Finance for ticker='{ticker}', interval='{interval}'. "
            "Check the ticker symbol, the interval, or try again later."
        )

    # Some versions of yfinance return a MultiIndex for columns:
    # e.g. ('Open', '^FCHI'), ('High', '^FCHI'), ...
    # Flatten the columns by keeping only the first level:
    # "Open", "High", "Low", "Close", ...
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

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
    Historical wrapper for the CAC 40, kept for compatibility.
    Now relies on the generic load_history() function.
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
    Return the latest closing price of the CAC 40.
    Uses load_cac40_history() with a '1d' interval.
    """
    data = load_cac40_history(interval="1d")
    if data.empty:
        raise ValueError("No data returned for the CAC 40.")
    return float(data["Close"].iloc[-1])
