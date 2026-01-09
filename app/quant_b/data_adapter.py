import pandas as pd


def normalize_interval(interval: str) -> str:
    """
    Normalize UI interval to yfinance format.
    Quant A often uses "60m" instead of "1h".
    """
    mapping = {
        "1h": "60m",
    }
    return mapping.get(interval, interval)


def _extract_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a yfinance output (possibly OHLCV / MultiIndex) into a price matrix.
    Returns: date index, ticker columns, Close or Adj Close values.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # Standard yfinance MultiIndex case: (Field, Ticker) or (Ticker, Field)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        ohlcv = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        # Detect which level contains the OHLCV fields
        if any(x in ohlcv for x in lvl0.unique()):
            field_level, ticker_level = 0, 1
        elif any(x in ohlcv for x in lvl1.unique()):
            field_level, ticker_level = 1, 0
        else:
            field_level, ticker_level = 0, 1

        fields = df.columns.get_level_values(field_level).unique()
        field = "Adj Close" if "Adj Close" in fields else "Close"

        if field_level == 0:
            out = df[field].copy()
        else:
            out = df.xs(field, axis=1, level=field_level).copy()

        out.columns = out.columns.astype(str)
        out = out.loc[:, ~out.columns.duplicated()]
        out.index = pd.to_datetime(out.index)
        return out.sort_index()

    # Flat columns case: if duplicated => not displayable, raise a clear error
    if df.columns.duplicated().any():
        raise ValueError(
            "Loader returned duplicated columns (flattened OHLCV). "
            "It must return a MultiIndex or directly a Close/Adj Close matrix."
        )

    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_prices_matrix(
    load_history_func,
    tickers: list[str],
    interval: str,
    intraday_days: int = 5,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Calls the existing loader (Quant A) and returns a clean price matrix.
    """
    interval = normalize_interval(interval)

    # Quant A: load_history(ticker=..., interval=...)
    try:
        raw = load_history_func(
            ticker=tickers,
            start=start,
            end=end,
            interval=interval,
            intraday_days=intraday_days,
        )

    except TypeError:
        # Fallback signatures
        raw = load_history_func(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            intraday_days=intraday_days,
        )

    prices = _extract_close_matrix(raw)

    # If the loader already returns a matrix but with extra columns
    if not prices.empty:
        cols = [c for c in prices.columns if c in tickers]
        if cols:
            prices = prices[cols]

    return prices
