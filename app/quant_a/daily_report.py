import datetime as dt
from pathlib import Path

import pandas as pd

from app.quant_a.data_loader import load_cac40_history
from app.quant_a.strategies import buy_and_hold
from app.quant_a.metrics import compute_all_metrics
from app.quant_a.data_loader import load_history
from app.quant_b.data_adapter import load_prices_matrix
from app.quant_b.backtest import compute_portfolio_value_from_weights


def _compute_period_return(prices: pd.Series, start_date: dt.date) -> float:
    """
    Percentage return between the first and last price starting from start_date.
    If there is not enough data, returns NaN.
    """
    if prices.empty:
        return float("nan")

    # Filter from start_date (compare dates without time)
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
    Compute statistics required for the report from a price DataFrame.

    df must contain at least one price series, ideally with 'Open' and 'Close'.
    This function is GENERIC: Quant B can reuse it for any asset
    by providing a DataFrame with the same structure.

    Robust behavior:
    - if df is a Series, it is converted to a DataFrame with a 'Close' column
    - if 'Close' is missing but 'Adj Close' exists, 'Adj Close' is used
    - if neither 'Close' nor 'Adj Close' exists, the first column is used
    - if 'Open' is missing, it is reconstructed from 'Close'
    """

    if df is None or len(df) == 0:
        raise ValueError("Empty price DataFrame in compute_report_stats.")

    # Ensure df is a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name="Close")

    df = df.copy()
    df = df.sort_index()

    # Normalize Close column
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            # Use the first available column as a proxy
            first_col = df.columns[0]
            df["Close"] = df[first_col]

    # Normalize Open column
    if "Open" not in df.columns:
        df["Open"] = df["Close"]

    # Drop rows without Close
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError("Price DataFrame empty after normalization in compute_report_stats.")

    # Reference date = last available market point
    last_ts = df.index[-1]
    as_of_date = last_ts.date()

    # Last point information
    last_row = df.iloc[-1]
    open_price = float(last_row["Open"]) if "Open" in df.columns else float(last_row["Close"])
    close_price = float(last_row["Close"])

    # Daily return (vs previous point)
    closes = df["Close"].astype(float)
    if len(closes) > 1:
        prev_close = float(closes.iloc[-2])
        if prev_close != 0:
            daily_return = (close_price / prev_close - 1) * 100.0
        else:
            daily_return = float("nan")
    else:
        daily_return = float("nan")

    # Periods for week / month / year to date (based on calendar dates)
    week_start_date = as_of_date - dt.timedelta(days=as_of_date.weekday())  # Monday
    month_start_date = as_of_date.replace(day=1)
    year_start_date = as_of_date.replace(month=1, day=1)

    week_return = _compute_period_return(closes, week_start_date)
    month_return = _compute_period_return(closes, month_start_date)
    ytd_return = _compute_period_return(closes, year_start_date)

    # Volatility & max drawdown over vol_window_days (e.g. last 90 days)
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
    Generate a daily report for the CAC 40.

    - Daily data over the last ~365 days
    - Daily info (open, close, daily return)
    - Week-to-date, month-to-date, year-to-date performance
    - Annualized volatility and max drawdown over the last 90 days
    - Report generation timestamp
    """

    # Date preparation (1 year back for YTD / month / week stats)
    today = dt.date.today()
    start_date = today - dt.timedelta(days=365)

    # Load CAC 40 daily data
    df = load_cac40_history(
        start=start_date.strftime("%Y-%m-%d"),
        end=today.strftime("%Y-%m-%d"),
        interval="1d",
    )

    if df is None or df.empty:
        raise ValueError("No data downloaded for CAC 40.")

    # Compute stats using the generic function (reusable by Quant B)
    stats = compute_report_stats(df, vol_window_days=90)

    # Create reports/ directory
    reports_dir = Path("reports") / "quant_a"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # File name based on market date
    as_of_date = stats["as_of_date"]
    filename = reports_dir / f"daily_report_{as_of_date.isoformat()}.txt"

    # Export time (local time)
    export_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt_pct(x: float) -> str:
        return "N/A" if pd.isna(x) else f"{x:.2f} %"

    # Report content
    content = (
        f"===== DAILY REPORT {as_of_date.isoformat()} =====\n"
        f"Report generation time: {export_time}\n\n"
        f"Asset: CAC 40 (^FCHI)\n\n"
        f"--- DAILY ---\n"
        f"Open price  : {stats['open_price']:.2f}\n"
        f"Close price : {stats['close_price']:.2f}\n"
        f"Daily return: {fmt_pct(stats['daily_return_pct'])}\n\n"
        f"--- CURRENT PERIODS ---\n"
        f"Week-to-date performance : {fmt_pct(stats['week_return_pct'])}\n"
        f"Month-to-date performance: {fmt_pct(stats['month_return_pct'])}\n"
        f"Year-to-date performance : {fmt_pct(stats['ytd_return_pct'])}\n\n"
        f"--- RISK (90-day window) ---\n"
        f"Annualized volatility: {fmt_pct(stats['vol_annual_pct'])}\n"
        f"Max drawdown         : {fmt_pct(stats['max_drawdown_pct'])}\n\n"
        f"Data source: Yahoo Finance via yfinance\n"
        f"Automatically generated by app.quant_a.daily_report\n"
    )

    # Save file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename
