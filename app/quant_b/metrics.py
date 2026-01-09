import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix of asset returns (multi-asset).
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    return returns.corr()


def periods_per_year_from_interval(interval: str) -> int:
    """
    Approximate number of periods per year based on the data interval.
    - 1d : 252 trading days
    - 1h : ~252*6.5 trading hours (approx)
    - 15m : ~252*6.5*4
    - 5m : ~252*6.5*12
    """
    mapping = {
        "1d": 252,
        "1h": int(252 * 6.5),
        "60m": int(252 * 6.5),
        "15m": int(252 * 6.5 * 4),
        "5m": int(252 * 6.5 * 12),
    }

    return mapping.get(interval, 252)

def cumulative_return_from_value(value: pd.Series) -> float:
    """
    Cumulative return (e.g. 0.12 = +12%) from a value series (base 100 or 1).
    """
    if value is None or len(value) < 2:
        raise ValueError("value must contain at least 2 observations.")
    return float(value.iloc[-1] / value.iloc[0] - 1.0)


def cagr_from_value(value: pd.Series, periods_per_year: int) -> float:
    """
    Approximate CAGR from a value series and periods per year.
    """
    if value is None or len(value) < 2:
        raise ValueError("value must contain at least 2 observations.")
    n_periods = len(value) - 1
    years = n_periods / float(periods_per_year)
    if years <= 0:
        return float("nan")
    total = float(value.iloc[-1] / value.iloc[0])
    return total ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """
    Annualized volatility (sigma * sqrt(N)).
    """
    if returns is None or len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def annualized_return_from_returns(returns: pd.Series, periods_per_year: int) -> float:
    """
    Approximate annualized return from log-return series.
    Converted via exp(mean * periods_per_year) - 1.
    """
    if returns is None or len(returns) < 2:
        return float("nan")
    mu = float(returns.dropna().mean())
    return float(np.exp(mu * periods_per_year) - 1.0)


def max_drawdown_from_value(value: pd.Series) -> float:
    """
    Maximum drawdown (e.g. -0.25 = -25%) from a value curve.
    """
    if value is None or len(value) < 2:
        return float("nan")
    v = value.dropna()
    running_max = v.cummax()
    dd = v / running_max - 1.0
    return float(dd.min())


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int,
    rf: float = 0.0,
) -> float:
    """
    Approximate annualized Sharpe ratio.
    rf = annualized risk-free rate (e.g. 0.02 for 2%).
    returns is a log-return series.
    """
    if returns is None or len(returns) < 2:
        return float("nan")

    r = returns.dropna()
    if r.std(ddof=1) == 0:
        return float("nan")

    # Convert annualized rf into per-period rf (log approximation)
    rf_per_period = np.log(1.0 + rf) / float(periods_per_year)

    excess = r - rf_per_period
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))
