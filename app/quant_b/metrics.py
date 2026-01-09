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

def portfolio_metrics(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    portfolio_value: pd.Series,
    interval: str = "1d",
) -> dict:
    """
    Aggregate the main portfolio metrics required for Quant B.
    """
    ppy = periods_per_year_from_interval(interval)

    corr = correlation_matrix(asset_returns)

    # asset volatilities
    asset_vols = asset_returns.apply(lambda s: annualized_volatility(s.dropna(), ppy))
    avg_asset_vol = float(np.nanmean(asset_vols.values))

    # portfolio metrics
    port_vol = annualized_volatility(portfolio_returns.dropna(), ppy)
    port_cumret = cumulative_return_from_value(portfolio_value)
    port_cagr = cagr_from_value(portfolio_value, ppy)
    port_ann_return = annualized_return_from_returns(portfolio_returns.dropna(), ppy)
    port_mdd = max_drawdown_from_value(portfolio_value)
    port_sharpe = sharpe_ratio(portfolio_returns.dropna(), ppy, rf=0.0)

    # diversification effect (simple)
    # ratio < 1 => portfolio is less volatile than the average asset
    div_ratio = float(port_vol / avg_asset_vol) if avg_asset_vol and not np.isnan(avg_asset_vol) else float("nan")

    return {
        "correlation": corr,
        "asset_vols_annualized": asset_vols,
        "portfolio_vol_annualized": port_vol,
        "portfolio_cumulative_return": port_cumret,
        "portfolio_cagr": port_cagr,
        "avg_asset_vol_annualized": avg_asset_vol,
        "diversification_ratio": div_ratio,
        "periods_per_year": ppy,
        "portfolio_return_annualized": port_ann_return,
        "portfolio_max_drawdown": port_mdd,
        "portfolio_sharpe": port_sharpe,
    }


def covariance_matrix_annualized(returns: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    """
    Annualized covariance matrix of log returns.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    cov = returns.cov()
    return cov * float(periods_per_year)


def correlation_distance_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation-based distance matrix (standard clustering form):
    d_ij = sqrt(0.5 * (1 - corr_ij))
    """
    if corr is None or corr.empty:
        raise ValueError("corr must be a non-empty matrix.")
    d = np.sqrt(0.5 * (1.0 - corr.clip(-1, 1)))
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def risk_contributions_from_cov(
    weights: pd.Series,
    cov_annualized: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, float]:
    """
    Risk contribution (variance) by asset:
    - RC_i = w_i * (Sigma w)_i / (w' Sigma w)

    Returns:
      - rc_pct : percentage contributions (sum = 1)
      - rc_abs : absolute contributions (variance share, also sums to 1)
      - port_vol : annualized portfolio volatility
    """
    if weights is None or len(weights) == 0:
        raise ValueError("weights is empty.")
    if cov_annualized is None or cov_annualized.empty:
        raise ValueError("cov_annualized is empty.")

    # align weights
    w = weights.astype(float).copy()
    w = w.reindex(cov_annualized.index).fillna(0.0)

    # if sum = 0 => no investment
    if float(w.abs().sum()) == 0.0:
        rc = pd.Series(0.0, index=cov_annualized.index, name="risk_contrib")
        return rc, rc, 0.0

    sigma = cov_annualized.values
    wv = w.values.reshape(-1, 1)

    port_var = float((wv.T @ sigma @ wv)[0, 0])
    if port_var <= 0:
        rc = pd.Series(0.0, index=cov_annualized.index, name="risk_contrib")
        return rc, rc, 0.0

    mrc = (sigma @ wv).flatten()  # marginal risk contribution to variance
    rc_abs = (w.values * mrc) / port_var
    rc_abs = pd.Series(rc_abs, index=cov_annualized.index, name="rc_abs")

    rc_pct = rc_abs / float(rc_abs.sum()) if float(rc_abs.sum()) != 0 else rc_abs.copy()
    rc_pct.name = "rc_pct"

    port_vol = float(np.sqrt(port_var))
    return rc_pct, rc_abs, port_vol
