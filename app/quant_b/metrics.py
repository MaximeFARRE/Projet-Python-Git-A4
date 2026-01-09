import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice de corrélation des rendements (multi-actifs).
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns doit être un DataFrame non vide.")
    return returns.corr()


def periods_per_year_from_interval(interval: str) -> int:
    """
    Approximation du nombre de périodes par an selon l'intervalle.
    - 1d : 252 jours de bourse
    - 1h : ~252*6.5 heures de bourse (approx)
    - 15m : ~252*6.5*4
    - 5m : ~252*6.5*12
    """
    mapping = {
        "1d": 252,
        "1h": int(252 * 6.5),
        "15m": int(252 * 6.5 * 4),
        "5m": int(252 * 6.5 * 12),
    }
    return mapping.get(interval, 252)


def cumulative_return_from_value(value: pd.Series) -> float:
    """
    Rendement cumulé (ex: 0.12 = +12%) à partir d'une série de valeur (base 100 ou 1).
    """
    if value is None or len(value) < 2:
        raise ValueError("value doit contenir au moins 2 points.")
    return float(value.iloc[-1] / value.iloc[0] - 1.0)


def cagr_from_value(value: pd.Series, periods_per_year: int) -> float:
    """
    CAGR approximatif à partir d'une série de valeur et d'un nombre de périodes/an.
    """
    if value is None or len(value) < 2:
        raise ValueError("value doit contenir au moins 2 points.")
    n_periods = len(value) - 1
    years = n_periods / float(periods_per_year)
    if years <= 0:
        return float("nan")
    total = float(value.iloc[-1] / value.iloc[0])
    return total ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """
    Volatilité annualisée (sigma * sqrt(N)).
    """
    if returns is None or len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def portfolio_metrics(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    portfolio_value: pd.Series,
    interval: str = "1d",
) -> dict:
    """
    Regroupe les metrics principales demandées pour Quant B.
    """
    ppy = periods_per_year_from_interval(interval)

    corr = correlation_matrix(asset_returns)

    # vols actifs
    asset_vols = asset_returns.apply(lambda s: annualized_volatility(s.dropna(), ppy))
    avg_asset_vol = float(np.nanmean(asset_vols.values))

    # portfolio
    port_vol = annualized_volatility(portfolio_returns.dropna(), ppy)
    port_cumret = cumulative_return_from_value(portfolio_value)
    port_cagr = cagr_from_value(portfolio_value, ppy)

    # diversification effect (simple)
    # ratio < 1 => le portefeuille est moins volatil que la moyenne des actifs
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
    }
