# app/quant_b/backtest.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_rebalance_freq(rebalance: str | None) -> str | None:
    """
    Streamlit UI -> Pandas resample freq.
    - "M" -> "ME" pour éviter warning pandas
    """
    if rebalance is None:
        return None
    if rebalance == "M":
        return "ME"
    return rebalance


def apply_rebalance(weights_df: pd.DataFrame, rebalance: str | None) -> pd.DataFrame:
    if weights_df is None or weights_df.empty:
        raise ValueError("weights_df vide.")

    w = weights_df.copy().astype(float)

    # normalise (somme=1 ou 0 si cash)
    row_sum = w.sum(axis=1)
    w = w.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)

    # ✅ si pas de rebal : juste hold
    if rebalance is None:
        return w.ffill().fillna(0.0)

    # ✅ ICI freq est TOUJOURS défini
    freq = _normalize_rebalance_freq(rebalance)

    # ✅ dates de rebal = vraies dates de l'index
    rebal_dates = w.groupby(pd.Grouper(freq=freq)).head(1).index

    w2 = w.copy()
    w2.loc[~w2.index.isin(rebal_dates), :] = np.nan
    w2 = w2.ffill().fillna(0.0)

    # re-normalisation
    row_sum2 = w2.sum(axis=1)
    w2 = w2.div(row_sum2.replace(0.0, np.nan), axis=0).fillna(0.0)
    return w2


def compute_portfolio_value_from_weights(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    rebalance: str | None = "M",
    base: float = 100.0,
) -> pd.Series:
    """
    Calcule la valeur portefeuille (base 100) à partir:
    - prices: DataFrame (dates x tickers)
    - weights_df: DataFrame (dates x tickers), poids potentiellement dynamiques
    - rebalance: "D","W","M" ou None (applique un "hold" entre rebal)
    """
    if prices is None or prices.empty:
        raise ValueError("prices vide.")
    if weights_df is None or weights_df.empty:
        raise ValueError("weights_df vide.")

    # aligne colonnes et index
    common_cols = [c for c in prices.columns if c in weights_df.columns]
    if len(common_cols) < 1:
        raise ValueError("Aucune colonne commune entre prices et weights_df.")

    p = prices[common_cols].copy().dropna()
    w = weights_df[common_cols].copy()

    # aligne index
    w = w.reindex(p.index).ffill().fillna(0.0)

    # applique rebalancing
    w = apply_rebalance(w, rebalance=rebalance)

    # log returns
    r = np.log(p / p.shift(1)).dropna()
    w = w.reindex(r.index).ffill().fillna(0.0)

    port_r = (w * r).sum(axis=1)
    value = base * np.exp(port_r.cumsum())
    value.name = "PORTFOLIO"
    return value


def compute_turnover(weights_df: pd.DataFrame) -> pd.Series:
    """
    Turnover simple : 0.5 * sum(|w_t - w_{t-1}|) par date.
    """
    if weights_df is None or weights_df.empty:
        return pd.Series(dtype=float)
    w = weights_df.fillna(0.0)
    tw = 0.5 * (w.diff().abs().sum(axis=1))
    tw.name = "turnover"
    return tw
