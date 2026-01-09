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

    # normalise (sum=1 or 0 if cash)
    row_sum = w.sum(axis=1)
    w = w.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)

    # ✅ if not rebal : juste hold
    if rebalance is None:
        return w.ffill().fillna(0.0)

    # ✅ here freq is always defined
    freq = _normalize_rebalance_freq(rebalance)

    # ✅ rebal dates = true index dates
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
    Computes the portfolio value (base 100) from:
    - prices: DataFrame (dates x tickers)
    - weights_df: DataFrame (dates x tickers), potentially dynamic weights
    - rebalance: "D", "W", "M", or None (applies a "hold" between rebalancing dates)

    """
    if prices is None or prices.empty:
        raise ValueError("prices empty.")
    if weights_df is None or weights_df.empty:
        raise ValueError("weights_df empty.")

    # aligns columns and index
    common_cols = [c for c in prices.columns if c in weights_df.columns]
    if len(common_cols) < 1:
        raise ValueError("No common column between prices and weights_df.")

    p = prices[common_cols].copy().dropna()
    w = weights_df[common_cols].copy()

    # align index
    w = w.reindex(p.index).ffill().fillna(0.0)

    # apply rebalancing
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
    Simple turnover  : 0.5 * sum(|w_t - w_{t-1}|) per date.
    """
    if weights_df is None or weights_df.empty:
        return pd.Series(dtype=float)
    w = weights_df.fillna(0.0)
    tw = 0.5 * (w.diff().abs().sum(axis=1))
    tw.name = "turnover"
    return tw
