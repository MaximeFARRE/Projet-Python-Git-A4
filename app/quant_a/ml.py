# app/quant_a/ml.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(prices: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        # Take the first column
        s = prices.iloc[:, 0]
    else:
        s = prices
    s = s.dropna().astype(float).sort_index()
    return s


def _infer_step(prices: pd.Series) -> pd.Timedelta:
    """
    Infer a reasonable time step from the median of index differences.
    Fallback: 1 day.
    """
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return pd.Timedelta(days=1)

    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(days=1)

    step = diffs.median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return pd.Timedelta(days=1)
    return step


def predict_next_return_linear(
    prices: pd.Series | pd.DataFrame,
    window: int = 20,
    train_min_rows: int = 80,
) -> dict:
    """
    Option 1 (ML):
    Linear regression (numpy) to predict the return of the next time step.

    Features at time t (to predict return at t+1):
      - ret(t)
      - mean_ret(window)
      - vol_ret(window)

    Returns a robust dictionary (NaN if not enough data).
    """
    p = _as_series(prices)
    if len(p) < max(train_min_rows, window + 5):
        return {
            "pred_next_return": np.nan,
            "pred_next_return_pct": np.nan,
            "direction": None,
            "r2_train": np.nan,
            "n_train": 0,
        }

    ret = p.pct_change()

    df = pd.DataFrame(
        {
            "ret": ret,
            "mean_ret": ret.rolling(window).mean(),
            "vol": ret.rolling(window).std(),
        }
    ).dropna()

    # y = next-step return
    y = df["ret"].shift(-1)
    X = df[["ret", "mean_ret", "vol"]]
    Xy = pd.concat([X, y.rename("y")], axis=1).dropna()

    if len(Xy) < train_min_rows:
        return {
            "pred_next_return": np.nan,
            "pred_next_return_pct": np.nan,
            "direction": None,
            "r2_train": np.nan,
            "n_train": int(len(Xy)),
        }

    X = Xy[["ret", "mean_ret", "vol"]].values
    y = Xy["y"].values

    # Add intercept
    X_design = np.column_stack([np.ones(len(X)), X])

    # Least squares
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    # Training R² (informational)
    y_hat = X_design @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Next-step prediction using last feature row
    last_feat = Xy.iloc[-1][["ret", "mean_ret", "vol"]].values.astype(float)
    last_design = np.array([1.0, *last_feat])
    pred = float(last_design @ beta)

    return {
        "pred_next_return": pred,
        "pred_next_return_pct": pred * 100.0,
        "direction": "UP" if pred > 0 else ("DOWN" if pred < 0 else "FLAT"),
        "r2_train": float(r2) if r2 == r2 else np.nan,
        "n_train": int(len(Xy)),
    }


def forecast_price_trend(
    prices: pd.Series | pd.DataFrame,
    horizon_steps: int = 20,
    use_log: bool = True,
) -> pd.Series:
    """
    Option 2 (light ML):
    Linear regression of price over time (log-price by default),
    then projection over horizon_steps.

    Returns a Series indexed by future dates.
    """
    p = _as_series(prices)
    if len(p) < 10 or horizon_steps <= 0:
        return pd.Series(dtype=float)

    # Time variable t = 0..n-1
    n = len(p)
    t = np.arange(n, dtype=float)

    y = np.log(p.values) if use_log else p.values
    X_design = np.column_stack([np.ones(n), t])

    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    # Generate future points
    step = _infer_step(p)
    last_ts = p.index[-1]

    future_index = pd.DatetimeIndex(
        [last_ts + step * (i + 1) for i in range(horizon_steps)]
    )
    t_future = np.arange(n, n + horizon_steps, dtype=float)

    y_future = np.column_stack([np.ones(horizon_steps), t_future]) @ beta

    pred_prices = np.exp(y_future) if use_log else y_future
    return pd.Series(pred_prices, index=future_index, name="forecast")
