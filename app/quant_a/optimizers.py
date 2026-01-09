# app/quant_a/optimizers.py

import pandas as pd

from .strategies import (
    moving_average_crossover,
    regime_switch_trend_meanrev,
)
from .metrics import compute_all_metrics


# =====================================================================
# 1) MOVING AVERAGE CROSSOVER OPTIMIZATION
# =====================================================================

def optimize_moving_average_params(
    prices: pd.Series,
    periods_per_year: int,
):
    """
    Search for (short_window, long_window) parameters for the
    Moving Average Crossover strategy that maximize total return.

    Grid search is intentionally limited to remain fast.
    """
    prices = prices.dropna()
    if prices.empty:
        return None

    best_score = -float("inf")
    best_short = None
    best_long = None
    best_metrics = None

    # Example grid (adjust bounds if needed)
    short_list = [10, 20, 30, 50]
    long_list = [100, 150, 200, 250]

    for short_window in short_list:
        for long_window in long_list:
            if short_window >= long_window:
                continue

            df = moving_average_crossover(
                prices,
                short_window=short_window,
                long_window=long_window,
            )

            metrics = compute_all_metrics(
                equity_curve=df["equity_curve"],
                returns=df["strategy_returns"],
                risk_free_rate=0.0,
                periods_per_year=periods_per_year,
            )

            score = metrics["total_return"]

            if pd.notna(score) and score > best_score:
                best_score = score
                best_short = short_window
                best_long = long_window
                best_metrics = metrics

    if best_short is None or best_long is None:
        return None

    return {
        "short_window": best_short,
        "long_window": best_long,
        "metrics": best_metrics,
    }


# =====================================================================
# 2) REGIME SWITCHING OPTIMIZATION (TREND + MEAN REVERSION)
# =====================================================================

def optimize_regime_switching(
    prices: pd.Series,
    periods_per_year: int,
):
    """
    Automatically optimize regime switching model parameters
    by testing a reduced set of values.
    """

    prices = prices.dropna()
    if prices.empty:
        return None

    # Reasonable grids to keep computation time under control
    vol_short_list = [10, 20, 30]
    vol_long_list = [80, 120, 150]
    alpha_list = [0.9, 1.0, 1.1]
    trend_ma_list = [30, 50, 100]
    mr_window_list = [15, 20, 30]
    z_threshold_list = [0.8, 1.0, 1.2]

    best_score = -float("inf")
    best_params = None
    best_metrics = None

    for vs in vol_short_list:
        for vl in vol_long_list:
            if vl <= vs:
                continue

            for alpha in alpha_list:
                for ma_trend in trend_ma_list:
                    for mr_w in mr_window_list:
                        for z_th in z_threshold_list:

                            df = regime_switch_trend_meanrev(
                                prices,
                                vol_short_window=vs,
                                vol_long_window=vl,
                                alpha=alpha,
                                trend_ma_window=ma_trend,
                                mr_window=mr_w,
                                z_threshold=z_th,
                            )

                            metrics = compute_all_metrics(
                                equity_curve=df["equity_curve"],
                                returns=df["strategy_returns"],
                                risk_free_rate=0.0,
                                periods_per_year=periods_per_year,
                            )

                            score = metrics["total_return"]

                            if pd.notna(score) and score > best_score:
                                best_score = score
                                best_params = (vs, vl, alpha, ma_trend, mr_w, z_th)
                                best_metrics = metrics

    if best_params is None:
        return None

    return {
        "vol_short_window": best_params[0],
        "vol_long_window": best_params[1],
        "alpha": best_params[2],
        "trend_ma_window": best_params[3],
        "mr_window": best_params[4],
        "z_threshold": best_params[5],
        "metrics": best_metrics,
    }
