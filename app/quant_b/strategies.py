# app/quant_b/strategies.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# ✅ Reuse Quant A (single-asset) strategies and orchestrate them for multi-asset
from app.quant_a.strategies import (
    buy_and_hold,
    moving_average_crossover,
    regime_switch_trend_meanrev,
)


def _align_series_to_index(s: pd.Series, index: pd.Index, fill_value: float = 0.0) -> pd.Series:
    """Align a Series to a target index."""
    return s.reindex(index).fillna(fill_value)


def _signals_to_equal_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a signal DataFrame (0/1) into equal-weight long-only weights.
    If no asset is active on a given date => cash (all weights = 0).
    """
    w = signals.astype(float)
    row_sum = w.sum(axis=1).replace(0.0, np.nan)
    w = w.div(row_sum, axis=0).fillna(0.0)
    return w

def _normalize_weight_dict(weights: Dict[str, float], tickers: list[str]) -> Dict[str, float]:
    """Normalize a weight dict over the provided ticker list."""
    total = sum(float(weights.get(t, 0.0)) for t in tickers)
    if total == 0:
        raise ValueError("The sum of weights cannot be zero.")
    return {t: float(weights.get(t, 0.0)) / total for t in tickers}

@dataclass
class StrategyResult:
    """Standard result object for Quant B."""
    weights: pd.DataFrame         # dates x tickers
    signals: pd.DataFrame         # dates x tickers (0/1 long-only)
    details: Dict[str, pd.DataFrame]  # ticker -> Quant A df (useful for debug/plots)

# 3A) BUY & HOLD (reuses Quant A)
def buy_and_hold_multi(
    prices: pd.DataFrame,
    base_weights: Dict[str, float] | None = None,
) -> StrategyResult:
    """
    Multi-asset Buy & Hold.
    - base_weights None => constant equal-weight
    - otherwise => user-provided constant weights (normalized)
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty.")

    tickers = list(prices.columns)
    idx = prices.index

    details: Dict[str, pd.DataFrame] = {}
    # Still call quant_a.buy_and_hold for consistency / debugging (position=1)
    for t in tickers:
        s = prices[t].dropna()
        df_a = buy_and_hold(s)
        details[t] = df_a

    if base_weights is None:
        signals = pd.DataFrame(1.0, index=idx, columns=tickers)
        weights = pd.DataFrame(1.0 / len(tickers), index=idx, columns=tickers)
        return StrategyResult(weights=weights, signals=signals, details=details)

    wdict = _normalize_weight_dict(base_weights, tickers)
    weights = pd.DataFrame(index=idx, columns=tickers, data=0.0)
    for t in tickers:
        weights[t] = wdict[t]

    signals = (weights > 0).astype(int)
    return StrategyResult(weights=weights, signals=signals, details=details)
