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
