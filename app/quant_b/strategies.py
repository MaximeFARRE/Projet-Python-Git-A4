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

# ============================================================
# 3B) TREND & REGIME STRATEGIES (reuse Quant A)
# ============================================================
def ma_crossover_multi(
    prices: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    allocation_rule: str = "Equal-weight",
    alloc_vol_window: int = 20,
) -> StrategyResult:

    """
    Apply the Quant A Moving Average Crossover strategy to each asset.
    - signal = Quant A position (0/1)
    - allocation = equal-weight across assets with signal=1
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty.")

    tickers = list(prices.columns)
    idx = prices.index

    signals_dict: Dict[str, pd.Series] = {}
    details: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        s = prices[t].dropna()
        if len(s) < max(short_window, long_window) + 5:
            # not enough history => never invest in this asset
            signals_dict[t] = pd.Series(0.0, index=idx)
            details[t] = pd.DataFrame(index=s.index)
            continue

        df_a = moving_average_crossover(s, short_window=short_window, long_window=long_window)
        details[t] = df_a

        pos = df_a["position"].astype(float)
        signals_dict[t] = _align_series_to_index(pos, idx, fill_value=0.0)

    signals = pd.DataFrame(signals_dict, index=idx).fillna(0.0)
    # long-only => already 0/1
    if allocation_rule == "Inverse-vol":
        weights = _signals_to_inverse_vol_weights(prices, signals, vol_window=alloc_vol_window)
    else:
        weights = _signals_to_equal_weights(signals)

    return StrategyResult(weights=weights, signals=signals.astype(int), details=details)


def regime_switch_multi(
    prices: pd.DataFrame,
    vol_short_window: int = 20,
    vol_long_window: int = 100,
    alpha: float = 1.0,
    trend_ma_window: int = 50,
    mr_window: int = 20,
    z_threshold: float = 1.0,
    long_only: bool = True,
    allocation_rule: str = "Equal-weight",
    alloc_vol_window: int = 20,
) -> StrategyResult:
    """
    Apply the Quant A regime_switch_trend_meanrev strategy to each asset.
    Quant A may output positions -1/0/1.
    - if long_only=True: convert to 0/1 signal via (pos > 0)
    - otherwise: keep -1/0/1 (allocation must then handle long/short)
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty.")

    tickers = list(prices.columns)
    idx = prices.index

    signals_dict: Dict[str, pd.Series] = {}
    details: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        s = prices[t].dropna()
        if len(s) < max(vol_long_window, trend_ma_window, mr_window) + 5:
            signals_dict[t] = pd.Series(0.0, index=idx)
            details[t] = pd.DataFrame(index=s.index)
            continue

        df_a = regime_switch_trend_meanrev(
            s,
            vol_short_window=vol_short_window,
            vol_long_window=vol_long_window,
            alpha=alpha,
            trend_ma_window=trend_ma_window,
            mr_window=mr_window,
            z_threshold=z_threshold,
        )
        details[t] = df_a

        pos = df_a["position"].astype(float)

        if long_only:
            sig = (pos > 0).astype(float)
        else:
            sig = pos

        signals_dict[t] = _align_series_to_index(sig, idx, fill_value=0.0)

    signals = pd.DataFrame(signals_dict, index=idx).fillna(0.0)

    if long_only:
        weights = _signals_to_equal_weights(signals)
        return StrategyResult(weights=weights, signals=signals.astype(int), details=details)

    if allocation_rule == "Inverse-vol":
        weights = _signals_to_inverse_vol_weights(prices, signals, vol_window=alloc_vol_window)
    else:
        weights = _signals_to_equal_weights(signals)

    return StrategyResult(weights=weights, signals=signals.astype(int), details=details)

# ============================================================
# 4) SINGLE ROUTER (convenient for the UI)
# ============================================================
def compute_strategy_weights(
    prices: pd.DataFrame,
    strategy: str,
    params: Dict[str, Any] | None = None,
    base_weights: Dict[str, float] | None = None,
) -> StrategyResult:
    """
    Single entry point:
    strategy in {"Buy & Hold", "MA Crossover", "Regime Switch"}
    """
    params = params or {}

    if strategy == "Buy & Hold":
        return buy_and_hold_multi(prices, base_weights=base_weights)

    if strategy == "MA Crossover":
        return ma_crossover_multi(
            prices,
            short_window=int(params.get("short_window", 50)),
            long_window=int(params.get("long_window", 200)),
            allocation_rule=str(params.get("allocation_rule", "Equal-weight")),
            alloc_vol_window=int(params.get("alloc_vol_window", 20)),
        )

    if strategy == "Regime Switch":
        return regime_switch_multi(
            prices,
            vol_short_window=int(params.get("vol_short_window", 20)),
            vol_long_window=int(params.get("vol_long_window", 100)),
            alpha=float(params.get("alpha", 1.0)),
            trend_ma_window=int(params.get("trend_ma_window", 50)),
            mr_window=int(params.get("mr_window", 20)),
            z_threshold=float(params.get("z_threshold", 1.0)),
            long_only=bool(params.get("long_only", True)),
            allocation_rule=str(params.get("allocation_rule", "Equal-weight")),
            alloc_vol_window=int(params.get("alloc_vol_window", 20)),
        )

    raise ValueError(f"Unknown strategy: {strategy}")

def _signals_to_inverse_vol_weights(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    vol_window: int = 20,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Inverse-volatility allocation across active assets (signals=1).
    w_i(t) ∝ signal_i(t) * 1/vol_i(t)
    """
    rets = np.log(prices / prices.shift(1)).fillna(0.0)
    vol = rets.rolling(vol_window).std().replace(0.0, np.nan)

    inv_vol = 1.0 / (vol + eps)
    raw = signals.astype(float) * inv_vol
    row_sum = raw.sum(axis=1).replace(0.0, np.nan)
    w = raw.div(row_sum, axis=0).fillna(0.0)
    return w
