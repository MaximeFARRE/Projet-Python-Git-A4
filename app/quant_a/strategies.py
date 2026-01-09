# app/quant_a/strategies.py
import pandas as pd
import numpy as np


def _to_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Force a 1D pandas object (Series).
    If a DataFrame (n,1) is provided, squeeze the first column.
    """
    if isinstance(x, pd.DataFrame):
        # Take the first column
        return x.iloc[:, 0]
    return x


def _compute_equity_curve_from_position(prices: pd.Series, position: pd.Series) -> pd.DataFrame:
    """
    Utility function: from a price series and a position series (0/1),
    compute returns and the equity curve.
    """

    prices = _to_series(prices).sort_index()
    position = _to_series(position)

    # Ensure index alignment
    position = position.reindex(prices.index).fillna(0.0)

    returns = prices.pct_change().fillna(0.0)

    # Apply the position known at the beginning of the period
    strategy_returns = position.shift(1).fillna(0.0) * returns

    equity_curve = (1 + strategy_returns).cumprod()

    df = pd.DataFrame(
        {
            "price": prices,
            "position": position,
            "strategy_returns": strategy_returns,
            "equity_curve": equity_curve,
        }
    )

    return df


def buy_and_hold(prices: pd.Series) -> pd.DataFrame:
    """
    Buy & Hold strategy: fully invested from start to end.
    """
    prices = _to_series(prices)
    position = pd.Series(1.0, index=prices.index, name="position")
    return _compute_equity_curve_from_position(prices, position)


def moving_average_crossover(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.DataFrame:
    """
    Moving Average Crossover strategy:
    - Buy (position = 1) when short MA > long MA
    - Exit the market (position = 0) otherwise.
    """
    prices = _to_series(prices).sort_index()

    if short_window >= long_window:
        raise ValueError("short_window must be strictly lower than long_window")

    ma_short = prices.rolling(window=short_window).mean()
    ma_long = prices.rolling(window=long_window).mean()

    # ma_short and ma_long are Series
    signal = (ma_short > ma_long).astype(float)
    signal = _to_series(signal)
    signal.name = "position"

    df = _compute_equity_curve_from_position(prices, signal)
    df["ma_short"] = ma_short
    df["ma_long"] = ma_long

    return df


# =====================================================================
# 3) REGIME SWITCHING STRATEGY: Trend-Following + Mean-Reversion
# =====================================================================

# ---------------------------------------------------------------------
# A. UTILITY COMPUTATIONS: volatility, MA, z-score
# ---------------------------------------------------------------------

def _rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Simple rolling volatility (standard deviation of returns).
    """
    return returns.rolling(window=window).std()


def _moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Simple moving average.
    """
    return prices.rolling(window=window).mean()


def _z_score(prices: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return (mu, sigma, z-score) for mean reversion.
    """
    mu = prices.rolling(window).mean()
    sigma = prices.rolling(window).std()
    z = (prices - mu) / sigma
    return mu, sigma, z


# ---------------------------------------------------------------------
# B. SIGNALS: Trend-Following and Mean-Reversion
# ---------------------------------------------------------------------

def _trend_signal(prices: pd.Series, ma_window: int) -> tuple[pd.Series, pd.Series]:
    """
    Trend-following: long if price > MA, short if price < MA.
    """
    ma = _moving_average(prices, ma_window)
    sig = pd.Series(0, index=prices.index, dtype=float)
    sig[prices > ma] = 1.0
    sig[prices < ma] = -1.0
    return sig, ma


def _mean_reversion_signal(prices: pd.Series, mr_window: int, z_threshold: float):
    """
    Mean-reversion: long if z-score < -threshold, short if z-score > +threshold.
    """
    mu, sigma, z = _z_score(prices, mr_window)
    sig = pd.Series(0, index=prices.index, dtype=float)
    sig[z < -z_threshold] = 1.0
    sig[z > +z_threshold] = -1.0
    return sig, mu, sigma, z


# ---------------------------------------------------------------------
# C. REGIME DETECTION (short-term vs long-term volatility)
# ---------------------------------------------------------------------

def _detect_regime(
    returns: pd.Series,
    vol_short_window: int,
    vol_long_window: int,
    alpha: float,
) -> pd.Series:
    """
    Regime = 'TREND' if vol_short > alpha * vol_long
           = 'MR' otherwise.
    """
    vol_short = _rolling_volatility(returns, vol_short_window)
    vol_long = _rolling_volatility(returns, vol_long_window)

    regime = pd.Series("MR", index=returns.index, dtype=object)
    regime[vol_short > alpha * vol_long] = "TREND"

    return regime, vol_short, vol_long


# ---------------------------------------------------------------------
# D. REGIME COMBINATION
# ---------------------------------------------------------------------

def _combine_signals(
    regime: pd.Series,
    trend_sig: pd.Series,
    mr_sig: pd.Series,
) -> pd.Series:
    """
    Combine both signals depending on the detected regime.
    """
    pos = pd.Series(0.0, index=regime.index)
    pos[regime == "TREND"] = trend_sig[regime == "TREND"]
    pos[regime == "MR"] = mr_sig[regime == "MR"]
    return pos


# ---------------------------------------------------------------------
# E. FINAL STRATEGY: Trend + Mean Reversion (Regime Switching)
# ---------------------------------------------------------------------

def regime_switch_trend_meanrev(
    prices: pd.Series,
    vol_short_window: int = 20,      # short-term volatility
    vol_long_window: int = 100,      # long-term volatility
    alpha: float = 1.0,              # regime change threshold
    trend_ma_window: int = 50,       # trend-following window
    mr_window: int = 20,             # mean-reversion window
    z_threshold: float = 1.0,        # MR threshold
) -> pd.DataFrame:
    """
    Regime Switching strategy:
    - TREND regime: trend-following (long or short)
    - MR regime   : mean reversion (contrarian)
    """

    prices = _to_series(prices).sort_index()
    returns = prices.pct_change().fillna(0.0)

    # 1. Regime detection
    regime, vol_short, vol_long = _detect_regime(
        returns,
        vol_short_window,
        vol_long_window,
        alpha,
    )

    # 2. Trend-following signal
    trend_sig, ma_trend = _trend_signal(prices, trend_ma_window)

    # 3. Mean-reversion signal
    mr_sig, mr_mu, mr_sigma, z = _mean_reversion_signal(prices, mr_window, z_threshold)

    # 4. Final position based on regime
    position = _combine_signals(regime, trend_sig, mr_sig)

    # 5. Equity curve computation (using existing utility)
    df = _compute_equity_curve_from_position(prices, position)

    # 6. Debug / analysis columns
    df["regime"] = regime
    df["vol_short"] = vol_short
    df["vol_long"] = vol_long
    df["ma_trend"] = ma_trend
    df["mr_mu"] = mr_mu
    df["mr_sigma"] = mr_sigma
    df["zscore"] = z
    df["trend_signal"] = trend_sig
    df["mr_signal"] = mr_sig

    return df


# =====================================================================
# 4) TRADE EXTRACTION FROM A POSITION SERIES
# =====================================================================

def extract_trades_from_position(
    prices: pd.Series,
    position: pd.Series,
) -> pd.DataFrame:
    """
    From a price series and a position series (-1, 0, +1),
    reconstruct the trade history:

    - entry_date
    - exit_date
    - direction (LONG / SHORT)
    - entry_price
    - exit_price
    - holding_period_bars (number of time steps)
    - trade_return (%)

    Assumptions:
    - A trade is opened when position goes from 0 to non-zero,
      or when the sign changes (e.g. +1 -> -1 = close then open).
    - A trade is closed when position goes back to 0
      or when the sign changes.
    """

    prices = _to_series(prices).sort_index()
    position = _to_series(position).reindex(prices.index).fillna(0.0)

    # Direction = sign of the position
    direction = position.apply(lambda x: 0 if x == 0 else (1 if x > 0 else -1))
    prev_direction = direction.shift(1).fillna(0).astype(int)

    trades = []
    open_trade = None  # temporary dict

    idx = prices.index

    for t in idx:
        dir_now = int(direction.loc[t])
        dir_prev = int(prev_direction.loc[t])
        price_now = float(prices.loc[t])

        # --- CASE 1: open trade ---
        if dir_prev == 0 and dir_now != 0:
            open_trade = {
                "entry_date": t,
                "entry_price": price_now,
                "direction": "LONG" if dir_now > 0 else "SHORT",
            }

        # --- CASE 2: close (and possibly reopen) ---
        elif dir_prev != 0:
            # If we go back to 0 or change sign → close current trade
            if dir_now == 0 or dir_now != dir_prev:
                if open_trade is not None:
                    open_trade["exit_date"] = t
                    open_trade["exit_price"] = price_now

                    # Compute trade return
                    if open_trade["direction"] == "LONG":
                        tr = open_trade["exit_price"] / open_trade["entry_price"] - 1.0
                    else:  # SHORT
                        tr = open_trade["entry_price"] / open_trade["exit_price"] - 1.0

                    open_trade["trade_return"] = tr
                    open_trade["holding_period_bars"] = (
                        prices.loc[
                            open_trade["entry_date"]:open_trade["exit_date"]
                        ].shape[0] - 1
                    )

                    trades.append(open_trade)

                open_trade = None

                # If dir_now != 0, immediately open a new trade
                if dir_now != 0:
                    open_trade = {
                        "entry_date": t,
                        "entry_price": price_now,
                        "direction": "LONG" if dir_now > 0 else "SHORT",
                    }

        # else: dir_prev == dir_now (stable position) → nothing to do

    # If a trade is still open at the end, close it at the last price
    if open_trade is not None and "exit_date" not in open_trade:
        last_t = idx[-1]
        last_price = float(prices.iloc[-1])

        open_trade["exit_date"] = last_t
        open_trade["exit_price"] = last_price

        if open_trade["direction"] == "LONG":
            tr = open_trade["exit_price"] / open_trade["entry_price"] - 1.0
        else:
            tr = open_trade["entry_price"] / open_trade["exit_price"] - 1.0

        open_trade["trade_return"] = tr
        open_trade["holding_period_bars"] = (
            prices.loc[
                open_trade["entry_date"]:open_trade["exit_date"]
            ].shape[0] - 1
        )

        trades.append(open_trade)

    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "exit_date",
                "direction",
                "entry_price",
                "exit_price",
                "holding_period_bars",
                "trade_return",
            ]
        )

    trades_df = pd.DataFrame(trades)
    trades_df["trade_return_pct"] = trades_df["trade_return"] * 100.0

    return trades_df
