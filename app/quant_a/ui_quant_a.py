# app/quant_a/ui_quant_a.py

# ===================== 1. IMPORTS & UTILITY FUNCTIONS =====================

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st
from .ml import predict_next_return_linear, forecast_price_trend
import streamlit as st
from .data_loader import load_history, load_cac40_history
from .universe import get_asset_classes, get_assets_by_class, get_default_asset

from .strategies import (
    buy_and_hold,
    moving_average_crossover,
    regime_switch_trend_meanrev,
    extract_trades_from_position,
)

from .metrics import compute_all_metrics, compute_trade_metrics
from .optimizers import (
    optimize_moving_average_params,
    optimize_regime_switching,
)
import plotly.graph_objects as go


def _get_periods_per_year(interval: str) -> int:
    """
    Return the number of periods per year based on the Yahoo interval.
    Intraday approximations:
    - 5m  : ~78 bars/day * 252 days ≈ 19,656
    - 15m : ~26 bars/day * 252 days ≈ 6,552
    - 60m : ~6.5 bars/day * 252 days ≈ 1,638
    """
    if interval == "1d":
        return 252
    if interval == "1wk":
        return 52
    if interval == "1mo":
        return 12
    if interval == "5m":
        return 19656
    if interval == "15m":
        return 6552
    if interval == "60m":
        return 1638
    # fallback
    return 252


def _build_comparison_messages(
    benchmark_metrics: dict,
    strategy_metrics: dict,
) -> list[str]:
    """
    Generate textual comparison messages between the strategy
    and Buy & Hold.
    """
    messages: list[str] = []

    tr_bh = benchmark_metrics["total_return"]
    tr_strat = strategy_metrics["total_return"]

    mdd_bh = benchmark_metrics["max_drawdown"]  # negative
    mdd_strat = strategy_metrics["max_drawdown"]

    sharpe_bh_val = benchmark_metrics["sharpe_ratio"]
    sharpe_strat_val = strategy_metrics["sharpe_ratio"]

    # Outperformance (return)
    if pd.notna(tr_bh) and pd.notna(tr_strat):
        if tr_strat > tr_bh:
            messages.append("✅ The strategy **outperforms** Buy & Hold in terms of total return.")
        elif tr_strat < tr_bh:
            messages.append("⚠️ The strategy **underperforms** Buy & Hold in terms of total return.")
        else:
            messages.append("ℹ️ The strategy has a total return **equivalent** to Buy & Hold.")

    # Max drawdown (compare magnitude)
    if pd.notna(mdd_bh) and pd.notna(mdd_strat):
        if abs(mdd_strat) < abs(mdd_bh):
            messages.append(
                "✅ The strategy has a **lower max drawdown** than Buy & Hold "
                "(better protection against losses)."
            )
        elif abs(mdd_strat) > abs(mdd_bh):
            messages.append(
                "⚠️ The strategy has a **higher max drawdown** than Buy & Hold "
                "(larger maximum loss risk)."
            )
        else:
            messages.append("ℹ️ The strategy has a max drawdown **similar** to Buy & Hold.")

    # Sharpe ratio (risk-adjusted performance)
    if pd.notna(sharpe_bh_val) and pd.notna(sharpe_strat_val):
        if sharpe_strat_val > sharpe_bh_val:
            messages.append(
                "✅ The strategy has a **better Sharpe ratio** than Buy & Hold "
                "(better risk-adjusted performance)."
            )
        elif sharpe_strat_val < sharpe_bh_val:
            messages.append(
                "⚠️ The strategy has a **lower Sharpe ratio** than Buy & Hold "
                "(worse risk-adjusted performance)."
            )
        else:
            messages.append("ℹ️ The strategy has a Sharpe ratio **close** to Buy & Hold.")
    else:
        messages.append(
            "ℹ️ The Sharpe ratio is not available for one of the two series "
            "(insufficient data or zero volatility)."
        )

    return messages


def _normalize_base1(s: pd.Series) -> pd.Series:
    """Normalize a series to base 1 using the first non-NaN point."""
    s = s.astype(float).copy()
    first_valid = s.dropna().iloc[0] if not s.dropna().empty else None
    if first_valid is None or first_valid == 0:
        return s * float("nan")
    return s / first_valid


def _plot_price_vs_equity_base1(
    dates,
    price_base1: pd.Series,
    equity_base1: pd.Series,
    asset_name: str,
    strategy_name: str,
):
    """Compliant chart: base-1 price + base-1 strategy equity on a single figure."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=price_base1,
            mode="lines",
            name=f"Price {asset_name} (base 1)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_base1,
            mode="lines",
            name=f"Strategy: {strategy_name} (base 1)",
        )
    )

    fig.update_layout(
        title="Main chart (compliant): Price vs Strategy (base 1)",
        xaxis_title="Date",
        yaxis_title="Value (base 1)",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig

# ===================== 2. MAIN UI FUNCTION =====================


def render_quant_a_page():
    # ---------- 2.1. TITLE & INTRO ----------
    st.title("Quant A – Univariate backtest (CAC 40)")

    st.markdown(
        """
        This module acts as the analytics engine for **Quant A**: it backtests
        systematic strategies on the **CAC 40** using Yahoo Finance data.

        Main features:

        - Selection of the **time range** and **frequency** (daily / weekly / monthly),
        - Strategy selection:
          - Buy & Hold,
          - Moving average crossover,
          - Regime Switching (Trend + Mean-Reversion),
        - **Automatic optimization** of selected parameters (moving averages, regime),
        - Visualization:
          - price curve, equity, drawdown,
          - volatility regimes and trading signals,
        - Analysis:
          - performance metrics (Sharpe, volatility, drawdown…),
          - trading metrics (number of trades, win rate, longs vs shorts),
          - detailed trade history.
        """
    )

    st.caption(
        "The results and signals produced here can be reused by Quant B "
        "to build a multi-strategy portfolio."
    )

    # ---------- 2.2. DATA CONTROLS (SIDEBAR) ----------
    with st.sidebar:
        # ---------- BLOCK 2.x: ASSET SELECTION ----------
        st.header("Analyzed asset")

        asset_classes = get_asset_classes()
        default_asset = get_default_asset()

        # Asset class (Indices, Forex, Equities, Commodities...)
        try:
            default_class_index = asset_classes.index(default_asset.asset_class)
        except ValueError:
            default_class_index = 0

        selected_class = st.selectbox(
            "Asset class",
            options=asset_classes,
            index=default_class_index,
        )

        # Assets within this class
        assets_in_class = get_assets_by_class(selected_class)
        asset_names = [a.name for a in assets_in_class]

        if not assets_in_class:
            st.error("No asset is defined for this class.")
            st.stop()

        # Specific asset
        # Default = first asset in the selected class
        selected_asset_index = 0
        # If the default class contains the default asset, try to preselect it
        for i, a in enumerate(assets_in_class):
            if a.ticker == default_asset.ticker:
                selected_asset_index = i
                break

        selected_asset_name = st.selectbox(
            "Asset",
            options=asset_names,
            index=selected_asset_index,
        )

        # Retrieve the corresponding Asset object
        selected_asset = assets_in_class[asset_names.index(selected_asset_name)]

        st.caption(
            f"Selected asset: **{selected_asset.name}** "
            f"(Yahoo ticker: `{selected_asset.ticker}`)"
        )

        st.markdown("---")

        # ---------- BLOCK 2.x: DATA PARAMETERS ----------
        st.header("Data parameters")

        period_choice = st.selectbox(
            "Data frequency",
            options=[
                "Daily (1d)",
                "Weekly (1wk)",
                "Monthly (1mo)",
                "Intraday 5 minutes (5m)",
                "Intraday 15 minutes (15m)",
                "Intraday 1 hour (60m)",
            ],
            index=0,
        )

        interval_map = {
            "Daily (1d)": "1d",
            "Weekly (1wk)": "1wk",
            "Monthly (1mo)": "1mo",
            "Intraday 5 minutes (5m)": "5m",
            "Intraday 15 minutes (15m)": "15m",
            "Intraday 1 hour (60m)": "60m",
        }
        interval = interval_map[period_choice]
        if st.button("🔄 Reload now"):
            # Force reload on the next run
            st.session_state.cached_df = None
            st.session_state.last_load_time = None

        periods_per_year = _get_periods_per_year(interval)

        today = dt.date.today()
        default_start = today - dt.timedelta(days=365 * 5)

        start_date, end_date = st.date_input(
            "Study period (backtest)",
            value=(default_start, today),
        )

        if start_date >= end_date:
            st.error("Start date must be strictly earlier than end date.")
            st.stop()

        st.markdown("---")

    # ---------- 2.3. DATA LOADING ----------
    # Cache data for automatic refresh every 5 minutes

    now = dt.datetime.now()

    # Keep dates as strings for easy comparisons
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    current_ticker = selected_asset.ticker

    # Initialize cache keys in the session if needed
    if "cached_df" not in st.session_state:
        st.session_state.cached_df = None
    if "last_load_time" not in st.session_state:
        st.session_state.last_load_time = None
    if "last_interval" not in st.session_state:
        st.session_state.last_interval = None
    if "last_start_date" not in st.session_state:
        st.session_state.last_start_date = None
    if "last_end_date" not in st.session_state:
        st.session_state.last_end_date = None
    if "last_ticker" not in st.session_state:
        st.session_state.last_ticker = None

    need_reload = False

    # 1) If no cached data, we must load
    if st.session_state.cached_df is None or st.session_state.cached_df.empty:
        need_reload = True
    else:
        # 2) If user changes period, interval, or ticker, reload
        if (
            st.session_state.last_interval != interval
            or st.session_state.last_start_date != start_str
            or st.session_state.last_end_date != end_str
            or st.session_state.last_ticker != current_ticker
        ):
            need_reload = True
        else:
            # 3) Otherwise, reload if more than 5 minutes have passed
            last_time = st.session_state.last_load_time
            if last_time is None or (now - last_time).total_seconds() > 300:
                need_reload = True

    if need_reload:
        with st.spinner(f"Loading data for {selected_asset.name} ({current_ticker})..."):
            try:
                df = load_history(
                    ticker=current_ticker,
                    start=start_str,
                    end=end_str,
                    interval=interval,
                )
            except Exception as e:
                st.error(f"Unable to load data: {e}")
                st.stop()

        st.session_state.cached_df = df
        st.session_state.last_load_time = now
        st.session_state.last_interval = interval
        st.session_state.last_start_date = start_str
        st.session_state.last_end_date = end_str
        st.session_state.last_ticker = current_ticker
    else:
        df = st.session_state.cached_df

    if df is None or df.empty:
        st.warning(f"No data available for {selected_asset.name} over this period.")
        return

    # Try to retrieve a clean closing price series
    if "Close" in df.columns:
        prices = df["Close"].astype(float).copy()
    elif "Adj Close" in df.columns:
        prices = df["Adj Close"].astype(float).copy()
    else:
        # fallback: first numeric column
        numeric_cols = df.select_dtypes(include="number")
        if numeric_cols.shape[1] == 0:
            st.error("Unable to find a price series in the downloaded data.")
            return
        prices = numeric_cols.iloc[:, 0].astype(float).copy()

    prices = prices.sort_index()
    prices.name = f"{selected_asset.name} (Close)"

    # ---------- ML BONUS (OPTIONAL) ----------
    with st.expander(" ML – Simple prediction (optional)", expanded=False):
        st.caption(
            "Goal: demonstrate the integration of a simple ML pipeline into the dashboard "
            "(no performance objective)."
        )

        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            ml_window = st.slider("Feature window (ret/vol)", 10, 120, 20, step=5)
        with col_ml2:
            ml_horizon = st.slider("Price projection horizon (time steps)", 5, 60, 20, step=5)

        # Option 1: predict next-step return
        try:
            pred = predict_next_return_linear(prices, window=int(ml_window))
            pred_pct = pred["pred_next_return_pct"]
            direction = pred["direction"]
            r2 = pred["r2_train"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted return (next step)", "N/A" if pd.isna(pred_pct) else f"{pred_pct:.2f} %")
            c2.metric("Direction", "N/A" if direction is None else direction)
            c3.metric("R² (train, indicative)", "N/A" if pd.isna(r2) else f"{r2:.2f}")

        except Exception as e:
            st.warning(f"ML return: unable to compute ({e}).")

        # Option 2: project future price (trend)
        try:
            fc = forecast_price_trend(prices, horizon_steps=int(ml_horizon), use_log=True)
            if fc.empty:
                st.info("Not enough data to project the price.")
            else:
                # Plotly display (dashed future curve)
                hist = prices.dropna().astype(float)

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="History"))
                fig_fc.add_trace(
                    go.Scatter(
                        x=fc.index,
                        y=fc.values,
                        mode="lines",
                        name="Projection (trend)",
                        line=dict(dash="dash"),
                    )
                )
                fig_fc.update_layout(
                    title=f"Simple price projection – {selected_asset.name}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=450,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_fc, use_container_width=True)

        except Exception as e:
            st.warning(f"ML price projection: unable to compute ({e}).")

    # ---------- 2.3.2 PRICE EXPLORER – GLOBAL VIEW ----------

    st.subheader("Price explorer – global view")

    # Display horizon selection (independent from the backtest)
    display_horizon = st.radio(
        "Display horizon",
        options=["1M", "3M", "6M", "1Y", "3Y", "MAX"],
        index=5,  # MAX by default
        horizontal=True,
    )

    # Build a sub-DataFrame for display, based on the full df
    last_date = df.index[-1].date()

    if display_horizon == "MAX":
        df_display = df.copy()
    else:
        horizon_days_map = {
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "3Y": 365 * 3,
        }
        nb_days = horizon_days_map[display_horizon]
        cutoff_date = last_date - dt.timedelta(days=nb_days)
        mask = df.index.date >= cutoff_date
        df_display = df[mask].copy()
        if df_display.empty:
            df_display = df.copy()

    # ---------- Display: candlesticks if possible, otherwise a line chart ----------

    has_ohlc = {"Open", "High", "Low", "Close"}.issubset(df_display.columns)

    if has_ohlc:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_display.index,
                    open=df_display["Open"].astype(float),
                    high=df_display["High"].astype(float),
                    low=df_display["Low"].astype(float),
                    close=df_display["Close"].astype(float),
                    name="Price",
                )
            ]
        )

        # Slightly thicker candles for readability
        fig.update_traces(
            increasing_line_width=2,
            decreasing_line_width=2,
            selector=dict(type="candlestick"),
        )

        fig.update_layout(
            title=f"Price of {selected_asset.name} – global view",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(
                rangebreaks=[
                    # Hide non-trading days between Saturday and Monday
                    dict(bounds=["sat", "mon"]),
                ]
            ),
            xaxis_rangeslider_visible=False,
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        # Clean fallback: simple close line chart
        # Retrieve the best available price series
        if "Close" in df_display.columns:
            closes_display = df_display["Close"].astype(float)
        elif "Adj Close" in df_display.columns:
            closes_display = df_display["Adj Close"].astype(float)
        else:
            numeric_cols = df_display.select_dtypes(include="number")
            if numeric_cols.shape[1] == 0:
                st.error("Unable to find a price series for display.")
                return
            closes_display = numeric_cols.iloc[:, 0].astype(float)

        st.line_chart(closes_display)
        st.info(
            "Full OHLC data (Open/High/Low/Close) is not available "
            f"for {selected_asset.name} at this interval. Displaying a simple line chart instead."
        )

    # ---------- 2.4. STRATEGY CONTROLS + OPTIMIZATION BUTTON ----------
    st.subheader("Strategy parameters")

    strategy_name = st.selectbox(
    "Strategy selection",
    ["Buy & Hold",
     "Moving Average Crossover",
     "Regime Switching (Trend + Mean-Reversion)"],
)


    short_window = None
    long_window = None

    if strategy_name == "Moving Average Crossover":
        st.markdown("Configure the **moving average windows**:")

        # Use session_state so sliders can be updated automatically
        if "short_window" not in st.session_state:
            st.session_state.short_window = 20
        if "long_window" not in st.session_state:
            st.session_state.long_window = 100

        col1, col2 = st.columns(2)

        with col1:
            short_window = st.slider(
                "Short moving average window",
                min_value=5,
                max_value=100,
                value=st.session_state.short_window,
                step=1,
                key="short_window_slider",
            )
        with col2:
            long_window = st.slider(
                "Long moving average window",
                min_value=20,
                max_value=300,
                value=st.session_state.long_window,
                step=5,
                key="long_window_slider",
            )

        # Sync slider values -> session_state
        st.session_state.short_window = short_window
        st.session_state.long_window = long_window

        # Optimization button
        if st.button("🔍 Optimize moving averages over the period"):
            with st.spinner("Searching for the best moving average parameters..."):
                result = optimize_moving_average_params(prices, periods_per_year)

            if result is None:
                st.error("Unable to find optimal parameters (insufficient data?).")
            else:
                best_short = result["short_window"]
                best_long = result["long_window"]
                best_metrics = result["metrics"]

                # Update sliders via session_state
                st.session_state.short_window = best_short
                st.session_state.long_window = best_long

                st.success(
                    f"Best parameters found over the period: "
                    f"Short MA = {best_short}, Long MA = {best_long} "
                    f"(total return ≈ {best_metrics['total_return'] * 100:.2f} %)."
                )

        if short_window >= long_window:
            st.warning("The short window must be strictly lower than the long window.")
            st.stop()

    # ===================== REGIME SWITCHING PARAMETERS =====================
    if strategy_name == "Regime Switching (Trend + Mean-Reversion)":

        st.markdown("### Regime switching parameters")

        col_vol1, col_vol2, col_alpha = st.columns(3)
        with col_vol1:
            vol_short_window = st.slider(
                "Short volatility window",
                min_value=5,
                max_value=60,
                value=30,
            )
        with col_vol2:
            vol_long_window = st.slider(
                "Long volatility window",
                min_value=50,
                max_value=300,
                value=80,
            )
        with col_alpha:
            alpha = st.slider(
                "Regime change threshold (α)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
            )

        st.markdown("### Trend-following parameters")
        trend_ma_window = st.slider(
            "Moving average window (Trend)",
            min_value=10,
            max_value=200,
            value=30,
        )

        st.markdown("### Mean-reversion parameters")
        col_mr1, col_mr2 = st.columns(2)
        with col_mr1:
            mr_window = st.slider(
                "Mean-reversion window",
                min_value=10,
                max_value=60,
                value=30,
            )
        with col_mr2:
            z_threshold = st.slider(
                "Z-score threshold",
                min_value=0.5,
                max_value=3.0,
                value=0.8,
                step=0.1,
            )
        
    
        # ----- Automatic optimization button -----
        if "optimize_regime" not in st.session_state:
            st.session_state.optimize_regime = False

        if st.button("Automatically optimize parameters", key="optimize_regime_button"):
            st.session_state.optimize_regime = True


        # ----- Run optimization -----
        if st.session_state.optimize_regime:
            with st.spinner("Optimization in progress..."):
                result = optimize_regime_switching(prices, periods_per_year)

            st.session_state.optimize_regime = False  # prevent infinite reruns

            if result is None:
                st.error("Unable to find optimal parameters.")
            else:
                st.success(
                    f"Optimal parameters found: "
                    f"vol_short={result['vol_short_window']}, "
                    f"vol_long={result['vol_long_window']}, "
                    f"α={result['alpha']}, "
                    f"MA_trend={result['trend_ma_window']}, "
                    f"MR_window={result['mr_window']}, "
                    f"Z={result['z_threshold']} "
                    f"(total return ≈ {result['metrics']['total_return']*100:.2f} %)"
                )

                # Update sliders directly via session_state
                st.session_state["vol_short_window"] = result["vol_short_window"]
                st.session_state["vol_long_window"] = result["vol_long_window"]
                st.session_state["alpha"] = result["alpha"]
                st.session_state["trend_ma_window"] = result["trend_ma_window"]
                st.session_state["mr_window"] = result["mr_window"]
                st.session_state["z_threshold"] = result["z_threshold"]



    if prices.empty:
        st.warning("No price data available.")
        return

    # ---------- 2.5. BENCHMARK & SELECTED STRATEGY ----------
    benchmark_df = buy_and_hold(prices)

    if strategy_name == "Buy & Hold":
        strat_df = benchmark_df.copy()

    elif strategy_name == "Moving Average Crossover":
        strat_df = moving_average_crossover(
            prices,
            short_window=st.session_state.short_window,
            long_window=st.session_state.long_window,
        )

    elif strategy_name == "Regime Switching (Trend + Mean-Reversion)":
        strat_df = regime_switch_trend_meanrev(
            prices,
            vol_short_window=vol_short_window,
            vol_long_window=vol_long_window,
            alpha=alpha,
            trend_ma_window=trend_ma_window,
            mr_window=mr_window,
            z_threshold=z_threshold,
        )

    else:
        st.error("Unknown strategy.")
        st.stop()

        strat_df = moving_average_crossover(
            prices,
            short_window=st.session_state.short_window,
            long_window=st.session_state.long_window,
        )
    
    # ---------- 2.5.bis. TRADE EXTRACTION ----------
    trades_df = extract_trades_from_position(
        strat_df["price"],
        strat_df["position"],
)


    # ---------- 2.6. COMPLIANT MAIN CHART ----------
    st.subheader("Main chart (compliant): Price vs Strategy (base 1)")

    # Base-1 price (same range as strat_df)
    price_base1 = _normalize_base1(strat_df["price"])

    # Base-1 strategy equity
    equity_base1 = _normalize_base1(strat_df["equity_curve"])

    fig_main = _plot_price_vs_equity_base1(
        dates=strat_df.index,
        price_base1=price_base1,
        equity_base1=equity_base1,
        asset_name=selected_asset.name,
        strategy_name=strategy_name,
    )

    st.plotly_chart(fig_main, use_container_width=True)

        
    st.subheader("Comparison on a normalized basis (starting value = 1)")

    chart_df = pd.DataFrame(
        {
            "Buy & Hold CAC 40": benchmark_df["equity_curve"],
            "Selected strategy": strat_df["equity_curve"],
        }
    )

    st.line_chart(chart_df)

    if strategy_name == "Moving Average Crossover":
        with st.expander("Show the moving averages used"):
            ma_df = strat_df[["price", "ma_short", "ma_long"]].dropna()
            ma_df = ma_df.rename(columns={"price": "CAC 40 price"})
            st.line_chart(ma_df)
            
    if strategy_name == "Regime Switching (Trend + Mean-Reversion)":
        with st.expander("📊 Model details (debug)"):
            debug_df = strat_df[[
                "regime",
                "vol_short",
                "vol_long",
                "ma_trend",
                "zscore",
                "trend_signal",
                "mr_signal",
                "position",
            ]].dropna()
            st.dataframe(debug_df.tail(25))


    # ---------- 2.7. METRICS: STRATEGY VS BUY & HOLD ----------


    st.subheader("Performance metrics: strategy vs Buy & Hold")

    benchmark_metrics = compute_all_metrics(
        equity_curve=benchmark_df["equity_curve"],
        returns=benchmark_df["strategy_returns"],
        risk_free_rate=0.0,
        periods_per_year=periods_per_year,
    )

    strategy_metrics = compute_all_metrics(
        equity_curve=strat_df["equity_curve"],
        returns=strat_df["strategy_returns"],
        risk_free_rate=0.0,
        periods_per_year=periods_per_year,
    )

    col_bh, col_strat = st.columns(2)

    with col_bh:
        st.markdown("### Buy & Hold CAC 40")
        st.metric("Total return", f"{benchmark_metrics['total_return'] * 100:.2f} %")
        st.metric("Annualized return", f"{benchmark_metrics['annualized_return'] * 100:.2f} %")
        st.metric(
            "Annualized volatility",
            f"{benchmark_metrics['annualized_volatility'] * 100:.2f} %",
        )
        sharpe_bh = benchmark_metrics["sharpe_ratio"]
        sharpe_bh_str = "N/A" if pd.isna(sharpe_bh) else f"{sharpe_bh:.2f}"
        st.metric("Sharpe ratio", sharpe_bh_str)
        st.metric("Max drawdown", f"{benchmark_metrics['max_drawdown'] * 100:.2f} %")

    with col_strat:
        st.markdown("### Selected strategy")
        st.metric("Total return", f"{strategy_metrics['total_return'] * 100:.2f} %")
        st.metric("Annualized return", f"{strategy_metrics['annualized_return'] * 100:.2f} %")
        st.metric(
            "Annualized volatility",
            f"{strategy_metrics['annualized_volatility'] * 100:.2f} %",
        )
        sharpe_strat = strategy_metrics["sharpe_ratio"]
        sharpe_strat_str = "N/A" if pd.isna(sharpe_strat) else f"{sharpe_strat:.2f}"
        st.metric("Sharpe ratio", sharpe_strat_str)
        st.metric("Max drawdown", f"{strategy_metrics['max_drawdown'] * 100:.2f} %")

        # ---------- 2.7.bis. QUALITATIVE COMPARISON MESSAGES ----------
    st.markdown("### Qualitative comparison")

    messages = _build_comparison_messages(
        benchmark_metrics=benchmark_metrics,
        strategy_metrics=strategy_metrics,
    )

    if messages:
        for msg in messages:
            st.markdown(f"- {msg}")
    else:
        st.info("No comparison message available (insufficient data).")

    # ---------- 2.8. DRAWDOWN CHART ----------
    st.subheader("Drawdown analysis")

    def _compute_drawdown(equity: pd.Series) -> pd.Series:
        running_max = equity.cummax()
        dd = equity / running_max - 1.0
        return dd

    dd_bh = _compute_drawdown(benchmark_df["equity_curve"])
    dd_strat = _compute_drawdown(strat_df["equity_curve"])

    dd_df = pd.DataFrame(
        {
            "Buy & Hold drawdown": dd_bh,
            "Strategy drawdown": dd_strat,
        }
    )

    st.line_chart(dd_df)

    # ---------- 2.9. TRADING METRICS ----------
    st.subheader("Trading metrics (from reconstructed trades)")

    trade_metrics = compute_trade_metrics(trades_df)

    ctm1, ctm2, ctm3, ctm4 = st.columns(4)
    ctm1.metric("Number of trades", f"{trade_metrics['n_trades']}")
    win_rate = trade_metrics["win_rate"]
    ctm2.metric("Win rate", "N/A" if pd.isna(win_rate) else f"{win_rate*100:.1f} %")

    pct_longs = trade_metrics["pct_longs"]
    pct_shorts = trade_metrics["pct_shorts"]
    ctm3.metric("Long trades", "N/A" if pd.isna(pct_longs) else f"{pct_longs*100:.1f} %")
    ctm4.metric("Short trades", "N/A" if pd.isna(pct_shorts) else f"{pct_shorts*100:.1f} %")

    ctm5, ctm6, ctm7, ctm8 = st.columns(4)
    avg_tr = trade_metrics["avg_trade_return"]
    avg_win = trade_metrics["avg_win_return"]
    avg_loss = trade_metrics["avg_loss_return"]
    avg_hold = trade_metrics["avg_holding_period"]

    ctm5.metric("Average trade return", "N/A" if pd.isna(avg_tr) else f"{avg_tr*100:.2f} %")
    ctm6.metric("Average winning trade", "N/A" if pd.isna(avg_win) else f"{avg_win*100:.2f} %")
    ctm7.metric("Average losing trade", "N/A" if pd.isna(avg_loss) else f"{avg_loss*100:.2f} %")
    ctm8.metric("Average holding period (bars)", "N/A" if pd.isna(avg_hold) else f"{avg_hold:.1f}")

    # ---------- 2.10. TRADE HISTORY TABLE ----------
    st.subheader("Trade history (details)")

    if trades_df is None or trades_df.empty:
        st.info("No trades were detected for this strategy over the selected period.")
    else:
        # Nice formatting for display
        show_trades = trades_df.copy()

        # Convert timestamps to readable strings (if index-like)
        if "entry_date" in show_trades.columns:
            show_trades["entry_date"] = pd.to_datetime(show_trades["entry_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "exit_date" in show_trades.columns:
            show_trades["exit_date"] = pd.to_datetime(show_trades["exit_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Display columns in a clean order if present
        preferred_cols = [
            "entry_date",
            "exit_date",
            "direction",
            "entry_price",
            "exit_price",
            "holding_period_bars",
            "trade_return_pct",
        ]
        existing_cols = [c for c in preferred_cols if c in show_trades.columns]
        remaining_cols = [c for c in show_trades.columns if c not in existing_cols]
        show_trades = show_trades[existing_cols + remaining_cols]

        st.dataframe(show_trades, use_container_width=True)

        st.download_button(
            label="⬇️ Download trade history (CSV)",
            data=show_trades.to_csv(index=False).encode("utf-8"),
            file_name=f"trades_{selected_asset.ticker}_{strategy_name.replace(' ', '_')}.csv",
            mime="text/csv",
        )




    st.subheader("Daily reports")

    reports_dir = Path("reports") / "quant_a"
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(reports_dir.glob("daily_report_*.txt"), reverse=True)

    if not files:
        st.info("No reports found in reports/quant_a. (Cron may not have run yet.)")
    else:
        chosen = st.selectbox("Select a report", options=[f.name for f in files])
        path = reports_dir / chosen
        content = path.read_text(encoding="utf-8")
        st.text(content)

        st.download_button(
            "⬇️ Download report",
            data=content.encode("utf-8"),
            file_name=chosen,
            mime="text/plain",
        )
