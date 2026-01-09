import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import datetime as dt 

from app.quant_b.portfolio import compute_portfolio_value, compute_returns
from app.quant_b.metrics import (
    portfolio_metrics,
    covariance_matrix_annualized,
    correlation_distance_matrix,
    risk_contributions_from_cov,
)
from app.quant_b.data_adapter import load_prices_matrix
from app.quant_b.strategies import compute_strategy_weights
from app.quant_b.backtest import compute_portfolio_value_from_weights, compute_turnover


# ========= ADAPTER: import the data loader =========
def _import_load_history():
    try:
        from app.quant_a.data_loader import load_history
        return load_history
    except Exception:
        pass

    try:
        from app.quant_a.data_loader import load_history
        return load_history
    except Exception:
        pass

    return None


# ========= ADAPTER: import the universe =========
def _import_universe():
    try:
        from app.quant_a.universe import ASSET_UNIVERSE
        return ASSET_UNIVERSE
    except Exception:
        return None


def _flatten_universe(asset_universe):
    label_to_ticker = {}
    all_labels = []

    for asset_class, assets in asset_universe.items():
        for a in assets:
            label = f"{asset_class} - {a.name} ({a.ticker})"
            label_to_ticker[label] = a.ticker
            all_labels.append(label)

    return all_labels, label_to_ticker


def _normalize_weights_dict(raw: dict) -> dict:
    s = float(sum(raw.values()))
    if s <= 0:
        raise ValueError("Sum of weights = 0.")
    return {k: float(v) / s for k, v in raw.items()}


def _normalize_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    return base * (prices / prices.iloc[0])


def _plot_prices_and_portfolio(norm_prices: pd.DataFrame, portfolio_value: pd.Series) -> go.Figure:
    fig = go.Figure()

    for col in norm_prices.columns:
        fig.add_trace(go.Scatter(
            x=norm_prices.index,
            y=norm_prices[col],
            mode="lines",
            name=f"{col} (base100)",
        ))

    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        mode="lines",
        name="PORTFOLIO (base100)",
        line=dict(width=3),
    ))

    fig.update_layout(
        title="Assets (base 100) vs Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Level (base 100)",
        legend_title="Series",
        height=520,
    )
    return fig


def _plot_corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(title="Correlation matrix (returns)", height=420)
    return fig

def _plot_cov_heatmap(cov: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cov.values,
            x=cov.columns,
            y=cov.index,
            colorbar=dict(title="Cov (ann.)"),
        )
    )
    fig.update_layout(title="Annualized covariance matrix (returns)", height=420)
    return fig


def _plot_dist_heatmap(dist: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=dist.values,
            x=dist.columns,
            y=dist.index,
            colorbar=dict(title="Distance"),
        )
    )
    fig.update_layout(title="Distance matrix (correlation-based)", height=420)
    return fig


def render():
    # ⚠️ IMPORTANT: do NOT call set_page_config here, since main.py already does it.
    st.title("Quant B — Multi-Asset Portfolio Module")

    st.markdown(
        """
        <style>
        /* Layout */
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: -0.2px; }

        /* Subtle section header */
        .section-title {
            font-size: 1.05rem;
            font-weight: 650;
            margin: 0.2rem 0 0.6rem 0;
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            background: rgba(59, 130, 246, 0.08); /* light blue */
            border: 1px solid rgba(59, 130, 246, 0.18);
        }
        .section-title.alt {
            background: rgba(16, 185, 129, 0.08); /* light green */
            border: 1px solid rgba(16, 185, 129, 0.18);
        }
        .section-title.neutral {
            background: rgba(148, 163, 184, 0.12); /* slate */
            border: 1px solid rgba(148, 163, 184, 0.22);
        }

        /* Metric styling */
        [data-testid="stMetric"]{
            background: rgba(15, 23, 42, 0.03);
            border: 1px solid rgba(15, 23, 42, 0.06);
            padding: 10px 12px;
            border-radius: 12px;
        }
        [data-testid="stMetricLabel"] { opacity: 0.85; }
        [data-testid="stMetricValue"] { font-weight: 750; }

        /* Dataframes */
        .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid rgba(15, 23, 42, 0.08); }
        </style>
        """,
        unsafe_allow_html=True
    )

    load_history = _import_load_history()
    if load_history is None:
        st.error("Unable to import `load_history` (Quant A).")
        st.stop()

    asset_universe = _import_universe()
    if asset_universe is None:
        st.error("Unable to import `ASSET_UNIVERSE` (Quant A).")
        st.stop()

    all_labels, label_to_ticker = _flatten_universe(asset_universe)

    st.subheader("Parameters (single-page)")

    # =========================
    # SINGLE FORM: all parameters here
    # =========================
    c1, c2, c3 = st.columns(3)

    with c1:
        interval = st.selectbox("Interval", ["1d", "60m", "15m", "5m"], index=0, key="qb_interval")

    with c2:
        rebalance_ui = st.selectbox(
            "Rebalancing",
            ["None", "Daily (D)", "Weekly (W)", "Monthly (M)"],
            index=3,
            key="qb_rebalance_ui",
        )
        rebalance_map = {"None": None, "Daily (D)": "D", "Weekly (W)": "W", "Monthly (M)": "M"}
        rebalance = rebalance_map[rebalance_ui]

    with c3:
        mode = st.radio("Portfolio type", ["Fixed weights", "Strategy"], index=0, horizontal=True, key="qb_mode")

    st.markdown("#### Asset selection (≥ 3)")
    selected_labels = st.multiselect(
        "Choose assets",
        options=all_labels,
        default=all_labels[:3] if len(all_labels) >= 3 else all_labels,
        key="qb_assets",
    )

    if len(selected_labels) < 3:
        st.warning("Select at least 3 assets.")
        st.stop()

    tickers = [label_to_ticker[lbl] for lbl in selected_labels]

    # =========================
    # Validation + tickers
    # =========================
    if len(selected_labels) < 3:
        st.warning("Select at least 3 assets.")
        st.stop()

    tickers = [label_to_ticker[lbl] for lbl in selected_labels]

    # =========================
    # Build weights if needed
    # =========================
    weights = {}

    if mode == "Fixed weights":
        st.markdown("#### Weights (Fixed weights)")
        wmode = st.radio("Mode", ["Equal-weight", "Custom"], index=0, horizontal=True, key="qb_wmode_fixed")

        if wmode == "Equal-weight":
            w = 1.0 / len(tickers)
            weights = {t: w for t in tickers}
            st.caption(f"Each asset = {w:.3f}")
        else:
            st.caption("Adjust the weights (they will be normalized automatically).")
            raw = {}
            cols = st.columns(min(4, len(tickers)))
            for i, t in enumerate(tickers):
                with cols[i % len(cols)]:
                    raw[t] = st.slider(f"Weight {t}", 0.0, 1.0, 1.0 / len(tickers), 0.01, key=f"qb_w_{t}")
            weights = _normalize_weights_dict(raw)

    elif mode == "Strategy":
        st.markdown("#### Strategy (Quant A → Quant B)")
        strategy_name = st.selectbox("Strategy", ["Buy & Hold", "MA Crossover", "Regime Switch"], index=1, key="qb_strategy")

        strategy_params = {}
        # ===== Allocation rule (to match requirements) =====
        alloc_rule = st.selectbox(
            "Allocation rule",
            ["Equal-weight", "Inverse-vol"],
            index=0,
            key="qb_alloc_rule",
        )

        strategy_params["allocation_rule"] = alloc_rule

        if alloc_rule == "Inverse-vol":
            strategy_params["alloc_vol_window"] = st.slider(
                "Vol window (Inverse-vol)",
                5, 120, 20, 1,
                key="qb_alloc_vol_window",
            )

        if strategy_name == "Buy & Hold":
            st.markdown("##### Weights (Buy & Hold)")
            bh_mode = st.radio("Mode", ["Equal-weight", "Custom"], index=0, horizontal=True, key="qb_bh_mode")
            if bh_mode == "Equal-weight":
                w = 1.0 / len(tickers)
                weights = {t: w for t in tickers}
            else:
                raw = {}
                cols = st.columns(min(4, len(tickers)))
                for i, t in enumerate(tickers):
                    with cols[i % len(cols)]:
                        raw[t] = st.slider(f"Weight {t}", 0.0, 1.0, 1.0 / len(tickers), 0.01, key=f"qb_bh_{t}")
                weights = _normalize_weights_dict(raw)

        elif strategy_name == "MA Crossover":
            c1, c2 = st.columns(2)
            with c1:
                strategy_params["short_window"] = st.slider("MA short window", 5, 300, 50, 1, key="qb_ma_s")
            with c2:
                strategy_params["long_window"] = st.slider("MA long window", 10, 600, 200, 1, key="qb_ma_l")

        elif strategy_name == "Regime Switch":
            c1, c2, c3 = st.columns(3)
            with c1:
                strategy_params["vol_short_window"] = st.slider("Vol short window", 5, 200, 20, 1, key="qb_rs_vs")
                strategy_params["trend_ma_window"] = st.slider("Trend MA window", 10, 400, 50, 1, key="qb_rs_tma")
            with c2:
                strategy_params["vol_long_window"] = st.slider("Vol long window", 10, 400, 100, 1, key="qb_rs_vl")
                strategy_params["mr_window"] = st.slider("Mean reversion window", 5, 200, 20, 1, key="qb_rs_mr")
            with c3:
                strategy_params["alpha"] = st.slider("Alpha", 0.1, 5.0, 1.0, 0.1, key="qb_rs_a")
                strategy_params["z_threshold"] = st.slider("Z threshold", 0.1, 5.0, 1.0, 0.1, key="qb_rs_z")
                strategy_params["long_only"] = st.checkbox("Long-only", value=True, key="qb_rs_lo")
