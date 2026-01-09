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

def _extract_close_matrix(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Transforme un DataFrame OHLCV multi-actifs (souvent MultiIndex) en matrice
    de prix de clôture: colonnes = tickers, valeurs = Close (ou Adj Close).
    """

    if isinstance(prices.columns, pd.MultiIndex):
        # Cas typique yfinance : niveaux = (field, ticker) OU (ticker, field)
        levels = [list(map(str, prices.columns.get_level_values(i).unique())) for i in range(prices.columns.nlevels)]

        # On essaie de détecter quel niveau contient OHLCV
        ohlcv_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        # Si level 0 = fields
        if any(f in ohlcv_fields for f in levels[0]):
            field_level, ticker_level = 0, 1
        # Si level 1 = fields
        elif prices.columns.nlevels > 1 and any(f in ohlcv_fields for f in levels[1]):
            ticker_level, field_level = 0, 1
        else:
            # fallback : on suppose (field, ticker)
            field_level, ticker_level = 0, 1

        # Choix du champ prix
        field_choice = "Adj Close" if "Adj Close" in levels[field_level] else "Close"

        # On garde uniquement les tickers choisis
        if field_level == 0:
            out = prices.loc[:, (field_choice, tickers)]
            out.columns = out.columns.get_level_values(ticker_level)
        else:
            out = prices.loc[:, (tickers, field_choice)]
            out.columns = out.columns.get_level_values(ticker_level)

        return out

    # Cas colonnes "plates" mais dupliquées: on ne peut pas deviner le mapping sans info.
    # On renvoie tel quel (mais ça plantera si on affiche). Mieux: lever une erreur explicite.
    if prices.columns.duplicated().any():
        raise ValueError(
            "prices a des colonnes dupliquées (format OHLCV aplati). "
            "Ton loader doit renvoyer un MultiIndex OU tu dois renvoyer directement une matrice de Close."
        )

    return prices[tickers]

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

    # =========================
    # Load data
    # =========================

    now = dt.datetime.now()

    # cache keys
    if "qb_cached_prices" not in st.session_state:
        st.session_state.qb_cached_prices = None
    if "qb_last_load_time" not in st.session_state:
        st.session_state.qb_last_load_time = None
    if "qb_last_interval" not in st.session_state:
        st.session_state.qb_last_interval = None
    if "qb_last_tickers" not in st.session_state:
        st.session_state.qb_last_tickers = None

    need_reload = False
    tickers_key = "|".join(tickers)

    if st.session_state.qb_cached_prices is None or st.session_state.qb_cached_prices.empty:
        need_reload = True
    elif (
        st.session_state.qb_last_interval != interval
        or st.session_state.qb_last_tickers != tickers_key
    ):
        need_reload = True
    else:
        last_time = st.session_state.qb_last_load_time
        if last_time is None or (now - last_time).total_seconds() > 300:
            need_reload = True

    if need_reload:
        with st.spinner("Loading data..."):

            end = dt.date.today().strftime("%Y-%m-%d")

            if interval == "1d":
                # ✅ FORCE enough history for MA / Regime
                start = (dt.date.today() - dt.timedelta(days=365 * 5)).strftime("%Y-%m-%d")
                intraday_days = 5
            else:
                start = None
                intraday_days = 180

            prices = load_prices_matrix(
                load_history,
                tickers=tickers,
                interval=interval,
                intraday_days=intraday_days,
                start=start,
                end=end,
            )

    else:
        prices = st.session_state.qb_cached_prices

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        st.error("The loader returned an empty/invalid DataFrame.")
        st.stop()

    prices = prices.dropna(how="all").dropna()
    if len(prices.columns) < 3:
        st.error("Not enough valid assets after cleaning (≥3 required).")
        st.stop()

    # =========================
    # Portfolio calc
    # =========================
    if mode == "Fixed weights":
        portfolio_value = compute_portfolio_value(prices, weights, rebalance=rebalance, base=100.0)

        weights_used_df = pd.DataFrame(index=prices.index, columns=prices.columns, data=np.nan)
        weights_used_df.iloc[0] = [weights.get(c, 0.0) for c in prices.columns]
        weights_used_df = weights_used_df.ffill().fillna(0.0)

    else:
        # ✅ STRATEGIES Quant A → Quant B (MA Crossover, Regime Switch, Buy&Hold)
        res = compute_strategy_weights(
            prices=prices,
            strategy=strategy_name,
            params=strategy_params,
            base_weights=weights if strategy_name == "Buy & Hold" else None,
        )

        weights_used_df = res.weights
        portfolio_value = compute_portfolio_value_from_weights(
            prices, weights_used_df, rebalance=rebalance, base=100.0
        )

    turnover = compute_turnover(weights_used_df)

    # =========================
    # Metrics
    # =========================
    asset_returns = compute_returns(prices)
    portfolio_returns = np.log(portfolio_value / portfolio_value.shift(1)).dropna()

    m = portfolio_metrics(
        asset_returns=asset_returns,
        portfolio_returns=portfolio_returns,
        portfolio_value=portfolio_value,
        interval=interval,
    )
    # =========================
    # Diversification KPIs (needed by UI)
    # =========================
    corr = m["correlation"]
    corr_vals = corr.values.copy()

    # average pairwise correlations (off-diagonal)
    mask = ~np.eye(corr_vals.shape[0], dtype=bool)
    avg_pairwise_corr = float(np.nanmean(corr_vals[mask])) if corr_vals.size > 1 else 1.0

    # last used weights
    last_weights = weights_used_df.iloc[-1].copy()

    # annualized vol per asset (already in m)
    asset_vols = m["asset_vols_annualized"].reindex(prices.columns).fillna(0.0)

    # annualized portfolio volatility
    port_vol = float(m["portfolio_vol_annualized"])

    # weighted average vol (proxy for "no diversification")
    w_aligned = last_weights.reindex(prices.columns).fillna(0.0).astype(float)
    vol_wavg = float((w_aligned.abs() * asset_vols).sum())

    # "classic" diversification ratio and volatility reduction
    div_ratio_true = (vol_wavg / port_vol) if port_vol > 0 else float("nan")
    vol_reduction = (1.0 - port_vol / vol_wavg) if vol_wavg > 0 else float("nan")

    # effective number of assets (weight concentration)
    w2 = float((w_aligned**2).sum())
    n_eff = (1.0 / w2) if w2 > 0 else float("nan")

    
    # =========================
    # Extra matrices (Cov, Distance) + Risk contributions
    # =========================
    ppy = m["periods_per_year"]

    cov_ann = covariance_matrix_annualized(asset_returns, periods_per_year=ppy)
    dist = correlation_distance_matrix(m["correlation"])

    last_weights = weights_used_df.iloc[-1].copy()
    rc_pct, rc_abs, port_vol_from_cov = risk_contributions_from_cov(last_weights, cov_ann)

    # =========================
    # Display (UI only)
    # =========================

    # --- Config recap in an expander (readability) ---
    with st.expander("Portfolio configuration", expanded=False):
        cR1, cR2, cR3 = st.columns(3)
        with cR1:
            st.write("**Mode**")
            st.write(mode)
            st.write("**Interval**")
            st.write(interval)
        with cR2:
            st.write("**Rebalancing**")
            st.write(rebalance_ui)
            if mode == "Strategy":
                st.write("**Strategy**")
                st.write(strategy_name)
        with cR3:
            st.write("**Assets**")
            st.write(", ".join(list(prices.columns)))

        if mode == "Fixed weights":
            st.write("**Fixed weights (normalized)**")
            st.dataframe(pd.Series(weights).reindex(prices.columns).fillna(0.0).to_frame("weight"))
        else:
            st.write("**Allocation rule**")
            st.write(strategy_params.get("allocation_rule", "Equal-weight"))
            if strategy_params.get("allocation_rule") == "Inverse-vol":
                st.write("**Vol window**")
                st.write(int(strategy_params.get("alloc_vol_window", 20)))

    # --- Main tabs ---
    tab_overview, tab_div, tab_details = st.tabs(["Overview", "Diversification", "Details"])

    # =========================
    # OVERVIEW
    # =========================
    with tab_overview:
        st.markdown('<div class="section-title">Visual comparison: assets vs portfolio</div>', unsafe_allow_html=True)

        # Main chart controls (asset selection)
        shown_assets = st.multiselect(
            "Asset series shown (portfolio is always displayed)",
            options=list(prices.columns),
            default=list(prices.columns),
            key="qb_chart_assets",
        )
        if len(shown_assets) == 0:
            shown_assets = list(prices.columns)

        norm_prices = _normalize_prices(prices[shown_assets], base=100.0)
        fig_main = _plot_prices_and_portfolio(norm_prices, portfolio_value)
        fig_main.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_main, use_container_width=True)

        st.markdown('<div class="section-title alt">Key Portfolio Metrics</div>', unsafe_allow_html=True)

        # Row 1 KPIs (performance/risk)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cumulative return", f"{m['portfolio_cumulative_return']*100:.2f}%")
        k2.metric("Annualized return", f"{m['portfolio_return_annualized']*100:.2f}%")
        k3.metric("Annualized vol", f"{m['portfolio_vol_annualized']*100:.2f}%")
        k4.metric("Max Drawdown", f"{m['portfolio_max_drawdown']*100:.2f}%")

        # Row 2 KPIs (ratios + diversification)
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Sharpe (rf=0)", f"{m['portfolio_sharpe']:.2f}")
        k6.metric("Diversification ratio (quick)", f"{m['diversification_ratio']:.3f}")
        k7.metric("Average correlation", f"{avg_pairwise_corr:.2f}")
        k8.metric("Effective N", f"{n_eff:.2f}")

        # Row 3 KPIs (explicit diversification)
        k9, k10, k11, k12 = st.columns(4)
        k9.metric("Diversification Ratio (vol_wavg/vol_p)", f"{div_ratio_true:.2f}")
        k10.metric("Vol reduction", f"{vol_reduction*100:.1f}%")
        k11.metric("Turnover (latest)", "—")
        k12.metric("Vol (from cov)", "—")


        st.caption(
            "Quick read: diversification ratio > 1 and vol reduction > 0% indicate a gain driven by correlations < 1."
        )

    # =========================
    # DIVERSIFICATION
    # =========================
    with tab_div:
        st.markdown('<div class="section-title neutral">Diversification effects: matrices & risk contributions</div>', unsafe_allow_html=True)

        subtab_mats, subtab_rc = st.tabs(["Matrices", "Risk contributions"])

        with subtab_mats:
            # Matrices: avoid 3 heatmaps at once => 2 columns + 1 sub-section
            cA, cB = st.columns(2)
            with cA:
                st.plotly_chart(_plot_corr_heatmap(m["correlation"]), use_container_width=True)
            with cB:
                st.plotly_chart(_plot_cov_heatmap(cov_ann), use_container_width=True)

            st.plotly_chart(_plot_dist_heatmap(dist), use_container_width=True)

        with subtab_rc:
            st.write("**Risk contributions (volatility / variance)**")
            df_rc = pd.DataFrame({
                "weight": last_weights.reindex(cov_ann.index).fillna(0.0),
                "risk_contrib_%": (rc_pct * 100.0).round(2),
            }).sort_values("risk_contrib_%", ascending=False)

            st.dataframe(df_rc, use_container_width=True)
            st.caption("An asset can have a small weight but a large risk contribution: this is a key diversification insight.")

    # =========================
    # DETAILS
    # =========================
    with tab_details:
        st.markdown('<div class="section-title neutral">Details (tables)</div>', unsafe_allow_html=True)

        dL, dR = st.columns([1, 1])

        with dL:
            st.write("**Latest weights used**")
            last_w_sorted = weights_used_df.iloc[-1].sort_values(ascending=False)
            st.dataframe(last_w_sorted.to_frame("weight"), use_container_width=True)

        with dR:
            st.write("**Annualized vol per asset**")
            st.dataframe(m["asset_vols_annualized"].to_frame("vol_annualized"), use_container_width=True)

        with st.expander("Weights matrix (over time)", expanded=False):
            st.dataframe(weights_used_df.tail(200), use_container_width=True)
