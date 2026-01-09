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
    st.set_page_config(page_title="Quant B — Portfolio", layout="wide")
    st.title("Quant B — Module Portefeuille Multi-Actifs")

    load_history = _import_load_history()
    if load_history is None:
        st.error(
            "Impossible d'importer `load_history`.\n"
            "➡️ Ouvre `app/quant_b/page_quant_b.py` et adapte la fonction `_import_load_history()` "
            "selon l'emplacement réel de votre data loader."
        )
        st.stop()

    asset_universe = _import_universe()
    if asset_universe is None:
        st.error(
            "Impossible d'importer `ASSET_UNIVERSE`.\n"
            "➡️ Vérifie que `app/quant_a/universe.py` existe et contient `ASSET_UNIVERSE`."
        )
        st.stop()

    all_labels, label_to_ticker, class_to_labels = _flatten_universe(asset_universe)

    # -------- Sidebar controls --------
    st.sidebar.header("Paramètres")

    interval = st.sidebar.selectbox("Intervalle", ["1d", "1h", "15m", "5m"], index=0)

    rebalance_ui = st.sidebar.selectbox("Rebalancement", ["Aucun", "Quotidien (D)", "Hebdo (W)", "Mensuel (M)"], index=3)
    rebalance_map = {"Aucun": None, "Quotidien (D)": "D", "Hebdo (W)": "W", "Mensuel (M)": "M"}
    rebalance = rebalance_map[rebalance_ui]

    st.sidebar.subheader("Sélection des actifs (≥ 3)")
    selected_labels = st.sidebar.multiselect(
        "Choisis des actifs",
        options=all_labels,
        default=all_labels[:3] if len(all_labels) >= 3 else all_labels,
    )

    if len(selected_labels) < 3:
        st.warning("Sélectionne au moins 3 actifs pour le module Quant B.")
        st.stop()

    tickers = [label_to_ticker[lbl] for lbl in selected_labels]

    st.sidebar.subheader("Poids")
    mode_weights = st.sidebar.radio("Mode", ["Equal-weight", "Custom"], index=0)

    weights = {}
    if mode_weights == "Equal-weight":
        w = 1.0 / len(tickers)
        weights = {t: w for t in tickers}
        st.sidebar.caption(f"Chaque actif = {w:.3f}")
    else:
        # sliders de poids, puis normalisation
        raw = {}
        for t in tickers:
            raw[t] = st.sidebar.slider(f"Poids {t}", min_value=0.0, max_value=1.0, value=1.0 / len(tickers), step=0.01)
        s = sum(raw.values())
        if s == 0:
            st.sidebar.error("La somme des poids ne peut pas être 0.")
            st.stop()
        weights = {k: v / s for k, v in raw.items()}
        st.sidebar.caption(f"Somme normalisée = 1.00 (somme brute={s:.2f})")

    # -------- Load data --------
    st.subheader("Données & Résultats")

    with st.spinner("Chargement des données..."):
        # IMPORTANT : adapte les paramètres selon votre signature load_history
        # Ici on suppose une signature type: load_history(tickers=..., interval=...)
        try:
            prices = load_history(tickers=tickers, interval=interval)
        except TypeError:
            # fallback si signature différente
            prices = load_history(tickers, interval=interval)

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        st.error("Le loader a renvoyé un objet non valide ou vide. Vérifie `load_history`.")
        st.stop()

    # S'assure que colonnes = tickers sélectionnés
    # Si votre loader renvoie OHLC multi-colonnes, adapte ici (ex: sélectionner Close)
    # Exemple d'adaptation possible (à décommenter si nécessaire) :
    # if "Close" in prices.columns:
    #     prices = prices[["Close"]]

    # -------- Portfolio calc --------
    # On garde uniquement les colonnes demandées (si le loader renvoie plus)
    cols = [c for c in prices.columns if c in tickers]
    if len(cols) < 3:
        st.error(
            "Je ne retrouve pas les tickers sélectionnés dans les colonnes de `prices`.\n"
            "➡️ Affiche `prices.columns` et adapte la sélection (souvent il faut extraire Close)."
        )
        st.write(prices.head())
        st.write(prices.columns)
        st.stop()

    prices = prices[cols].dropna(how="all")
    prices = prices.dropna()  # simple: on garde les dates complètes

    if len(prices) < 10:
        st.warning("Peu de données chargées (moins de 10 points). Les stats peuvent être peu fiables.")

    portfolio_value = compute_portfolio_value(prices, weights, rebalance=rebalance, base=100.0)

    asset_returns = compute_returns(prices)
    portfolio_returns = np.log(portfolio_value / portfolio_value.shift(1)).dropna()

    m = portfolio_metrics(
        asset_returns=asset_returns,
        portfolio_returns=portfolio_returns,
        portfolio_value=portfolio_value,
        interval=interval,
    )

    # -------- Layout --------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement cumulé (portefeuille)", f"{m['portfolio_cumulative_return']*100:.2f}%")
    c2.metric("Vol annualisée (portefeuille)", f"{m['portfolio_vol_annualized']*100:.2f}%")
    c3.metric("CAGR approx (portefeuille)", f"{m['portfolio_cagr']*100:.2f}%")
    c4.metric("Ratio diversification", f"{m['diversification_ratio']:.3f}")

    left, right = st.columns([2, 1])

    with left:
        norm_prices = _normalize_prices(prices, base=100.0)
        fig_main = _plot_prices_and_portfolio(norm_prices, portfolio_value)
        st.plotly_chart(fig_main, use_container_width=True)

    with right:
        st.write("**Poids utilisés (normalisés)**")
        st.dataframe(pd.DataFrame({"ticker": list(weights.keys()), "weight": list(weights.values())}).set_index("ticker"))

        st.write("**Vol annualisée par actif**")
        st.dataframe(m["asset_vols_annualized"].to_frame("vol_annualized"))

    st.divider()
    st.subheader("Corrélation & Diversification")

    fig_corr = _plot_corr_heatmap(m["correlation"])
    st.plotly_chart(fig_corr, use_container_width=True)

    st.caption(
        "Interprétation du ratio diversification : "
        "ratio < 1 → portefeuille moins volatil que la moyenne des actifs (diversification utile)."
    )
