# app/quant_a/ui_quant_a.py

import datetime as dt

import pandas as pd
import streamlit as st

from .data_loader import load_cac40_history
from .strategies import buy_and_hold, moving_average_crossover
from .metrics import compute_all_metrics


def _get_periods_per_year(interval: str) -> int:
    """
    Renvoie le nombre de périodes par an en fonction de l'intervalle Yahoo.
    """
    if interval == "1d":
        return 252
    if interval == "1wk":
        return 52
    if interval == "1mo":
        return 12
    # fallback
    return 252


def render_quant_a_page():
    st.title("Quant A – Analyse univariée du CAC 40")

    st.markdown(
        """
        Ce module analyse **exclusivement le CAC 40** à partir de données Yahoo Finance.
        
        Utilisez les contrôles ci-dessous pour :
        - choisir la **périodicité** des données (journalier, hebdomadaire, mensuel),
        - configurer les **paramètres de stratégie** (Buy & Hold ou Crossover de moyennes mobiles),
        - visualiser la performance de la stratégie et ses **métriques**.
        """
    )

    # ===================== CONTRÔLES : DONNÉES (PÉRIODICITÉ) =====================
    with st.sidebar:
        st.header("Paramètres des données (CAC 40)")

        period_choice = st.selectbox(
            "Périodicité des données",
            options=[
                "Journalier (1d)",
                "Hebdomadaire (1wk)",
                "Mensuel (1mo)",
            ],
            index=0,
        )

        interval_map = {
            "Journalier (1d)": "1d",
            "Hebdomadaire (1wk)": "1wk",
            "Mensuel (1mo)": "1mo",
        }
        interval = interval_map[period_choice]

        today = dt.date.today()
        default_start = today - dt.timedelta(days=365 * 5)

        start_date, end_date = st.date_input(
            "Période d'étude (début / fin)",
            value=(default_start, today),
        )

        if start_date >= end_date:
            st.error("La date de début doit être strictement inférieure à la date de fin.")
            st.stop()

    # ===================== CHARGEMENT DES DONNÉES CAC 40 =====================
    with st.spinner("Chargement des données du CAC 40..."):
        df = load_cac40_history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
        )

    if df is None or df.empty:
        st.warning("Aucune donnée disponible pour le CAC 40 sur cette période.")
        return

    prices = df["Close"].astype(float).copy()
    prices = prices.sort_index()
    prices.name = "CAC 40 (Close)"

    st.subheader("Prix du CAC 40")
    st.line_chart(prices)

    # ===================== CONTRÔLES : STRATÉGIE =====================
    st.subheader("Paramètres de stratégie")

    strategy_name = st.selectbox(
        "Choix de la stratégie",
        ["Buy & Hold", "Moving Average Crossover"],
    )

    short_window = None
    long_window = None

    if strategy_name == "Moving Average Crossover":
        st.markdown("Configurez les **périodes** des moyennes mobiles :")
        col1, col2 = st.columns(2)

        with col1:
            short_window = st.slider(
                "Période moyenne courte",
                min_value=5,
                max_value=100,
                value=20,
                step=1,
            )
        with col2:
            long_window = st.slider(
                "Période moyenne longue",
                min_value=20,
                max_value=300,
                value=100,
                step=5,
            )

        if short_window >= long_window:
            st.warning("La période courte doit être strictement inférieure à la période longue.")
            st.stop()

    if prices.empty:
        st.warning("Aucune donnée de prix disponible.")
        return

    # ===================== APPLICATION DE LA STRATÉGIE =====================
    if strategy_name == "Buy & Hold":
        strat_df = buy_and_hold(prices)
    else:
        strat_df = moving_average_crossover(
            prices,
            short_window=short_window,
            long_window=long_window,
        )

    # ===================== GRAPHIQUE : PRIX VS STRATÉGIE =====================
    st.subheader("Comparaison prix / stratégie")

    chart_df = pd.DataFrame(
        {
            "Prix CAC 40": strat_df["price"],
            "Stratégie (valeur cumulée)": strat_df["equity_curve"],
        }
    )

    st.line_chart(chart_df)

    if strategy_name == "Moving Average Crossover":
        with st.expander("Afficher les moyennes mobiles utilisées"):
            ma_df = strat_df[["price", "ma_short", "ma_long"]].dropna()
            ma_df = ma_df.rename(columns={"price": "Prix CAC 40"})
            st.line_chart(ma_df)

    # ===================== MÉTRIQUES DE PERFORMANCE =====================
    st.subheader("Métriques de performance de la stratégie")

    periods_per_year = _get_periods_per_year(interval)

    metrics = compute_all_metrics(
        equity_curve=strat_df["equity_curve"],
        returns=strat_df["strategy_returns"],
        risk_free_rate=0.0,
        periods_per_year=periods_per_year,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rendement total", f"{metrics['total_return'] * 100:.2f} %")
    col2.metric("Rendement annualisé", f"{metrics['annualized_return'] * 100:.2f} %")
    col3.metric("Volatilité annualisée", f"{metrics['annualized_volatility'] * 100:.2f} %")

    col4, col5 = st.columns(2)
    sharpe_val = metrics["sharpe_ratio"]
    sharpe_str = "N/A" if pd.isna(sharpe_val) else f"{sharpe_val:.2f}"
    col4.metric("Sharpe ratio", sharpe_str)
    col5.metric("Max drawdown", f"{metrics['max_drawdown'] * 100:.2f} %")

    # ===================== APERÇU DES DONNÉES =====================
    with st.expander("Voir un extrait des données brutes"):
        st.dataframe(df.tail(10))
