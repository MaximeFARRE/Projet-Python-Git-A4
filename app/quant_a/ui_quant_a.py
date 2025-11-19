import streamlit as st
import pandas as pd

from .data_loader import load_cac40_history, get_last_cac40_close

from .strategies import buy_and_hold, moving_average_crossover
from .metrics import compute_all_metrics


def render_quant_a_page():
    st.title("Module Quant A - Analyse d’un actif")

    # TODO: adapter au nom réel de ta fonction de chargement
    # prices = load_price_data(...)
    # On suppose que 'prices' est une pd.Series indexée par date
    # et que tu as déjà géré la sélection d'actif / période au-dessus.
        # Chargement des données CAC40
    df = load_cac40_history()

    if df is None or df.empty:
        st.warning("Aucune donnée disponible pour le CAC40.")
        return

    # On récupère la série de prix
    prices = df["Close"].copy()
    prices = prices.sort_index()

    strategy_name = st.selectbox(
        "Choix de la stratégie",
        ["Buy & Hold", "Moving Average Crossover"],
    )

    short_window = None
    long_window = None

    if strategy_name == "Moving Average Crossover":
        st.subheader("Paramètres de la stratégie")
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

    st.title("Quant A - Analyse d'un indice (CAC40)")

    st.sidebar.subheader("Paramètres de données")

    # Paramètres de date
    default_start = pd.to_datetime("2020-01-01")
    start_date = st.sidebar.date_input("Date de début", value=default_start)
    end_date = st.sidebar.date_input("Date de fin", value=pd.Timestamp.today())

    if start_date > end_date:
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    with st.spinner("Chargement des données du CAC40..."):
        try:
            data = load_cac40_history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
            )
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            return

    if data.empty:
        st.error("Aucune donnée disponible pour cette période.")
        return

    # Dernier cours
    last_close = float(data["Close"].iloc[-1])
    st.metric("Dernier cours du CAC40", f"{last_close:,.2f} pts")

    # Graphique de prix
    st.subheader("Historique du CAC40 (clôture)")
    st.line_chart(data["Close"])

    # Aperçu des données
    with st.expander("Aperçu des données brutes"):
        st.dataframe(data.tail())

    st.info("Étape suivante : ajouter les stratégies de backtest et les métriques de performance.")
    
    if prices is None or prices.empty:
        st.warning("Aucune donnée de prix disponible.")
        return

    if strategy_name == "Buy & Hold":
        strat_df = buy_and_hold(prices)
    else:
        # Moving Average Crossover
        if short_window is None or long_window is None or short_window >= long_window:
            st.stop()
        strat_df = moving_average_crossover(
            prices,
            short_window=short_window,
            long_window=long_window,
        )
 
    st.subheader("Évolution du prix et de la stratégie")

    chart_df = pd.DataFrame(
        {
            "Prix": strat_df["price"],
            "Stratégie (valeur cumulée)": strat_df["equity_curve"],
        }
    )

    st.line_chart(chart_df)

    st.subheader("Métriques de performance")

    metrics = compute_all_metrics(
        equity_curve=strat_df["equity_curve"],
        returns=strat_df["strategy_returns"],
        risk_free_rate=0.0,      # tu peux exposer ça plus tard dans un slider / input
        periods_per_year=252,    # daily
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rendement total",
        f"{metrics['total_return'] * 100:.2f} %",
    )
    col2.metric(
        "Rendement annualisé",
        f"{metrics['annualized_return'] * 100:.2f} %",
    )
    col3.metric(
        "Volatilité annualisée",
        f"{metrics['annualized_volatility'] * 100:.2f} %",
    )

    col4, col5 = st.columns(2)
    col4.metric(
        "Sharpe ratio",
        f"{metrics['sharpe_ratio']:.2f}" if metrics["sharpe_ratio"] == metrics["sharpe_ratio"] else "N/A",
    )
    col5.metric(
        "Max drawdown",
        f"{metrics['max_drawdown'] * 100:.2f} %",
    )

