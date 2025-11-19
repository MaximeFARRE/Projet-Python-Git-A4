import streamlit as st
import pandas as pd

from .data_loader import load_cac40_history, get_last_cac40_close


def render_quant_a_page():
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
