# app/quant_a/ui_quant_a.py
import streamlit as st

def render_quant_a_page():
    st.title("Quant A - Analyse d'un indice (CAC40)")
    st.write(
        """
        Ce module se concentre sur **un seul actif** : le CAC40.

        Ici, on aura :
        - Le chargement des données (prix du CAC40)
        - Plusieurs stratégies de backtest (Buy & Hold, Moving Average, etc.)
        - Les métriques de performance (Sharpe, max drawdown, volatilité...)
        - Des contrôles interactifs pour ajuster les paramètres de stratégie
        """
    )

    st.warning(
        "🎯 Prochaine étape : connecter les données du CAC40 et ajouter les stratégies."
    )
