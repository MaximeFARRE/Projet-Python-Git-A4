# app/quant_a/ui_quant_a.py

# ===================== 1. IMPORTS & FONCTIONS UTILITAIRES =====================

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


def _optimize_moving_average_params(
    prices: pd.Series,
    periods_per_year: int,
):
    """
    Recherche des paramètres (short_window, long_window) pour la stratégie
    Moving Average Crossover maximisant le rendement total sur la période.

    On fait un petit grid search raisonnable pour ne pas exploser le temps de calcul.
    """
    prices = prices.dropna()
    if prices.empty:
        return None

    best_score = -float("inf")
    best_short = None
    best_long = None
    best_metrics = None

    # Grille simple : short entre 5 et 60, long entre 50 et 250
    # (on reste raisonnable pour ne pas faire un truc trop lent)
    for short_window in range(5, 61, 5):
        for long_window in range(50, 251, 5):
            if short_window >= long_window:
                continue

            try:
                strat_df = moving_average_crossover(
                    prices,
                    short_window=short_window,
                    long_window=long_window,
                )
            except Exception:
                continue

            metrics = compute_all_metrics(
                equity_curve=strat_df["equity_curve"],
                returns=strat_df["strategy_returns"],
                risk_free_rate=0.0,
                periods_per_year=periods_per_year,
            )

            score = metrics.get("total_return", float("nan"))
            if pd.isna(score):
                continue

            if score > best_score:
                best_score = score
                best_short = short_window
                best_long = long_window
                best_metrics = metrics

    if best_short is None or best_long is None:
        return None

    return {
        "short_window": best_short,
        "long_window": best_long,
        "metrics": best_metrics,
    }


def _build_comparison_messages(
    benchmark_metrics: dict,
    strategy_metrics: dict,
) -> list[str]:
    """
    Génère des messages textuels de comparaison entre la stratégie
    et le Buy & Hold.
    """
    messages: list[str] = []

    tr_bh = benchmark_metrics["total_return"]
    tr_strat = strategy_metrics["total_return"]

    mdd_bh = benchmark_metrics["max_drawdown"]  # négatif
    mdd_strat = strategy_metrics["max_drawdown"]

    sharpe_bh_val = benchmark_metrics["sharpe_ratio"]
    sharpe_strat_val = strategy_metrics["sharpe_ratio"]

    # Surperformance (rendement)
    if pd.notna(tr_bh) and pd.notna(tr_strat):
        if tr_strat > tr_bh:
            messages.append("✅ La stratégie **surperforme** le Buy & Hold en termes de rendement total.")
        elif tr_strat < tr_bh:
            messages.append("⚠️ La stratégie **sous-performe** le Buy & Hold en termes de rendement total.")
        else:
            messages.append("ℹ️ La stratégie a un rendement total **équivalent** au Buy & Hold.")

    # Max drawdown (on compare la magnitude de la perte max)
    if pd.notna(mdd_bh) and pd.notna(mdd_strat):
        if abs(mdd_strat) < abs(mdd_bh):
            messages.append(
                "✅ La stratégie a un **max drawdown plus faible** que le Buy & Hold "
                "(meilleure protection contre les pertes)."
            )
        elif abs(mdd_strat) > abs(mdd_bh):
            messages.append(
                "⚠️ La stratégie a un **max drawdown plus élevé** que le Buy & Hold "
                "(risque de perte max plus important)."
            )
        else:
            messages.append("ℹ️ La stratégie a un max drawdown **similaire** au Buy & Hold.")

    # Sharpe ratio (performance ajustée du risque)
    if pd.notna(sharpe_bh_val) and pd.notna(sharpe_strat_val):
        if sharpe_strat_val > sharpe_bh_val:
            messages.append(
                "✅ La stratégie présente un **meilleur Sharpe ratio** que le Buy & Hold "
                "(meilleure performance ajustée du risque)."
            )
        elif sharpe_strat_val < sharpe_bh_val:
            messages.append(
                "⚠️ La stratégie a un **Sharpe ratio plus faible** que le Buy & Hold "
                "(moins bonne performance ajustée du risque)."
            )
        else:
            messages.append("ℹ️ La stratégie a un Sharpe ratio **proche** de celui du Buy & Hold.")
    else:
        messages.append(
            "ℹ️ Le Sharpe ratio n'est pas disponible pour l'une des deux séries "
            "(données insuffisantes ou volatilité nulle)."
        )

    return messages


# ===================== 2. FONCTION PRINCIPALE UI =====================


def render_quant_a_page():
    # ---------- 2.1. TITRE & INTRO ----------
    st.title("Quant A – Analyse univariée du CAC 40")

    st.markdown(
        """
        Ce module analyse **exclusivement le CAC 40** à partir de données Yahoo Finance.
        
        Objectifs :
        - Configurer la **périodicité** des données (journalier, hebdomadaire, mensuel),
        - Choisir une **stratégie** (Buy & Hold ou Crossover de moyennes mobiles),
        - (NOUVEAU) **Optimiser automatiquement** les paramètres de moyennes mobiles sur la période,
        - Comparer la **stratégie** au **Buy & Hold sur l’indice** :
            - rendement,
            - volatilité,
            - Sharpe,
            - max drawdown.
        """
    )

    # ---------- 2.2. CONTRÔLES DONNÉES (SIDEBAR) ----------
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
        periods_per_year = _get_periods_per_year(interval)

        today = dt.date.today()
        default_start = today - dt.timedelta(days=365 * 5)

        start_date, end_date = st.date_input(
            "Période d'étude (début / fin)",
            value=(default_start, today),
        )

        if start_date >= end_date:
            st.error("La date de début doit être strictement inférieure à la date de fin.")
            st.stop()

    # ---------- 2.3. CHARGEMENT DES DONNÉES ----------
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

    # ---------- 2.4. CONTRÔLES STRATÉGIE + BOUTON D'OPTIMISATION ----------
    st.subheader("Paramètres de stratégie")

    strategy_name = st.selectbox(
        "Choix de la stratégie",
        ["Buy & Hold", "Moving Average Crossover"],
    )

    short_window = None
    long_window = None

    if strategy_name == "Moving Average Crossover":
        st.markdown("Configurez les **périodes** des moyennes mobiles :")

        # Utilisation de session_state pour pouvoir mettre à jour les sliders
        if "short_window" not in st.session_state:
            st.session_state.short_window = 20
        if "long_window" not in st.session_state:
            st.session_state.long_window = 100

        col1, col2 = st.columns(2)

        with col1:
            short_window = st.slider(
                "Période moyenne courte",
                min_value=5,
                max_value=100,
                value=st.session_state.short_window,
                step=1,
                key="short_window_slider",
            )
        with col2:
            long_window = st.slider(
                "Période moyenne longue",
                min_value=20,
                max_value=300,
                value=st.session_state.long_window,
                step=5,
                key="long_window_slider",
            )

        # On synchronise les valeurs sliders -> session_state
        st.session_state.short_window = short_window
        st.session_state.long_window = long_window

        # Bouton d'optimisation
        if st.button("🔍 Optimiser les moyennes mobiles sur la période"):
            with st.spinner("Recherche des meilleurs paramètres de moyennes mobiles..."):
                result = _optimize_moving_average_params(prices, periods_per_year)

            if result is None:
                st.error("Impossible de trouver des paramètres optimaux (données insuffisantes ?).")
            else:
                best_short = result["short_window"]
                best_long = result["long_window"]
                best_metrics = result["metrics"]

                # Mise à jour des sliders via session_state
                st.session_state.short_window = best_short
                st.session_state.long_window = best_long

                st.success(
                    f"Meilleurs paramètres trouvés sur la période : "
                    f"MA courte = {best_short}, MA longue = {best_long} "
                    f"(rendement total ≈ {best_metrics['total_return'] * 100:.2f} %)."
                )

        if short_window >= long_window:
            st.warning("La période courte doit être strictement inférieure à la période longue.")
            st.stop()

    if prices.empty:
        st.warning("Aucune donnée de prix disponible.")
        return

    # ---------- 2.5. BENCHMARK & STRATÉGIE SÉLECTIONNÉE ----------
    benchmark_df = buy_and_hold(prices)

    if strategy_name == "Buy & Hold":
        strat_df = benchmark_df.copy()
    else:
        strat_df = moving_average_crossover(
            prices,
            short_window=st.session_state.short_window,
            long_window=st.session_state.long_window,
        )

    # ---------- 2.6. GRAPHIQUE COMPARATIF ----------
    st.subheader("Comparaison sur une base normalisée (valeur 1 au départ)")

    chart_df = pd.DataFrame(
        {
            "Buy & Hold CAC 40": benchmark_df["equity_curve"],
            "Stratégie sélectionnée": strat_df["equity_curve"],
        }
    )

    st.line_chart(chart_df)

    if strategy_name == "Moving Average Crossover":
        with st.expander("Afficher les moyennes mobiles utilisées"):
            ma_df = strat_df[["price", "ma_short", "ma_long"]].dropna()
            ma_df = ma_df.rename(columns={"price": "Prix CAC 40"})
            st.line_chart(ma_df)

    # ---------- 2.7. MÉTRIQUES : STRATÉGIE VS BUY & HOLD ----------
    st.subheader("Métriques de performance : stratégie vs Buy & Hold")

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
        st.metric("Rendement total", f"{benchmark_metrics['total_return'] * 100:.2f} %")
        st.metric("Rendement annualisé", f"{benchmark_metrics['annualized_return'] * 100:.2f} %")
        st.metric(
            "Volatilité annualisée",
            f"{benchmark_metrics['annualized_volatility'] * 100:.2f} %",
        )
        sharpe_bh = benchmark_metrics["sharpe_ratio"]
        sharpe_bh_str = "N/A" if pd.isna(sharpe_bh) else f"{sharpe_bh:.2f}"
        st.metric("Sharpe ratio", sharpe_bh_str)
        st.metric("Max drawdown", f"{benchmark_metrics['max_drawdown'] * 100:.2f} %")

    with col_strat:
        st.markdown("### Stratégie sélectionnée")
        st.metric("Rendement total", f"{strategy_metrics['total_return'] * 100:.2f} %")
        st.metric("Rendement annualisé", f"{strategy_metrics['annualized_return'] * 100:.2f} %")
        st.metric(
            "Volatilité annualisée",
            f"{strategy_metrics['annualized_volatility'] * 100:.2f} %",
        )
        sharpe_strat = strategy_metrics["sharpe_ratio"]
        sharpe_strat_str = "N/A" if pd.isna(sharpe_strat) else f"{sharpe_strat:.2f}"
        st.metric("Sharpe ratio", sharpe_strat_str)
        st.metric("Max drawdown", f"{strategy_metrics['max_drawdown'] * 100:.2f} %")

    # ---------- 2.8. INTERPRÉTATION QUALITATIVE ----------
    st.subheader("Comparaison qualitative")

    messages = _build_comparison_messages(benchmark_metrics, strategy_metrics)
    for msg in messages:
        st.markdown(msg)

    # ---------- 2.9. APERÇU DES DONNÉES BRUTES ----------
    with st.expander("Voir un extrait des données brutes"):
        st.dataframe(df.tail(10))
# ===================== FIN DU FICHIER =====================