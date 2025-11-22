# app/quant_a/ui_quant_a.py

# ===================== 1. IMPORTS & FONCTIONS UTILITAIRES =====================

import datetime as dt

import pandas as pd
import streamlit as st

from .data_loader import load_cac40_history

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


def _get_periods_per_year(interval: str) -> int:
    """
    Renvoie le nombre de périodes par an en fonction de l'intervalle Yahoo.
    Approximations pour l'intraday :
    - 5m  : ~78 barres par jour * 252 jours ≈ 19 656
    - 15m : ~26 barres par jour * 252 jours ≈ 6 552
    - 60m : ~6.5 barres par jour * 252 jours ≈ 1 638
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




    """
    Optimise automatiquement les paramètres du modèle regime switching
    en testant un ensemble réduit de valeurs.
    """

    # Grilles raisonnables pour ne pas exploser le temps de calcul
    vol_short_list = [10, 20, 30]
    vol_long_list = [80, 120, 150]
    alpha_list = [0.9, 1.0, 1.1]
    trend_ma_list = [30, 50, 100]
    mr_window_list = [15, 20, 30]
    z_threshold_list = [0.8, 1.0, 1.2]

    best_score = -float("inf")
    best_params = None
    best_metrics = None

    for vs in vol_short_list:
        for vl in vol_long_list:
            if vl <= vs:
                continue

            for alpha in alpha_list:
                for ma_trend in trend_ma_list:
                    for mr_w in mr_window_list:
                        for z_th in z_threshold_list:

                            df = regime_switch_trend_meanrev(
                                prices,
                                vol_short_window=vs,
                                vol_long_window=vl,
                                alpha=alpha,
                                trend_ma_window=ma_trend,
                                mr_window=mr_w,
                                z_threshold=z_th,
                            )

                            metrics = compute_all_metrics(
                                df["equity_curve"],
                                df["strategy_returns"],
                                periods_per_year=periods_per_year,
                            )

                            score = metrics["total_return"]

                            if pd.notna(score) and score > best_score:
                                best_score = score
                                best_params = (vs, vl, alpha, ma_trend, mr_w, z_th)
                                best_metrics = metrics

    if best_params is None:
        return None

    return {
        "vol_short_window": best_params[0],
        "vol_long_window": best_params[1],
        "alpha": best_params[2],
        "trend_ma_window": best_params[3],
        "mr_window": best_params[4],
        "z_threshold": best_params[5],
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
    st.title("Quant A – Backtest univarié du CAC 40")

    st.markdown(
        """
        Ce module sert de moteur d'analyse pour le **Compte A** : il backteste des
        stratégies systématiques sur le **CAC 40** à partir de données Yahoo Finance.

        Fonctions principales :

        - Sélection de la **période** et de la **périodicité** (journalier / hebdomadaire / mensuel),
        - Choix de la **stratégie** :
          - Buy & Hold,
          - Crossover de moyennes mobiles,
          - Regime Switching (Trend + Mean-Reversion),
        - **Optimisation automatique** de certains paramètres (moyennes mobiles, régime),
        - Visualisation :
          - courbe de prix, équity, drawdown,
          - régimes de volatilité et signaux de trading,
        - Analyse :
          - métriques de performance (Sharpe, volatilité, drawdown…),
          - métriques de trading (nombre de trades, win rate, longs vs shorts),
          - historique détaillé des trades.
        """
    )

    st.caption(
        "Les résultats et signaux produits ici sont réutilisables par le Compte B "
        "pour la construction d'un portefeuille multi-stratégies."
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
                "Intraday 5 minutes (5m)",
                "Intraday 15 minutes (15m)",
                "Intraday 1 heure (60m)",
            ],
            index=0,
        )

        interval_map = {
            "Journalier (1d)": "1d",
            "Hebdomadaire (1wk)": "1wk",
            "Mensuel (1mo)": "1mo",
            "Intraday 5 minutes (5m)": "5m",
            "Intraday 15 minutes (15m)": "15m",
            "Intraday 1 heure (60m)": "60m",
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
        # ---------- 2.3. CHARGEMENT DES DONNÉES ----------
    # Mise en cache des données pour un rafraîchissement automatique toutes les 5 minutes

    now = dt.datetime.now()

    # On garde les dates sous forme de chaînes pour les comparer facilement
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Initialisation des clés de cache dans la session si besoin
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

    need_reload = False

    # 1) Si pas de données en cache, on doit charger
    if st.session_state.cached_df is None or st.session_state.cached_df.empty:
        need_reload = True
    else:
        # 2) Si l'utilisateur change la période ou l'intervalle, on recharge
        if (
            st.session_state.last_interval != interval
            or st.session_state.last_start_date != start_str
            or st.session_state.last_end_date != end_str
        ):
            need_reload = True
        else:
            # 3) Sinon, on regarde si plus de 5 minutes se sont écoulées
            last_time = st.session_state.last_load_time
            if last_time is None or (now - last_time).total_seconds() > 300:
                need_reload = True

    if need_reload:
        with st.spinner("Chargement des données du CAC 40..."):
            df = load_cac40_history(
                start=start_str,
                end=end_str,
                interval=interval,
            )

        st.session_state.cached_df = df
        st.session_state.last_load_time = now
        st.session_state.last_interval = interval
        st.session_state.last_start_date = start_str
        st.session_state.last_end_date = end_str
    else:
        df = st.session_state.cached_df

    if df is None or df.empty:
        st.warning("Aucune donnée disponible pour le CAC 40 sur cette période.")
        return

    prices = df["Close"].astype(float).copy()
    prices = prices.sort_index()
    prices.name = "CAC 40 (Close)"


    # ---------- 2.4. CONTRÔLES STRATÉGIE + BOUTON D'OPTIMISATION ----------
    st.subheader("Paramètres de stratégie")

    strategy_name = st.selectbox(
    "Choix de la stratégie",
    ["Buy & Hold",
     "Moving Average Crossover",
     "Regime Switching (Trend + Mean-Reversion)"],
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
                result = optimize_moving_average_params(prices, periods_per_year)

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

    # ===================== PARAMÈTRES REGIME SWITCHING =====================
    if strategy_name == "Regime Switching (Trend + Mean-Reversion)":

        st.markdown("### Paramètres Regime Switching")

        col_vol1, col_vol2, col_alpha = st.columns(3)
        with col_vol1:
            vol_short_window = st.slider(
                "Fenêtre volatilité courte",
                min_value=5,
                max_value=60,
                value=30,
            )
        with col_vol2:
            vol_long_window = st.slider(
                "Fenêtre volatilité longue",
                min_value=50,
                max_value=300,
                value=80,
            )
        with col_alpha:
            alpha = st.slider(
                "Seuil de changement de régime (α)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
            )

        st.markdown("### Paramètres Trend-Following")
        trend_ma_window = st.slider(
            "Fenêtre moyenne mobile (Trend)",
            min_value=10,
            max_value=200,
            value=30,
        )

        st.markdown("### Paramètres Mean-Reversion")
        col_mr1, col_mr2 = st.columns(2)
        with col_mr1:
            mr_window = st.slider(
                "Fenêtre Mean-Reversion",
                min_value=10,
                max_value=60,
                value=30,
            )
        with col_mr2:
            z_threshold = st.slider(
                "Seuil Z-score",
                min_value=0.5,
                max_value=3.0,
                value=0.8,
                step=0.1,
            )
        
    
        # ----- Bouton d'optimisation automatique -----
        if "optimize_regime" not in st.session_state:
            st.session_state.optimize_regime = False

        if st.button("🔍 Optimiser automatiquement les paramètres", key="optimize_regime_button"):
            st.session_state.optimize_regime = True


        # ----- exécution de l’optimisation -----
        if st.session_state.optimize_regime:
            with st.spinner("Optimisation en cours..."):
                result = optimize_regime_switching(prices, periods_per_year)

            st.session_state.optimize_regime = False  # évite les reruns infinis

            if result is None:
                st.error("Impossible de trouver des paramètres optimaux.")
            else:
                st.success(
                    f"Paramètres optimaux trouvés : "
                    f"vol_short={result['vol_short_window']}, "
                    f"vol_long={result['vol_long_window']}, "
                    f"α={result['alpha']}, "
                    f"MA_trend={result['trend_ma_window']}, "
                    f"MR_window={result['mr_window']}, "
                    f"Z={result['z_threshold']} "
                    f"(Rendement total ≈ {result['metrics']['total_return']*100:.2f} %)"
                )

                # On met à jour les sliders directement via session_state
                st.session_state["vol_short_window"] = result["vol_short_window"]
                st.session_state["vol_long_window"] = result["vol_long_window"]
                st.session_state["alpha"] = result["alpha"]
                st.session_state["trend_ma_window"] = result["trend_ma_window"]
                st.session_state["mr_window"] = result["mr_window"]
                st.session_state["z_threshold"] = result["z_threshold"]



    if prices.empty:
        st.warning("Aucune donnée de prix disponible.")
        return

    # ---------- 2.5. BENCHMARK & STRATÉGIE SÉLECTIONNÉE ----------
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
        st.error("Stratégie inconnue.")
        st.stop()

        strat_df = moving_average_crossover(
            prices,
            short_window=st.session_state.short_window,
            long_window=st.session_state.long_window,
        )
    
    # ---------- 2.5.bis. EXTRACTION DES TRADES ----------
    trades_df = extract_trades_from_position(
        strat_df["price"],
        strat_df["position"],
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
            
    if strategy_name == "Regime Switching (Trend + Mean-Reversion)":
        with st.expander("📊 Détails du modèle (debug)"):
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

    # ---------- 2.7.bis. MÉTRIQUES DE TRADING (STRATÉGIE SÉLECTIONNÉE) ----------
    st.subheader("Métriques de trading de la stratégie sélectionnée")

    if trades_df.empty:
        st.info("Aucun trade détecté pour cette stratégie sur la période.")
    else:
        trade_metrics = compute_trade_metrics(trades_df)

        n_trades = trade_metrics["n_trades"]
        win_rate = trade_metrics["win_rate"]
        pct_longs = trade_metrics["pct_longs"]
        pct_shorts = trade_metrics["pct_shorts"]
        avg_trade_return = trade_metrics["avg_trade_return"]
        avg_win_return = trade_metrics["avg_win_return"]
        avg_loss_return = trade_metrics["avg_loss_return"]
        avg_holding_period = trade_metrics["avg_holding_period"]

        col_t1, col_t2, col_t3 = st.columns(3)

        col_t1.metric("Nombre de trades", f"{n_trades}")
        col_t2.metric(
            "Taux de trades gagnants",
            f"{win_rate * 100:.2f} %" if win_rate == win_rate else "N/A",
        )
        col_t3.metric(
            "Trades LONG vs SHORT",
            (
                f"{pct_longs * 100:.1f} % long / {pct_shorts * 100:.1f} % short"
                if pct_longs == pct_longs and pct_shorts == pct_shorts
                else "N/A"
            ),
        )

        col_t4, col_t5, col_t6 = st.columns(3)
        col_t4.metric(
            "Rendement moyen par trade",
            f"{avg_trade_return * 100:.2f} %" if avg_trade_return == avg_trade_return else "N/A",
        )
        col_t5.metric(
            "Gain moyen (trades gagnants)",
            f"{avg_win_return * 100:.2f} %" if avg_win_return == avg_win_return else "N/A",
        )
        col_t6.metric(
            "Perte moyenne (trades perdants)",
            f"{avg_loss_return * 100:.2f} %" if avg_loss_return == avg_loss_return else "N/A",
        )

        st.metric(
            "Durée moyenne des trades (en barres)",
            f"{avg_holding_period:.1f}" if avg_holding_period == avg_holding_period else "N/A",
        )

        with st.expander("📜 Historique détaillé des trades"):
            display_cols = [
                "entry_date",
                "exit_date",
                "direction",
                "entry_price",
                "exit_price",
                "holding_period_bars",
                "trade_return_pct",
            ]
            st.dataframe(trades_df[display_cols])

    # ---------- 2.8. INTERPRÉTATION QUALITATIVE ----------
    st.subheader("Comparaison qualitative")

    messages = _build_comparison_messages(benchmark_metrics, strategy_metrics)
    for msg in messages:
        st.markdown(msg)

    # ---------- 2.9. APERÇU DES DONNÉES BRUTES ----------
    with st.expander("Voir un extrait des données brutes"):
        st.dataframe(df.tail(10))
