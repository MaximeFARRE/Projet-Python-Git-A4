# Quant A – Moteur univarié CAC 40

Ce module correspond au **Compte A** du projet :  
il fournit un moteur de backtest **univarié sur le CAC 40**, réutilisable par le **Compte B** (gestion de portefeuille).

L’interface publique de Quant A permet de :

- charger les prix du CAC 40,
- exécuter plusieurs stratégies (Buy & Hold, Moving Average Crossover, Regime Switching Trend+Mean-Reversion),
- extraire l’historique des trades,
- calculer des métriques de performance globales,
- calculer des métriques de trading (trade par trade).

Ce README décrit uniquement les **fonctions à considérer comme API publique** pour le compte B.


## 1. Structure des fichiers

Dans `app/quant_a/` :

- `data_loader.py`  
  → chargement des données CAC 40 (yfinance).

- `strategies.py`  
  → implémentation des stratégies univariées et extraction des trades.

- `metrics.py`  
  → calcul des métriques de performance et de trading.

- `ui_quant_a.py`  
  → interface Streamlit pour visualiser et tester les stratégies.  
  ⚠ **Le compte B ne doit pas dépendre de ce fichier.**


## 2. Formats standards utilisés

### 2.1. Série de prix

Partout dans Quant A, on utilise :

- `prices`: `pandas.Series`
  - index : `pandas.DatetimeIndex` (dates)
  - valeurs : float (prix de clôture du CAC 40)

### 2.2. DataFrame de stratégie (`strat_df`)

Toutes les fonctions de stratégie retournent un `pandas.DataFrame` avec au minimum :

- `price` : float  
- `position` : float (entre -1 et +1)  
- `strategy_returns` : float (rendements de la stratégie, en décimal, ex: 0.01 = +1 %)  
- `equity_curve` : float (valeur cumulée, base 1 au début)

Colonnes supplémentaires possibles (suivant la stratégie) :

- `ma_short`, `ma_long` (pour MA Crossover)
- `regime` ("TREND" / "MR")
- `vol_short`, `vol_long`
- `ma_trend`
- `mr_mu`, `mr_sigma`
- `zscore`
- `trend_signal`, `mr_signal`

Le compte B n’a besoin que des colonnes **standard** pour fonctionner :  
`price`, `position`, `strategy_returns`, `equity_curve`.

### 2.3. DataFrame de trades (`trades_df`)

Format :

- `entry_date` : datetime (index temporel ou colonne)  
- `exit_date` : datetime  
- `direction` : "LONG" ou "SHORT"  
- `entry_price` : float  
- `exit_price` : float  
- `holding_period_bars` : int (nombre de barres entre entrée et sortie)  
- `trade_return` : float (rendement décimal du trade, ex: 0.05 = +5 %)  
- `trade_return_pct` : float (rendement en %, ex: 5.0 = +5 %)


## 3. Module `data_loader.py`

### 3.1. `load_cac40_history`

# API Quant A (Compte A)

Ce document décrit toutes les fonctions publiques disponibles pour le Compte B.
Il sert de référence pour réutiliser les stratégies, les données et les métriques
dans un code externe (Compte B) sans dépendre de Streamlit.

====================================================================
1) MODULE : data_loader.py
====================================================================

FUNCTION:
    load_cac40_history(start=None, end=None, interval="1d")

DESCRIPTION:
    Charge l'historique du CAC40 (^FCHI) via Yahoo Finance.

PARAMÈTRES:
    - start (str, ex: "2020-01-01")
    - end   (str)
    - interval ("1d", "1wk", "1mo")

RETURNE:
    pandas.DataFrame indexé par date, avec colonnes:
    Open, High, Low, Close, Adj Close, Volume


FUNCTION:
    get_last_cac40_close()

DESCRIPTION:
    Renvoie le dernier prix de clôture du CAC40.

RETURN:
    float


====================================================================
2) MODULE : strategies.py
====================================================================

Toutes les stratégies renvoient un DataFrame standard avec au minimum :
    - price
    - position   (float, entre -1 et +1)
    - strategy_returns
    - equity_curve

---------------------------------------------------------------
FUNCTION:
    buy_and_hold(prices)

DESCRIPTION:
    Stratégie Buy & Hold (position = 1 en permanence).

PARAMÈTRES:
    - prices (pd.Series): série des prix (Close)

RETURN:
    strat_df minimal (voir format standard)


---------------------------------------------------------------
FUNCTION:
    moving_average_crossover(prices,
                             short_window=50,
                             long_window=200)

DESCRIPTION:
    Stratégie croisement de moyennes mobiles :
        position = 1 si MA_courte > MA_longue
        position = 0 sinon

PARAMÈTRES:
    - prices (pd.Series)
    - short_window (int)
    - long_window (int)

RETURNE:
    strat_df standard + colonnes:
        ma_short
        ma_long


---------------------------------------------------------------
FUNCTION:
    regime_switch_trend_meanrev(prices,
                                vol_short_window=20,
                                vol_long_window=100,
                                alpha=1.0,
                                trend_ma_window=50,
                                mr_window=20,
                                z_threshold=1.0)

DESCRIPTION:
    Stratégie avancée avec changement de régime :
        Régime TREND : suivi de tendance via MA
        Régime MR    : retour à la moyenne via z-score

PARAMÈTRES:
    - prices (pd.Series)
    - vol_short_window (int)   : fenêtre volatilité court terme
    - vol_long_window (int)    : fenêtre volatilité long terme
    - alpha (float)            : seuil de bascule de régime
    - trend_ma_window (int)    : fenêtre MA pour TREND
    - mr_window (int)          : fenêtre pour moyenne et sigma MR
    - z_threshold (float)      : seuil d’activation MR

RETURNE:
    strat_df standard + colonnes avancées :
        regime          (TREND / MR)
        vol_short
        vol_long
        ma_trend
        mr_mu
        mr_sigma
        zscore
        trend_signal
        mr_signal


---------------------------------------------------------------
FUNCTION:
    extract_trades_from_position(prices, position)

DESCRIPTION:
    Reconstruit l’historique complet des trades à partir d’une série
    de prix et d’une série de positions (-1, 0, +1).

PARAMÈTRES:
    - prices (pd.Series)
    - position (pd.Series)

RETURNE:
    trades_df avec colonnes :
        entry_date
        exit_date
        direction (LONG/SHORT)
        entry_price
        exit_price
        holding_period_bars
        trade_return         (décimal)
        trade_return_pct     (%)    


====================================================================
3) MODULE : metrics.py
====================================================================

---------------------------------------------------------------
FUNCTION:
    compute_all_metrics(equity_curve,
                        returns,
                        risk_free_rate,
                        periods_per_year)

DESCRIPTION:
    Calcule les métriques globales de performance d’une stratégie.

PARAMÈTRES:
    - equity_curve (pd.Series)
    - returns (pd.Series)
    - risk_free_rate (float)
    - periods_per_year (int) ex: 252 (daily)

RETURNE (dict):
    {
        "total_return": float,
        "annualized_return": float,
        "annualized_volatility": float,
        "sharpe_ratio": float,
        "max_drawdown": float
    }


---------------------------------------------------------------
FUNCTION:
    compute_trade_metrics(trades_df)

DESCRIPTION:
    Calcule des métriques au niveau "trade" à partir de trades_df.

PARAMÈTRES:
    - trades_df (pd.DataFrame)

RETURNE (dict):
    {
        "n_trades": int,
        "win_rate": float,
        "pct_longs": float,
        "pct_shorts": float,
        "avg_trade_return": float,
        "avg_win_return": float,
        "avg_loss_return": float,
        "avg_holding_period": float
    }


====================================================================
4) SUMMARY – FONCTIONS QUE LE COMPTE B DOIT UTILISER
====================================================================

DATA :
    - load_cac40_history
    - get_last_cac40_close

STRATÉGIES :
    - buy_and_hold
    - moving_average_crossover
    - regime_switch_trend_meanrev

TRADES :
    - extract_trades_from_position

MÉTRIQUES :
    - compute_all_metrics
    - compute_trade_metrics

====================================================================
FIN DU README TECHNIQUE (API QUANT A)
====================================================================
