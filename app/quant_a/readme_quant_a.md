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

```python
from app.quant_a.data_loader import load_cac40_history

load_cac40_history(
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame
