# Quantitative Finance Dashboard  
**Python · Git · Linux for Finance**

---

## 1. Project Context

This project was developed as part of the **Python, Git, Linux for Finance** course.  
It simulates the work of a **quantitative research team** in an asset management company, whose role is to support portfolio managers with **quantitative tools**.

The objective is to design and deploy a **professional interactive dashboard** able to:
- retrieve financial market data from a dynamic source (API),
- implement quantitative strategies and backtesting,
- simulate multi-asset portfolios,
- display results and key metrics in a clear and user-friendly way.

The application is built in **Python with Streamlit**, versioned using **Git/GitHub**, and **deployed on a Linux virtual machine** to ensure continuous availability.

---

## 2. Team & Division of Work

This project was completed by **two students**, with a strict separation of responsibilities, as required by the project guidelines.

- **Maxime Farré — Quant A (Single Asset Analysis)**  
  Responsible for:
  - single-asset data loading and preprocessing,
  - implementation of quantitative strategies on one asset,
  - backtesting logic and performance metrics,
  - visualization of asset price vs strategy performance.

- **Emilien Combaret — Quant B (Multi-Asset Portfolio Analysis)**  
  Responsible for:
  - extension to multi-asset portfolios (minimum 3 assets),
  - portfolio allocation and rebalancing logic,
  - portfolio-level metrics and diversification analysis,
  - Streamlit user interface for portfolio configuration and results.

Both modules are integrated into a **single unified Streamlit application**.

---

## 3. Data Sources & API

Market data is retrieved from a **public financial API** through the Quant A data loader  
(e.g. *yfinance* or an equivalent public data provider).

Key characteristics:
- daily and intraday data (`1d`, `60m`, `15m`, `5m`),
- OHLCV data handling,
- support for MultiIndex formats,
- automatic refresh approximately every 5 minutes,
- robust handling of missing or invalid data.

---

## 4. Quant A — Single Asset Module

### Objective
Analyze and backtest **one asset at a time** (equities, FX, commodities, etc.).

### Implemented strategies
- **Buy & Hold**
- **Moving Average Crossover**
- **Regime Switching (Trend / Mean Reversion)**

### Metrics
For each strategy:
- cumulative return,
- annualized return,
- annualized volatility,
- Sharpe ratio,
- maximum drawdown.

### Visualization
- main chart showing:
  - raw asset price,
  - cumulative strategy value (base 100),
- interactive controls for:
  - strategy parameters,
  - data frequency selection.

---

## 5. Quant B — Multi-Asset Portfolio Module

### Objective
Extend the analysis to a **portfolio of multiple assets** (at least 3 simultaneously).

### Portfolio construction
Two portfolio modes are available:
1. **Fixed weights**
   - equal-weight,
   - custom user-defined weights.
2. **Strategy-based allocation**
   - portfolio weights derived from Quant A strategy signals.

### Allocation & rebalancing
- allocation rules:
  - equal-weight,
  - inverse volatility,
- rebalancing frequency:
  - none,
  - daily,
  - weekly,
  - monthly.

### Portfolio metrics
The module computes:
- portfolio value and returns (base 100),
- annualized volatility and return,
- CAGR,
- Sharpe ratio,
- maximum drawdown,
- correlation matrix,
- annualized covariance matrix,
- diversification ratio,
- effective number of assets,
- risk contributions by asset.

### Visualization
- comparison of individual assets vs portfolio value,
- correlation, covariance and distance heatmaps,
- tables of weights, volatilities and risk contributions.

---

## 6. Application Structure

The repository follows the structure below:

```text
PROJET/
├── .streamlit/
│   └── config.toml              # Streamlit configuration (layout, theme)
│
├── app/
│   ├── quant_a/                 # Quant A — Single Asset module
│   │   ├── data_loader.py       # Market data retrieval (API)
│   │   ├── strategies.py        # Single-asset strategies
│   │   ├── metrics.py           # Performance metrics
│   │   ├── optimizers.py        # Portfolio/strategy optimizers
│   │   ├── daily_report.py      # Daily report generation (cron)
│   │   ├── ui_quant_a.py        # Streamlit UI for Quant A
│   │   ├── universe.py          # Asset universe definition
│   │   └── readme_quant_a.md    # Quant A technical documentation
│   │
│   ├── quant_b/                 # Quant B — Multi-Asset Portfolio module
│   │   ├── data_adapter.py      # Adapter to reuse Quant A data loader
│   │   ├── strategies.py        # Multi-asset strategies
│   │   ├── portfolio.py         # Portfolio valuation logic
│   │   ├── backtest.py          # Portfolio backtesting & turnover
│   │   ├── metrics.py           # Portfolio & diversification metrics
│   │   └── page_quant_b.py      # Streamlit UI for Quant B
│   │
│   └── __init__.py
│
├── reports/                     # Daily reports generated via cron
│   ├── ...
│
├── main.py                      # Streamlit application entry point
├── .gitignore
├── README.md                    # Project documentation
└── todo.txt
