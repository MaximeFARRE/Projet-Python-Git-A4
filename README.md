# Quantitative Finance Dashboard  
**Python · Git · Linux for Finance**

## Overview

This project is a **quantitative finance dashboard** developed as part of the *Python, Git, Linux for Finance* course.  
It simulates a **professional asset management workflow**, combining real-time data retrieval, quantitative strategies, portfolio backtesting, and interactive visualization.

The application is built with **Python and Streamlit**, collaboratively developed using **Git/GitHub**, and **deployed on a Linux virtual machine** to ensure continuous availability.

---

## Team & Responsibilities

This project was completed by a **two-student team**, with a strict separation of responsibilities:

- **Maxime Farré — Quant A (Single Asset Analysis)**
  - Univariate strategies and backtesting
  - Single-asset metrics and visualizations
  - Strategy logic reused by Quant B

- **Emilien Combaret — Quant B (Multi-Asset Portfolio)**
  - Multi-asset portfolio construction
  - Allocation rules and rebalancing
  - Portfolio-level metrics, diversification analysis, and UI

Both modules are fully integrated into a **single unified dashboard**.

---

## Key Features

### Data & Infrastructure
- Market data retrieved from a **dynamic public API** (via Quant A data loader, e.g. *yfinance*)
- Automatic data refresh (approximately every 5 minutes)
- Robust data handling (OHLCV formats, MultiIndex support, missing data)
- Application deployed on a **Linux VM**, designed to run **24/7**

---

## Quant A — Single Asset Module

**Focus:** One asset at a time (e.g. equities, FX, commodities)

**Features:**
- Backtesting strategies:
  - Buy & Hold
  - Moving Average Crossover
  - Regime Switching (Trend / Mean Reversion)
- Performance metrics:
  - Cumulative return
  - Annualized return & volatility
  - Sharpe ratio
  - Maximum drawdown
- Interactive controls:
  - Strategy parameters
  - Time interval selection
- Main visualization:
  - Asset price vs strategy cumulative value

---

## Quant B — Multi-Asset Portfolio Module

**Focus:** Portfolio analysis with **at least 3 assets simultaneously**

**Features:**
- Portfolio construction:
  - Fixed weights (equal-weight or custom)
  - Strategy-driven allocation (signals from Quant A)
- Allocation rules:
  - Equal-weight
  - Inverse volatility
- Rebalancing frequencies:
  - Daily, Weekly, Monthly, or None
- Portfolio metrics:
  - Correlation and covariance matrices
  - Portfolio volatility and returns
  - CAGR, Sharpe ratio, maximum drawdown
  - Diversification ratio and effective number of assets
  - Risk contributions by asset
- Visual analysis:
  - Individual assets vs portfolio (base 100)
  - Heatmaps (correlation, covariance, distance)
- Interactive **Streamlit UI** with tabs:
  - Overview
  - Diversification
  - Details

---

## Technical Stack

- **Language:** Python 3
- **Framework:** Streamlit
- **Libraries:** NumPy, Pandas, Plotly
- **Data source:** Public market data API (*yfinance* via Quant A loader)
- **Version control:** Git & GitHub
- **Deployment:** Linux Virtual Machine
- **Automation:** Ready for cron-based daily reporting

---

## Repository Structure

```text
app/
├── quant_a/          # Single Asset module (strategies, data loader, metrics)
├── quant_b/          # Multi-Asset portfolio module
│   ├── strategies.py
│   ├── portfolio.py
│   ├── metrics.py
│   ├── backtest.py
│   └── page_quant_b.py
├── main.py           # Streamlit entry point
└── README.md
