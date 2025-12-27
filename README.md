# Binance Pair Trading with OU Process

A pair trading strategy implementation backtest using Binance data. This project utilizes rolling-window cointegration analysis and models the resulting spread using an Ornstein-Uhlenbeck (OU) process to determine optimal trading bands through statistical optimal stopping theory.
Based on info from the book "ALGORITHMIC AND HIGH-FREQUENCY TRADING" by CARTEA and JAIMUNGAL and PENALVA, and FrenchQuant videos.

## Features

- **Automated Data Management**: Fetches historical klines directly from Binance Futures API.
- **Statistical Calibration**:
    - Rolling window cointegration (Hedge ratio and ADF stationarity tests).
    - OU process parameter estimation using Maximum Likelihood Estimation (MLE) and Method of Moments (MoM).
- **Optimal Trading Bands**: Calculates entry and exit thresholds by solving the optimal stopping problem for a mean-reverting process.
- **Robust Backtesting**: Full backtest engine accounting for transaction fees, and turnover.
- **Visualization**: Generates detailed equity curves saved automatically to the `data/plots` directory.

## Project Structure

- `download_data.py`: Handles data ingestion and synchronization.
- `coint_calibrate.py`: Performs rolling cointegration analysis.
- `ou_calibrate.py`: Calibrates the OU process (Kappa, Mu, Sigma).
- `band_calc.py`: Computes optimal entry/exit bands.
- `backtest.py`: Executes the strategy and generates performance reports.
- `run_all.py`: The main entry point to run the full pipeline.
- `rank_coint_pairs.py`: Utility to find the most cointegrated pairs across the market.
- `config.json`: Centralized configuration for parameters, pairs, and intervals.

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies: `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`, `requests`, `pyarrow`

Install requirements:
```bash
pip install -r requirements.txt
```

### Usage

1. **Configure your pairs**: Edit `config.json` to define the assets you want to trade and your preferred rolling window.
2. **Run the pipeline**:
```bash
python run_all.py
```

Results including processed data and equity plots will be available in the `data/` directory.

## License

MIT
