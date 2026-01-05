# Pair Trading: CARTEA-JAIMUNGAL-PENALVA Method

A pair trading strategy implementation using Binance data daily candles (backtest only). The trading strategy utilizes rolling-window cointegration analysis and models the resulting spread using an Ornstein-Uhlenbeck (OU) process to determine optimal trading bands through statistical optimal stopping theory.

It is based on info from the book "ALGORITHMIC AND HIGH-FREQUENCY TRADING" by Cartea and Jaimungal and Penalva, and FrenchQuant videos ( https://youtu.be/_Sq6KoP7m1c?si=2N9ufvkx3fcU7zZe   https://youtu.be/DeqpOFrH_Bg?si=YdeAJMz_xz04e2bd   https://youtu.be/EYRk5nk6eDA?si=dBLKiuJM68GLDys0 ).

## Features

- **Automated Data Management**: Fetches historical klines directly from Binance Futures API.
- **Statistical Calibration**:
    - Rolling window cointegration (Hedge ratio and ADF stationarity tests).
    - **Bayesian Rolling Regression**: Dynamic hedge ratio estimation using PyMC with random-walk coefficients (ADVI or NUTS inference).
    - OU process parameter estimation using Maximum Likelihood Estimation (MLE) and Method of Moments (MoM).
- **Optimal Trading Bands**: Calculates entry and exit thresholds by solving the optimal stopping problem for a mean-reverting process.
- **Robust Backtesting**: Full backtest engine accounting for transaction fees and turnover.
- **Pair Ranking**: Automated ranking of cointegrated pairs across the market.
- **Visualization**: Generates detailed equity curves saved automatically to the `data/plots` directory.

## Example Equity Curve

<p align="center">
  <img src="data/plots/equity_BNB-SOL_1d_w300_20200914-20251227.png" alt="Equity curve for BNB-SOL 1d window 300" width="700" />
</p>

## Project Structure

### Core Pipeline
- `download_data.py`: Handles data ingestion and synchronization from Binance Futures API.
- `coint_calibrate.py`: Performs rolling OLS cointegration analysis with ADF tests.
- `bayesian_coint.py`: Bayesian rolling hedge ratio calibration using PyMC (alternative to OLS).
- `rolling_bayesian_hedge.py`: Bayesian random-walk regression engine with ADVI/NUTS inference.
- `ou_calibrate.py`: Calibrates the OU process parameters (Kappa, Mu, Sigma).
- `band_calc.py`: Computes optimal entry/exit bands via optimal stopping theory.
- `backtest.py`: Executes the strategy and generates performance reports.

### Orchestration
- `run_all.py`: Main entry point to run the full pipeline sequentially.
- `run_and_rank.py`: Bulk backtest runner with parallel execution and ranking.

### Utilities
- `utils.py`: Shared utilities for data loading, path management, and configuration.
- `rank_coint_pairs.py`: Find the most cointegrated pairs across the market.
- `rank_backtests.py`: Summarize and rank backtest results.

### Reference
- `QuantPy_OU_process/`: Reference implementation of OU process calibration using MLE.
- `pair_trading_video*_frenchquant.py`: Educational scripts from FrenchQuant video tutorials.

### Configuration
- `config.json`: Centralized configuration for parameters, pairs, and intervals.
  - `transaction_cost`: Used in `band_calc.py` to shift the optimal OU bands.
  - `fee_rate`: Used in `backtest.py` to apply turnover-based trading fees.
  - `bayesian_inference_method`: Choose `"advi"` (fast) or `"nuts"` (precise) for Bayesian calibration.

## Getting Started

### Prerequisites

- Python 3.9+
- Dependencies: `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`, `requests`, `pyarrow`, `pymc`, `tqdm`

Install requirements:
```bash
pip install -r requirements.txt
```

> **Note**: PyMC requires a C compiler for PyTensor. On Windows, install Visual Studio Build Tools. On Linux/Mac, ensure `gcc` is available.

### Usage

1. **Configure your pairs**: Edit `config.json` to define the assets you want to trade and your preferred rolling window.
2. **Run the pipeline**:
```bash
python run_all.py
```

Results including processed data and equity plots will be available in the `data/` directory.

## License

MIT
