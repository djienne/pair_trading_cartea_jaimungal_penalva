import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for safer plotting in scripts/parallel processes
import matplotlib.pyplot as plt

from utils import (
    load_config,
    pair_id,
    get_dirs,
    load_pair_data,
    save_pair_data,
)


def backtest_pair(pair: Dict, config: Dict) -> pd.DataFrame:
    # Load data using consolidated utilities
    coint_df = load_pair_data(pair, config, "coint")
    bands_df = load_pair_data(pair, config, "bands")

    # Merge coint and bands data
    df = coint_df.join(bands_df, how="left", rsuffix="_bands")
    n = len(df)
    if n == 0:
        raise ValueError("No data available for backtest.")

    # Prepare arrays for fast iteration
    y = df["y_close"].values
    x = df["x_close"].values
    # alpha = df["alpha"].values # Not explicitly needed for trading delta, but part of signal
    beta = df["beta"].values
    epsilon = df["epsilon"].values
    lower = df["lower"].values
    upper = df["upper"].values
    mu = df["mu"].values

    # Simulation State
    pos = 0          # Current position: +1 (Long Portfolio), -1 (Short Portfolio), 0 (Flat)
    m1 = 0.0         # Quantity of Asset 1 (Y)
    m2 = 0.0         # Quantity of Asset 2 (X)
    cash = float(config.get("start_equity", 1000)) # Start with all cash
    
    # We track Book Value (BV) = Cash + Market Value of Positions
    book_value_arr = np.zeros(n)
    pos_arr = np.zeros(n)
    m1_arr = np.zeros(n)
    m2_arr = np.zeros(n)
    cash_arr = np.zeros(n)
    turnover_arr = np.zeros(n)

    fee_rate = float(config.get("fee_rate", 0.001))
    
    # Initial state
    book_value_arr[0] = cash
    cash_arr[0] = cash

    for i in range(1, n):
        # Skip if any required data is missing (NaN)
        if not (np.isfinite(epsilon[i]) and np.isfinite(lower[i]) and np.isfinite(upper[i])):
            # Carry forward state
            book_value_arr[i] = cash + m1 * y[i] + m2 * x[i]
            pos_arr[i] = pos
            m1_arr[i] = m1
            m2_arr[i] = m2
            cash_arr[i] = cash
            continue

        z = epsilon[i]
        curr_lower = lower[i]
        curr_upper = upper[i]
        curr_mu = mu[i]
        curr_beta = beta[i]
        
        # Prices
        py = y[i]
        px = x[i]

        prev_pos = pos
        
        # --- Trading Logic ---
        # 1. Check Entries
        if pos == 0:
            if z <= curr_lower:
                # Enter LONG Portfolio: Buy 1 unit of Y, Sell beta units of X
                pos = 1
                target_m1 = 1.0
                target_m2 = -curr_beta
            elif z >= curr_upper:
                # Enter SHORT Portfolio: Sell 1 unit of Y, Buy beta units of X
                pos = -1
                target_m1 = -1.0
                target_m2 = curr_beta
            else:
                # Stay Flat
                target_m1 = 0.0
                target_m2 = 0.0
        
        # 2. Check Exits
        elif pos == 1: # Currently Long
            if z >= curr_mu:
                # Exit to Flat
                pos = 0
                target_m1 = 0.0
                target_m2 = 0.0
            else:
                # Hold (Simple hold, no re-hedging logic in this basic version to match simple backtest speed)
                # Note: In a full dynamic hedge, we might adjust m2 to match new beta. 
                # For now, we hold the initial bundle until exit.
                target_m1 = m1
                target_m2 = m2

        elif pos == -1: # Currently Short
            if z <= curr_mu:
                # Exit to Flat
                pos = 0
                target_m1 = 0.0
                target_m2 = 0.0
            else:
                # Hold
                target_m1 = m1
                target_m2 = m2
        
        # --- Execution ---
        # Calculate turnover and costs
        dm1 = target_m1 - m1
        dm2 = target_m2 - m2
        
        trade_value = abs(dm1 * py) + abs(dm2 * px)
        cost = trade_value * fee_rate
        
        # Update cash: Cash decreases by cost of buying assets, increases by selling
        # Cost of buying dm1 of Y is (dm1 * py)
        cash -= (dm1 * py + dm2 * px)
        cash -= cost
        
        m1 = target_m1
        m2 = target_m2
        
        turnover_arr[i] = trade_value
        pos_arr[i] = pos
        m1_arr[i] = m1
        m2_arr[i] = m2
        cash_arr[i] = cash
        
        # Mark to Market
        book_value_arr[i] = cash + m1 * py + m2 * px

    df["position"] = pos_arr
    df["m1"] = m1_arr
    df["m2"] = m2_arr
    df["cash"] = cash_arr
    df["equity"] = book_value_arr
    df["turnover"] = turnover_arr
    
    # Calculate returns for metrics
    df["strategy_return"] = df["equity"].pct_change().fillna(0.0)

    # Save using consolidated utility
    save_pair_data(df, pair, config, "backtest")
    return df


def plot_equity(
    results: pd.DataFrame,
    pair_name: str,
    save_path: str = None,
    show_plot: bool = True,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["equity"], label="Equity", linewidth=1.5)
    plt.title(f"Pair Trading Equity ({pair_name})")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved equity plot to {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def main(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    pairs = config.get("pairs", [])
    if not pairs:
        raise ValueError("No pairs configured in config.json")
        
    # Prepare plots directory
    data_dir = config.get("data_dir", "data")
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 30))

    for pair in pairs:
        pair_name = pair.get("name") or pair_id(pair)
        print(f"Running backtest for {pair_name}...")
        results = backtest_pair(pair, config)
        print(f"Final equity for {pair_name}: {results['equity'].iloc[-1]:.2f} USD")
        
        # Construct meaningful filename
        start_date = results.index[0].strftime('%Y%m%d')
        end_date = results.index[-1].strftime('%Y%m%d')
        
        # Format: equity_{PAIR}_{INTERVAL}_w{WINDOW}_{START}-{END}.png
        filename = f"equity_{pair_name}_{interval}_w{window}_{start_date}-{end_date}.png"
        
        # Sanitize filename (replace forbidden characters)
        filename = filename.replace(os.path.sep, "_").replace(":", "")
        
        save_path = os.path.join(plots_dir, filename)
        show_plot = bool(config.get("show_plots", True))
        plot_equity(results, pair_name, save_path, show_plot=show_plot)


if __name__ == "__main__":
    main()
