import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for safer plotting in scripts/parallel processes
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from utils import load_config, pair_id, get_dirs, load_coint_data, load_bands_data


def backtest_pair(pair: Dict, config: Dict) -> pd.DataFrame:
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 30))
    _, intermediate_dir, output_dir = get_dirs(config)

    coint_path = os.path.join(
        intermediate_dir,
        f"coint_{pair_id(pair)}_{interval}_w{window}.feather",
    )
    bands_path = os.path.join(
        intermediate_dir,
        f"bands_{pair_id(pair)}_{interval}_w{window}.feather",
    )

    coint_df = load_coint_data(coint_path)
    bands_df = load_bands_data(bands_path)

    df = coint_df.join(bands_df, how="left", rsuffix="_bands")
    n = len(df)
    if n == 0:
        raise ValueError("No data available for backtest.")

    pos_arr = np.zeros(n)
    epsilon = df["epsilon"].values
    lower = df["lower"].values
    upper = df["upper"].values
    mu = df["mu"].values

    for i in range(1, n):
        if not (
            np.isfinite(epsilon[i])
            and np.isfinite(lower[i])
            and np.isfinite(upper[i])
            and np.isfinite(mu[i])
        ):
            pos_arr[i] = 0.0
            continue

        prev_pos = pos_arr[i - 1]
        new_pos = prev_pos

        if prev_pos == 0:
            if epsilon[i] < lower[i]:
                new_pos = 1
            elif epsilon[i] > upper[i]:
                new_pos = -1
        elif prev_pos == 1 and epsilon[i] >= mu[i]:
            new_pos = 0
        elif prev_pos == -1 and epsilon[i] <= mu[i]:
            new_pos = 0

        pos_arr[i] = new_pos

    flip_signals = bool(config.get("flip_signals", False))
    if flip_signals:
        pos_arr = -pos_arr
    df["position"] = pos_arr

    returns_y = df["y_close"].pct_change()
    returns_x = df["x_close"].pct_change()
    beta_series = df["beta"].shift(1)

    spread_return = (returns_y - beta_series * returns_x) / (1 + beta_series.abs())
    strategy_return = df["position"].shift(1) * spread_return
    strategy_return = strategy_return.fillna(0.0)

    fee_rate = float(config.get("fee_rate", 0.001))
    turnover = df["position"].diff().abs().fillna(0.0)
    fee_cost = fee_rate * turnover
    strategy_return = strategy_return - fee_cost

    start_equity = float(config.get("start_equity", 1000))
    equity = start_equity * (1 + strategy_return).cumprod()

    df["strategy_return"] = strategy_return
    df["equity"] = equity
    df["turnover"] = turnover
    df["fee_cost"] = fee_cost

    out_path = os.path.join(
        output_dir,
        f"backtest_{pair_id(pair)}_{interval}_w{window}.feather",
    )
    df.reset_index().to_feather(out_path)
    print(f"Saved backtest results to {out_path}")
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