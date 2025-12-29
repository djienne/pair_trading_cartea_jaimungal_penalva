import json
import os
import sys
import pandas as pd
import itertools
import contextlib
import concurrent.futures as cf
from typing import Dict, List, Optional, Tuple

# Add current directory to path
sys.path.append(os.getcwd())

# Import necessary functions
try:
    from coint_calibrate import calibrate_pair as coint_calibrate_pair
    from ou_calibrate import calibrate_pair as ou_calibrate_pair
    from band_calc import calculate_bands
    from backtest import backtest_pair, plot_equity
    from utils import load_config, pair_id, get_dirs
except ImportError:
    print("Error: Could not import required modules. Ensure you are in the project root.")
    sys.exit(1)


def get_available_symbols(config: Dict, min_rows: int) -> List[str]:
    feather_dir, _, _ = get_dirs(config)
    interval = config.get("candle_interval", "1d")
    suffix = f"_{interval}.feather"
    
    available = []
    if not os.path.exists(feather_dir):
        return []
        
    for f in os.listdir(feather_dir):
        if f.endswith(suffix):
            path = os.path.join(feather_dir, f)
            try:
                # Check file size/rows quickly
                df = pd.read_feather(path)
                if len(df) >= min_rows:
                    symbol = f[: -len(suffix)]
                    available.append(symbol)
            except:
                pass
    return sorted(list(set(available))) 

@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout to keep logs clean."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def process_single_pair(args: Tuple[str, str, Dict]) -> Optional[Dict]:
    sym_y, sym_x, config = args
    pair = {
        "y_symbol": sym_y,
        "x_symbol": sym_x,
        "name": f"{sym_y}-{sym_x}"
    }
    name = pair["name"]
    start_equity = float(config.get("start_equity", 1000))

    try:
        # Use a copy of config to be absolutely safe in parallel
        local_config = dict(config)
        
        with suppress_stdout():
            coint_calibrate_pair(pair, local_config)
            ou_calibrate_pair(pair, local_config)
            calculate_bands(pair, local_config)
            df = backtest_pair(pair, local_config)
        
        if df is None or df.empty:
            return None

        final_equity = float(df["equity"].iloc[-1])
        ret_pct = (final_equity / start_equity - 1) * 100
        trades = int(df["turnover"].gt(0).sum()) if "turnover" in df.columns else 0
        
        return {
            "Pair": name,
            "Final Equity": final_equity,
            "Return %": ret_pct,
            "Trades": trades,
            "y_symbol": sym_y,
            "x_symbol": sym_x
        }
    except Exception:
        return None

def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        print("config.json not found.")
        return

    config = load_config(config_path)
    
    # Global overrides for ranking phase
    config["log_adf_each_window"] = False
    config["show_plots"] = False
    
    # Use 1000 days of history as requested
    min_rows = 1000
    
    print("--- Scanning Data ---")
    symbols = get_available_symbols(config, min_rows)
    print(f"Found {len(symbols)} symbols with >= {min_rows} rows.")
    
    if len(symbols) < 2:
        print("Need at least 2 symbols to form a pair.")
        return

    # Generate unique combinations
    pairs_list = list(itertools.combinations(symbols, 2))
    print(f"Generated {len(pairs_list)} unique pairs to test.")
    
    results = []
    max_workers = int(config.get("ranking_threads", 4))
    
    print(f"\n--- Starting Bulk Backtest (Parallel, Workers={max_workers}) ---")
    
    best_equity_so_far = -float('inf')
    best_pair_so_far = None

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [(sym_y, sym_x, config) for sym_y, sym_x in pairs_list]
        
        count = 0
        total = len(tasks)
        for res in executor.map(process_single_pair, tasks):
            count += 1
            if res:
                results.append(res)
                if res["Final Equity"] > best_equity_so_far:
                    best_equity_so_far = res["Final Equity"]
                    best_pair_so_far = res["Pair"]
            
            if count % 10 == 0 or count == total:
                best_info = f" | Best so far: {best_pair_so_far} ({best_equity_so_far:.2f} USD)" if best_pair_so_far else ""
                print(f"Progress: {count}/{total} (Valid: {len(results)}){best_info}")

    # Final Ranking
    print("\n--- Final Ranking (Equity) ---")
    if not results:
        print("No valid results generated.")
        return

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("Final Equity", ascending=False).reset_index(drop=True)
    
    print(df_res.head(50).to_string(index=False))
    
    _, _, output_dir = get_dirs(config)
    rank_path = os.path.join(output_dir, "equity_ranking_all.csv")
    df_res.to_csv(rank_path, index=False)
    print(f"\nSaved full ranking to {rank_path}")
    
    # Verification & Plotting for Best Pair
    best_row = df_res.iloc[0]
    best_pair_name = best_row["Pair"]
    
    print(f"\n--- Verifying Best Pair: {best_pair_name} ---")
    best_pair = {
        "y_symbol": best_row["y_symbol"],
        "x_symbol": best_row["x_symbol"],
        "name": best_pair_name
    }
    
    # Re-run in main process to confirm numbers and generate plot
    df_best = backtest_pair(best_pair, config)
    final_val = df_best["equity"].iloc[-1]
    print(f"Main Process Verification Equity: {final_val:.2f} USD")
    
    data_dir = config.get("data_dir", "data")
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 300))
    start_date = df_best.index[0].strftime('%Y%m%d')
    end_date = df_best.index[-1].strftime('%Y%m%d')
    
    filename = f"BEST_EQUITY_{best_pair_name}_{interval}_w{window}_{start_date}-{end_date}.png"
    filename = filename.replace(os.path.sep, "_").replace(":", "")
    save_path = os.path.join(plots_dir, filename)
    
    try:
        plot_equity(df_best, best_pair_name, save_path, show_plot=False)
        print(f"Best equity plot saved to {save_path}")
    except Exception as e:
        print(f"Error plotting best pair: {e}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()