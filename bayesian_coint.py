import json
import os
import sys
import hashlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    load_config,
    pair_id,
    get_dirs,
    prepare_pair_data,
    get_symbol_path,
    get_pair_data_path,
    TIME_COLUMN,
)

# Import the Bayesian logic from the file we just created
try:
    from rolling_bayesian_hedge import rolling_bayesian_rw_hedge_ratio
except ImportError:
    # If strictly running as a script without . in path
    sys.path.append(os.path.dirname(__file__))
    from rolling_bayesian_hedge import rolling_bayesian_rw_hedge_ratio


def get_file_info(path: str) -> str:
    """Returns a string combining file size and modification time for hashing."""
    if not os.path.exists(path):
        return "MISSING"
    stats = os.stat(path)
    return f"{stats.st_size}_{stats.st_mtime}"

def compute_cache_key(pair: Dict, config: Dict, y_path: str, x_path: str) -> str:
    """
    Generates a unique hash based on:
    - Config parameters affecting the model
    - Input data file states (mtime + size)
    """
    bayes_cfg = config.get("bayesian_config", {})
    relevant_config = {
        "window": int(config.get("rolling_window_days", 30)),
        "inference_method": bayes_cfg.get("inference_method", "advi"),
        "advi_steps": int(bayes_cfg.get("advi_steps", 2000)),
        "draws": int(bayes_cfg.get("draws", 500)),
        "tune": int(bayes_cfg.get("tune", 500)),
        "candle_interval": config.get("candle_interval", "1d"),
        # We don't include ranking_threads because it affects speed, not the result
    }
    
    data_state = {
        "y_file": get_file_info(y_path),
        "x_file": get_file_info(x_path)
    }
    
    # Serialize and hash
    combined = json.dumps({**relevant_config, **data_state}, sort_keys=True)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def calibrate_pair(pair: Dict, config: Dict, n_jobs_override: Optional[int] = None, show_progress: bool = True) -> pd.DataFrame:
    interval = config.get("candle_interval", "1d")
    feather_dir, _, _ = get_dirs(config)

    # Identify input file paths for caching check
    y_path = get_symbol_path(pair["y_symbol"], interval, feather_dir)
    x_path = get_symbol_path(pair["x_symbol"], interval, feather_dir)

    # Calculate Cache Key
    current_cache_key = compute_cache_key(pair, config, y_path, x_path)

    # Use centralized path builder
    out_path = get_pair_data_path(pair, config, "coint")
    metadata_path = out_path + ".metadata.json"

    # --- CACHE CHECK ---
    if os.path.exists(out_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                stored_metadata = json.load(f)

            if stored_metadata.get("cache_key") == current_cache_key:
                if show_progress:
                    print(f"Skipping {pair_id(pair)}: Cache hit (configuration and data unchanged).")
                # Load cached result
                df = pd.read_feather(out_path)
                if TIME_COLUMN in df.columns:
                    df = df.sort_values(TIME_COLUMN).set_index(TIME_COLUMN)
                return df
        except Exception as e:
            if show_progress:
                print(f"Cache check failed for {pair_id(pair)}, re-running. Error: {e}")

    # --- RUN CALCULATION ---

    # 1. Load Data
    data = prepare_pair_data(pair["y_symbol"], pair["x_symbol"], interval, feather_dir)
    if data.empty:
        raise ValueError(f"No overlapping data for {pair_id(pair)}.")

    # Configuration for Bayesian Rolling Window
    window = int(config.get("rolling_window_days", 30))

    # Bayesian settings from nested config block
    bayes_cfg = config.get("bayesian_config", {})
    inference_method = bayes_cfg.get("inference_method", "advi")
    advi_steps = int(bayes_cfg.get("advi_steps", 2000))
    draws = int(bayes_cfg.get("draws", 500))
    tune = int(bayes_cfg.get("tune", 500))
    target_accept = float(bayes_cfg.get("target_accept", 0.9))
    use_ols_init = bool(bayes_cfg.get("use_ols_init", True))
    
    # Determine n_jobs: use override if present, else default 10
    if n_jobs_override is not None:
        n_jobs = int(n_jobs_override)
    else:
        n_jobs = 10

    if len(data) <= window + 2:
        raise ValueError(f"Insufficient data for {pair_id(pair)} (len={len(data)} <= window={window}).")

    if show_progress:
        print(f"  [Bayesian] Calibrating {pair_id(pair)} | Window={window} | CPU Threads={n_jobs} (Parallel MCMC/ADVI)")
    
    # 2. Run Rolling Bayesian Hedge
    try:
        hed = rolling_bayesian_rw_hedge_ratio(
            s1=data["y_close"],
            s2=data["x_close"],
            window=window,
            inference=inference_method,
            advi_steps=advi_steps,
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            update_every=1,
            warm_start=True,
            random_seed=42,
            use_ols_init=use_ols_init,
            n_jobs=n_jobs,
            show_progress=show_progress
        )
    except Exception as e:
        if show_progress:
            print(f"Error during Bayesian estimation for {pair_id(pair)}: {e}")
        raise
    
    # 3. Merge results back into the dataframe
    results = data.copy()
    # Align indices carefully
    results["alpha"] = hed.alpha_hat
    results["beta"] = hed.beta_hat
    results["sigma_obs"] = hed.sigma_obs_hat
    
    # Fill missing values before calculation if needed
    results = results.dropna(subset=["alpha", "beta"])
    
    results["epsilon"] = results["y_close"] - (results["alpha"] + results["beta"] * results["x_close"])
    
    # Dummy columns for compatibility
    results["r2"] = np.nan 
    results["adf_pvalue"] = np.nan 
    results["adf_pass"] = False

    # 4. Save Result
    results.reset_index().to_feather(out_path)
    
    # 5. Save Metadata
    try:
        with open(metadata_path, "w") as f:
            json.dump({"cache_key": current_cache_key}, f)
    except Exception as e:
        if show_progress:
            print(f"Warning: Failed to save metadata for {pair_id(pair)}: {e}")

    if show_progress:
        print(f"Saved Bayesian cointegration results to {out_path}")
        
    return results


def main(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    pairs = config.get("pairs", [])
    if not pairs:
        print("No pairs configured in config.json")
        sys.exit(1)

    print(f"Found {len(pairs)} pairs to process.")
    
    # Use tqdm for the list of pairs
    for pair in tqdm(pairs, desc="Processing Pairs"):
        try:
            calibrate_pair(pair, config)
        except Exception as e:
            print(f"Error processing {pair_id(pair)}: {e}")



if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
