import json
import os
import subprocess
import sys
from typing import Dict, List, Set

from utils import load_config

def get_required_symbols(config: Dict) -> Set[str]:
    symbols = set()
    for pair in config.get("pairs", []):
        symbols.add(pair["y_symbol"])
        symbols.add(pair["x_symbol"])
    return symbols


def check_missing_data(config: Dict) -> List[str]:
    data_dir = config.get("data_dir", "data")
    feather_dir = config.get("feather_dir") or os.path.join(data_dir, "feather")
    interval = config.get("candle_interval", "1d")
    
    required_symbols = get_required_symbols(config)
    missing = []
    
    for symbol in required_symbols:
        path = os.path.join(feather_dir, f"{symbol}_{interval}.feather")
        if not os.path.exists(path):
            missing.append(symbol)
            
    return missing


def run_script(script_name: str, args: List[str] = None) -> None:
    print(f"--- Running {script_name} ---")
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
        
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"--- Finished {script_name} ---")


def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    config = load_config(config_path)
    
    # 1. Check for missing data and run download_data.py if needed
    missing_symbols = check_missing_data(config)
    if missing_symbols:
        print(f"Missing data for symbols: {', '.join(missing_symbols)}")
        run_script("download_data.py", ["--symbols"] + missing_symbols)
    else:
        print("All required symbol data found. Skipping download.")
        
    # 2. Run coint_calibrate.py
    run_script("coint_calibrate.py")
    
    # 3. Run ou_calibrate.py
    run_script("ou_calibrate.py")
    
    # 4. Run band_calc.py
    run_script("band_calc.py")
    
    # 5. Run backtest.py
    run_script("backtest.py")
    
    print("All steps completed successfully.")


if __name__ == "__main__":
    main()