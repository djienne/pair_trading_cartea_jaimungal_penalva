import json
import os
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

def load_config(path: str = "config.json") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pair_id(pair: Dict) -> str:
    return f"{pair['y_symbol']}__{pair['x_symbol']}"

def get_dirs(config: Dict) -> Tuple[str, str, str]:
    """
    Returns tuple of (feather_dir, intermediate_dir, output_dir).
    Creates them if they don't exist.
    """
    data_dir = config.get("data_dir", "data")
    feather_dir = config.get("feather_dir") or os.path.join(data_dir, "feather")
    intermediate_dir = config.get("intermediate_dir") or os.path.join(data_dir, "intermediate")
    output_dir = config.get("output_dir") or os.path.join(data_dir, "output")
    
    os.makedirs(feather_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return feather_dir, intermediate_dir, output_dir

def get_symbol_path(symbol: str, interval: str, feather_dir: str) -> str:
    return os.path.join(feather_dir, f"{symbol}_{interval}.feather")

def load_symbol_data(symbol: str, interval: str, feather_dir: str) -> pd.DataFrame:
    path = get_symbol_path(symbol, interval, feather_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data for {symbol}: {path}")
    df = pd.read_feather(path)
    if "open_time_dt" in df.columns:
        df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    else:
        df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def load_coint_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cointegration data: {path}")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    return df.sort_values("open_time_dt").set_index("open_time_dt")

def load_ou_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing OU calibration data: {path}")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    return df.sort_values("open_time_dt").set_index("open_time_dt")

def load_bands_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing band data: {path}")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    return df.sort_values("open_time_dt").set_index("open_time_dt")

def valid_ou_params(params: np.ndarray, min_sigma: float, min_kappa: float) -> bool:
    if params is None or len(params) < 3:
        return False
    theta, mu, sigma = params
    if not np.isfinite(theta) or not np.isfinite(mu) or not np.isfinite(sigma):
        return False
    if theta <= min_kappa or sigma <= min_sigma:
        return False
    return True
