import json
import os
from typing import Dict, Tuple, Optional, Literal
import pandas as pd
import numpy as np

# =============================================================================
# Constants - Centralized file prefixes and column names
# =============================================================================
FILE_PREFIX_COINT = "coint"
FILE_PREFIX_OU = "ou"
FILE_PREFIX_BANDS = "bands"
FILE_PREFIX_BACKTEST = "backtest"
FILE_EXTENSION = ".feather"
TIME_COLUMN = "open_time_dt"

DataType = Literal["coint", "ou", "bands", "backtest"]


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


# =============================================================================
# Centralized File Path Builders
# =============================================================================
def get_symbol_path(symbol: str, interval: str, feather_dir: str) -> str:
    return os.path.join(feather_dir, f"{symbol}_{interval}{FILE_EXTENSION}")


def get_pair_data_path(
    pair: Dict,
    config: Dict,
    data_type: DataType,
    directory: Optional[str] = None,
) -> str:
    """
    Centralized path builder for all pair-related data files.

    Args:
        pair: Dictionary with y_symbol and x_symbol keys
        config: Configuration dictionary
        data_type: One of "coint", "ou", "bands", "backtest"
        directory: Override directory (uses intermediate_dir or output_dir by default)

    Returns:
        Full path to the data file
    """
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 30))
    pid = pair_id(pair)

    prefix_map = {
        "coint": FILE_PREFIX_COINT,
        "ou": FILE_PREFIX_OU,
        "bands": FILE_PREFIX_BANDS,
        "backtest": FILE_PREFIX_BACKTEST,
    }
    prefix = prefix_map[data_type]

    if directory is None:
        _, intermediate_dir, output_dir = get_dirs(config)
        directory = output_dir if data_type == "backtest" else intermediate_dir

    filename = f"{prefix}_{pid}_{interval}_w{window}{FILE_EXTENSION}"
    return os.path.join(directory, filename)


# =============================================================================
# Consolidated Data Loading
# =============================================================================
def _load_feather_with_time_index(path: str, data_type: str) -> pd.DataFrame:
    """
    Generic feather loader that handles time column parsing and indexing.
    Consolidates load_coint_data, load_ou_data, load_bands_data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {data_type} data: {path}")
    df = pd.read_feather(path)
    if TIME_COLUMN in df.columns:
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
        df = df.sort_values(TIME_COLUMN).set_index(TIME_COLUMN)
    return df


def load_pair_data(
    pair: Dict,
    config: Dict,
    data_type: DataType,
    directory: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load any pair-related data file by type.

    Args:
        pair: Dictionary with y_symbol and x_symbol keys
        config: Configuration dictionary
        data_type: One of "coint", "ou", "bands", "backtest"
        directory: Override directory

    Returns:
        DataFrame with time index
    """
    path = get_pair_data_path(pair, config, data_type, directory)
    return _load_feather_with_time_index(path, data_type)


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

# Legacy wrappers for backward compatibility - use load_pair_data() for new code
def load_coint_data(path: str) -> pd.DataFrame:
    """Deprecated: Use load_pair_data(pair, config, 'coint') instead."""
    return _load_feather_with_time_index(path, "cointegration")


def load_ou_data(path: str) -> pd.DataFrame:
    """Deprecated: Use load_pair_data(pair, config, 'ou') instead."""
    return _load_feather_with_time_index(path, "OU calibration")


def load_bands_data(path: str) -> pd.DataFrame:
    """Deprecated: Use load_pair_data(pair, config, 'bands') instead."""
    return _load_feather_with_time_index(path, "band")


def save_pair_data(
    df: pd.DataFrame,
    pair: Dict,
    config: Dict,
    data_type: DataType,
    directory: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Save pair-related data to standardized feather file.

    Args:
        df: DataFrame to save (will reset index if needed)
        pair: Dictionary with y_symbol and x_symbol keys
        config: Configuration dictionary
        data_type: One of "coint", "ou", "bands", "backtest"
        directory: Override directory
        verbose: Print save confirmation

    Returns:
        Path where file was saved
    """
    path = get_pair_data_path(pair, config, data_type, directory)
    df_to_save = df.reset_index() if df.index.name == TIME_COLUMN else df
    df_to_save.to_feather(path)
    if verbose:
        print(f"Saved {data_type} data to {path}")
    return path


def valid_ou_params(params: np.ndarray, min_sigma: float, min_kappa: float) -> bool:
    if params is None or len(params) < 3:
        return False
    theta, mu, sigma = params
    if not np.isfinite(theta) or not np.isfinite(mu) or not np.isfinite(sigma):
        return False
    if theta <= min_kappa or sigma <= min_sigma:
        return False
    return True


def prepare_pair_data(
    y_symbol: str, x_symbol: str, interval: str, feather_dir: str
) -> pd.DataFrame:
    """
    Loads and merges data for two symbols on open_time_dt.
    Returns dataframe with columns [y_close, x_close].
    """
    try:
        df_y = load_symbol_data(y_symbol, interval, feather_dir)
        df_x = load_symbol_data(x_symbol, interval, feather_dir)
    except FileNotFoundError:
        return pd.DataFrame()

    merged = pd.merge(
        df_y[["open_time_dt", "close"]],
        df_x[["open_time_dt", "close"]],
        on="open_time_dt",
        how="inner",
        suffixes=("_y", "_x"),
    )
    merged = merged.sort_values("open_time_dt").set_index("open_time_dt")
    merged.rename(columns={"close_y": "y_close", "close_x": "x_close"}, inplace=True)
    merged = merged.dropna()
    return merged
