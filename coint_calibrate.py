import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from utils import load_config, pair_id, get_dirs, load_symbol_data


def prepare_pair_data(
    y_symbol: str, x_symbol: str, interval: str, feather_dir: str
) -> pd.DataFrame:
    df_y = load_symbol_data(y_symbol, interval, feather_dir)
    df_x = load_symbol_data(x_symbol, interval, feather_dir)

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


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> Tuple[float, float, float]:
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    beta = model.params[x.name]
    return float(alpha), float(beta), float(model.rsquared)


def compute_epsilon(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    return y - (alpha + beta * x)


def adf_check(series: pd.Series, alpha: float) -> Tuple[float, bool]:
    try:
        adf_result = adfuller(series, autolag="AIC")
        p_value = adf_result[1]
        return float(p_value), p_value < alpha
    except Exception:
        return 1.0, False


def calibrate_pair(pair: Dict, config: Dict) -> pd.DataFrame:
    interval = config.get("candle_interval", "1d")
    feather_dir, intermediate_dir, _ = get_dirs(config)

    data = prepare_pair_data(pair["y_symbol"], pair["x_symbol"], interval, feather_dir)
    if data.empty:
        raise ValueError("No overlapping data found for the pair.")

    window = int(config.get("rolling_window_days", 30))
    if len(data) <= window:
        raise ValueError("Not enough data for the rolling window size.")

    adf_alpha = float(config.get("adf_alpha", 0.05))
    log_adf = bool(config.get("log_adf_each_window", True))
    log_every = int(config.get("log_every_n", 1))

    idx = data.index
    n = len(data)

    alpha_arr = np.full(n, np.nan)
    beta_arr = np.full(n, np.nan)
    epsilon_arr = np.full(n, np.nan)
    r2_arr = np.full(n, np.nan)
    adf_p_arr = np.full(n, np.nan)
    adf_pass_arr = np.full(n, False, dtype=bool)

    for i in range(window, n):
        window_slice = data.iloc[i - window : i]
        y_window = window_slice["y_close"]
        x_window = window_slice["x_close"]

        alpha, beta, r2 = ols_hedge_ratio(y_window, x_window)
        epsilon_window = compute_epsilon(y_window, x_window, alpha, beta)

        p_value, passed = adf_check(epsilon_window, adf_alpha)
        if log_adf and (i % log_every == 0):
            status = "PASS" if passed else "FAIL"
            print(f"{idx[i].date()} ADF p={p_value:.4f} -> {status}")

        epsilon_t = float(
            data.iloc[i]["y_close"] - (alpha + beta * data.iloc[i]["x_close"])
        )

        alpha_arr[i] = alpha
        beta_arr[i] = beta
        epsilon_arr[i] = epsilon_t
        r2_arr[i] = r2
        adf_p_arr[i] = p_value
        adf_pass_arr[i] = passed

    results = data.copy()
    results["alpha"] = alpha_arr
    results["beta"] = beta_arr
    results["epsilon"] = epsilon_arr
    results["r2"] = r2_arr
    results["adf_pvalue"] = adf_p_arr
    results["adf_pass"] = adf_pass_arr

    out_path = os.path.join(
        intermediate_dir,
        f"coint_{pair_id(pair)}_{interval}_w{window}.feather",
    )
    results.reset_index().to_feather(out_path)
    print(f"Saved cointegration results to {out_path}")
    return results


def main(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    pairs = config.get("pairs", [])
    if not pairs:
        raise ValueError("No pairs configured in config.json")

    for pair in pairs:
        pair_name = pair.get("name") or pair_id(pair)
        print(f"Running cointegration calibration for {pair_name}...")
        calibrate_pair(pair, config)


if __name__ == "__main__":
    main()