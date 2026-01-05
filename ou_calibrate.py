from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as spopt
import statsmodels.api as sm

from utils import load_config, pair_id, load_pair_data, save_pair_data, valid_ou_params


def method_moments(x: np.ndarray, dt: float) -> Tuple[float, float, float]:
    mu = float(np.mean(x))
    y = (x[1:] - x[:-1]) / dt
    exog = mu - x[:-1]
    res = sm.OLS(y, exog).fit()
    theta = float(res.params[0])
    sigma = float(np.std(res.resid) / np.sqrt(dt))
    return theta, mu, sigma


def neg_log_likelihood(params: np.ndarray, x: np.ndarray, dt: float) -> float:
    theta, mu, sigma = params
    if sigma <= 0 or theta < 0:
        return 1e10
    m = x[:-1] + theta * (mu - x[:-1]) * dt
    s = sigma * np.sqrt(dt)
    resid = x[1:] - m
    n = len(resid)
    ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(s) - 0.5 * np.sum((resid / s) ** 2)
    return -ll


def ou_mle(
    x: np.ndarray, dt: float, min_sigma: float, min_kappa: float
) -> Tuple[np.ndarray, bool]:
    mu_init = float(np.mean(x))
    sigma_init = float(np.std(x)) if np.std(x) > 0 else 1.0
    theta_init = 1.0
    x0 = np.array([theta_init, mu_init, sigma_init])
    res = spopt.minimize(neg_log_likelihood, x0, args=(x, dt), method="Nelder-Mead")
    params = res.x
    ok = bool(res.success) and valid_ou_params(params, min_sigma, min_kappa)
    return params, ok


def calibrate_pair(pair: Dict, config: Dict) -> pd.DataFrame:
    window = int(config.get("rolling_window_days", 30))
    dt = float(config.get("ou_dt_days", 1.0))
    min_sigma = float(config.get("ou_min_u", 1e-6))
    min_sigma = max(min_sigma, 1e-12)
    min_kappa = float(config.get("ou_min_kappa", 1e-4))
    min_kappa = max(min_kappa, 1e-12)

    # Load cointegration data using consolidated utility
    df = load_pair_data(pair, config, "coint")

    n = len(df)
    kappa_arr = np.full(n, np.nan)
    mu_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
    method_arr = np.full(n, "", dtype=object)
    valid_arr = np.full(n, False, dtype=bool)

    for i in range(window, n):
        row = df.iloc[i]
        alpha = row["alpha"]
        beta = row["beta"]
        if not np.isfinite(alpha) or not np.isfinite(beta):
            continue

        window_slice = df.iloc[i - window : i]
        epsilon_window = window_slice["y_close"] - (
            alpha + beta * window_slice["x_close"]
        )
        if len(epsilon_window) < 2:
            continue

        params, ok = ou_mle(epsilon_window.values, dt, min_sigma, min_kappa)
        method = "mle"
        if not ok:
            params = np.array(method_moments(epsilon_window.values, dt))
            method = "mom"

        if valid_ou_params(params, min_sigma, min_kappa):
            kappa_arr[i] = params[0]
            mu_arr[i] = params[1]
            sigma_arr[i] = params[2]
            method_arr[i] = method
            valid_arr[i] = True
        else:
            method_arr[i] = "invalid"

    out = pd.DataFrame(
        {
            "open_time_dt": df.index,
            "kappa": kappa_arr,
            "mu": mu_arr,
            "sigma": sigma_arr,
            "ou_method": method_arr,
            "ou_valid": valid_arr,
        }
    )

    evaluated_mask = out["ou_method"] != ""
    n_eval = int(evaluated_mask.sum())
    n_valid = int(out["ou_valid"].sum())
    mle_used = int((out["ou_method"] == "mle").sum())
    mom_used = int((out["ou_method"] == "mom").sum())
    invalid = int((out["ou_method"] == "invalid").sum())

    valid_params = out.loc[out["ou_valid"], ["kappa", "mu", "sigma"]]
    if not valid_params.empty:
        med_kappa = float(valid_params["kappa"].median())
        med_mu = float(valid_params["mu"].median())
        med_sigma = float(valid_params["sigma"].median())
    else:
        med_kappa = float("nan")
        med_mu = float("nan")
        med_sigma = float("nan")

    interval = config.get("candle_interval", "1d")
    print("OU calibration summary")
    print(f"Pair: {pair_id(pair)} | Interval: {interval} | Window: {window}")
    print(
        f"Evaluated windows: {n_eval} | Valid: {n_valid} | "
        f"MLE: {mle_used} | MOM: {mom_used} | Invalid: {invalid}"
    )
    if np.isfinite(med_kappa) and np.isfinite(med_mu) and np.isfinite(med_sigma):
        print(
            f"Median params -> kappa: {med_kappa:.6f} | mu: {med_mu:.6f} | "
            f"sigma: {med_sigma:.6f}"
        )

    # Save using consolidated utility
    save_pair_data(out, pair, config, "ou")
    return out


def main(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    pairs = config.get("pairs", [])
    if not pairs:
        raise ValueError("No pairs configured in config.json")

    for pair in pairs:
        pair_name = pair.get("name") or pair_id(pair)
        print(f"Running OU calibration for {pair_name}...")
        calibrate_pair(pair, config)


if __name__ == "__main__":
    main()