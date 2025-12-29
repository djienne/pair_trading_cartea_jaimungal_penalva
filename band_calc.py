import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import root

from utils import load_config, pair_id, get_dirs, load_coint_data, load_ou_data, valid_ou_params


class CointOpti:
    def __init__(
        self,
        c: float,
        theta: float,
        kappa: float,
        sigma: float,
        rho: float,
        min_u: float = 1e-6,
    ):
        self.c = c
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho = rho
        self.min_u = max(float(min_u), 1e-12)

    def _safe_integrand(self, u: float) -> float:
        exponent = self.rho / self.kappa - 1.0
        log_val = exponent * np.log(u) - (self.kappa * (u - self.theta) ** 2) / (
            self.sigma**2
        )
        if log_val > 700:
            return float(np.exp(700))
        if log_val < -745:
            return 0.0
        return float(np.exp(log_val))

    def int_plus(self, u, epsilon):
        u_safe = max(float(u), self.min_u)
        return self._safe_integrand(u_safe)

    def int_minus(self, u, epsilon):
        u_safe = max(float(u), self.min_u)
        return self._safe_integrand(u_safe)

    def dint_plus(self, epsilon):
        e_val = np.asarray(epsilon).item() if isinstance(epsilon, np.ndarray) else epsilon
        e_safe = max(float(e_val), self.min_u)
        return -1 * self._safe_integrand(e_safe)

    def dint_minus(self, epsilon):
        e_val = np.asarray(epsilon).item() if isinstance(epsilon, np.ndarray) else epsilon
        e_safe = max(float(e_val), self.min_u)
        return 1 * self._safe_integrand(e_safe)

    def F_plus(self, epsilon):
        e_val = np.asarray(epsilon).item() if isinstance(epsilon, np.ndarray) else epsilon
        e_safe = max(float(e_val), self.min_u)
        res, _ = quad(self.int_plus, e_safe, np.inf, args=(e_safe,))
        if not np.isfinite(res):
            raise ValueError("F_plus not finite")
        return res

    def F_minus(self, epsilon):
        e_val = np.asarray(epsilon).item() if isinstance(epsilon, np.ndarray) else epsilon
        e_safe = max(float(e_val), self.min_u)
        if e_safe <= self.min_u:
            return 0.0
        res, _ = quad(self.int_minus, self.min_u, e_safe, args=(e_safe,))
        if not np.isfinite(res):
            raise ValueError("F_minus not finite")
        return res

    def dF_plus(self, epsilon, analytic=True):
        if analytic:
            return self.dint_plus(epsilon)
        h = 1e-4
        e_val = np.asarray(epsilon).item() if isinstance(epsilon, np.ndarray) else epsilon
        return (self.F_plus(e_val + h) - self.F_plus(e_val - h)) / (2 * h)

    def dF_minus(self, epsilon, analytic=True):
        if analytic:
            return self.dint_minus(epsilon)
        h = 1e-4
        e_val = np.asarray(epsilon).item() if isinstance(epsilon, np.ndarray) else epsilon
        return (self.F_minus(e_val + h) - self.F_minus(e_val - h)) / (2 * h)

    def H_plus(self, epsilon, analytic=True):
        e_val = np.asarray(epsilon).item() if isinstance(epsilon, np.ndarray) else epsilon
        e_safe = max(float(e_val), self.min_u)
        return self.F_plus(e_safe) - (e_safe - self.c) * self.dF_plus(e_safe, analytic)

    def solve_optimal_long_short(self, epsilon_star, analytic=True):
        solution = root(self.H_plus, epsilon_star, args=(analytic,))
        if not solution.success or not np.isfinite(solution.x[0]):
            raise ValueError("Root solve failed for long-short")
        return solution.x[0]

    def get_opti_params(self, analytic=True, init_low=0.5, init_high=1.5):
        epsilon_star = self.solve_optimal_long_short(init_low, analytic)
        return epsilon_star, -epsilon_star + 2 * self.theta


def compute_bands(
    epsilon: np.ndarray,
    ou_params: np.ndarray,
    config: Dict,
) -> Tuple[float, float, float, bool]:
    min_u = float(config.get("ou_min_u", 1e-6))
    min_u = max(min_u, 1e-12)
    min_kappa = float(config.get("ou_min_kappa", 1e-4))
    min_kappa = max(min_kappa, 1e-12)

    std_mult = float(config.get("bands_std_mult", 3.0))
    if not valid_ou_params(ou_params, min_u, min_kappa) or np.std(epsilon) <= min_u:
        mean = float(np.mean(epsilon))
        std = float(np.std(epsilon))
        return mean - std_mult * std, mean + std_mult * std, mean, True

    kappa, mu, sigma = ou_params
    shift = max(0.0, -float(np.min(epsilon)) + min_u)
    mu_shift = mu + shift

    trans_cost = float(config.get("transaction_cost", 0.0))
    if trans_cost <= 0:
        trans_cost = 0.001

    coint = CointOpti(
        c=mu_shift - trans_cost,
        theta=mu_shift,
        kappa=float(kappa),
        sigma=float(sigma),
        rho=float(config.get("discount_rate", 0.05)),
        min_u=min_u,
    )

    try:
        init_low = max(min_u, mu_shift - sigma)
        init_high = mu_shift + sigma
        lower_shift, _ = coint.get_opti_params(
            analytic=True,
            init_low=init_low,
            init_high=init_high,
        )
        if not np.isfinite(lower_shift) or lower_shift <= min_u:
            raise ValueError("Invalid long-short bands")
        lower = float(lower_shift - shift)
        upper = float(-lower + 2 * mu)
        if not np.isfinite(lower) or not np.isfinite(upper) or not (lower < mu < upper):
            raise ValueError("Invalid long-short bands")
        return lower, upper, float(mu), False
    except Exception:
        mean = float(np.mean(epsilon))
        std = float(np.std(epsilon))
        return mean - std_mult * std, mean + std_mult * std, mean, True


def calculate_bands(pair: Dict, config: Dict) -> pd.DataFrame:
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 30))
    _, intermediate_dir, _ = get_dirs(config)

    coint_path = os.path.join(
        intermediate_dir,
        f"coint_{pair_id(pair)}_{interval}_w{window}.feather",
    )
    ou_path = os.path.join(
        intermediate_dir,
        f"ou_{pair_id(pair)}_{interval}_w{window}.feather",
    )

    coint_df = load_coint_data(coint_path)
    ou_df = load_ou_data(ou_path)

    merged = coint_df.join(ou_df, how="left", rsuffix="_ou")
    n = len(merged)

    lower_arr = np.full(n, np.nan)
    upper_arr = np.full(n, np.nan)
    mu_arr = np.full(n, np.nan)
    fallback_arr = np.full(n, False, dtype=bool)

    for i in range(window, n):
        row = merged.iloc[i]
        alpha = row["alpha"]
        beta = row["beta"]
        if not np.isfinite(alpha) or not np.isfinite(beta):
            continue

        window_slice = merged.iloc[i - window : i]
        epsilon_window = window_slice["y_close"] - (
            alpha + beta * window_slice["x_close"]
        )

        ou_params = np.array([row["kappa"], row["mu"], row["sigma"]])
        lower, upper, mu, used_fallback = compute_bands(
            epsilon_window.values, ou_params, config
        )

        lower_arr[i] = lower
        upper_arr[i] = upper
        mu_arr[i] = mu
        fallback_arr[i] = used_fallback

    out = pd.DataFrame(
        {
            "open_time_dt": merged.index,
            "lower": lower_arr,
            "upper": upper_arr,
            "mu": mu_arr,
            "bands_fallback": fallback_arr,
        }
    )

    valid_mask = (
        np.isfinite(out["lower"])
        & np.isfinite(out["upper"])
        & np.isfinite(out["mu"])
    )
    n_valid = int(valid_mask.sum())
    n_total = int(len(out))
    fallback_count = int((out["bands_fallback"] & valid_mask).sum())
    fallback_rate = (fallback_count / n_valid) if n_valid > 0 else 0.0

    if n_valid > 0:
        med_lower = float(out.loc[valid_mask, "lower"].median())
        med_upper = float(out.loc[valid_mask, "upper"].median())
        med_mu = float(out.loc[valid_mask, "mu"].median())
    else:
        med_lower = float("nan")
        med_upper = float("nan")
        med_mu = float("nan")

    print("Band calculation summary")
    print(f"Pair: {pair_id(pair)} | Interval: {interval} | Window: {window}")
    print(
        f"Total rows: {n_total} | Valid bands: {n_valid} | "
        f"Fallbacks: {fallback_count} ({fallback_rate:.2%})"
    )
    if np.isfinite(med_lower) and np.isfinite(med_upper) and np.isfinite(med_mu):
        print(
            f"Median bands -> lower: {med_lower:.6f} | mu: {med_mu:.6f} | "
            f"upper: {med_upper:.6f}"
        )
    if n_valid > 0 and fallback_count == n_valid:
        print("Warning: all valid bands fell back to mean/std.")

    out_path = os.path.join(
        intermediate_dir,
        f"bands_{pair_id(pair)}_{interval}_w{window}.feather",
    )
    out.to_feather(out_path)
    print(f"Saved bands to {out_path}")
    return out


def main(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    pairs = config.get("pairs", [])
    if not pairs:
        raise ValueError("No pairs configured in config.json")

    for pair in pairs:
        pair_name = pair.get("name") or pair_id(pair)
        print(f"Running band calculation for {pair_name}...")
        calculate_bands(pair, config)


if __name__ == "__main__":
    main()