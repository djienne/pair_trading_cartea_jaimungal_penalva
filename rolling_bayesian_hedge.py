from __future__ import annotations

import math
import multiprocessing as mp
import warnings
import sys
import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, List
from functools import partial

import numpy as np
import pandas as pd

# Suppress PyTensor and PyMC noise
warnings.filterwarnings("ignore", module="pytensor.link.c.cmodule")

# Configure logging to be quiet
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)


# ============================================================
# 0) OU calibration
# ============================================================

@dataclass
class OUParams:
    kappa: float
    theta: float
    sigma: float
    dt: float

def fit_ou_discrete(eps: pd.Series, dt: float = 1.0) -> OUParams:
    """
    Fit OU via AR(1):
        eps_{t+1} = a + b*eps_t + e_t
    b = exp(-kappa*dt), a = theta*(1-b)
    """
    x = eps.values[:-1].astype(float)
    y = eps.values[1:].astype(float)
    X = np.column_stack([np.ones_like(x), x])
    a_hat, b_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    b_hat = float(np.clip(b_hat, 1e-8, 1 - 1e-8))
    kappa = -math.log(b_hat) / dt
    theta = float(a_hat) / (1.0 - b_hat)
    resid = y - (a_hat + b_hat * x)
    s2 = float(np.var(resid, ddof=2))
    sigma = math.sqrt(s2 * (2.0 * kappa) / (1.0 - math.exp(-2.0 * kappa * dt)))
    return OUParams(kappa=kappa, theta=theta, sigma=sigma, dt=dt)


# ============================================================
# 1) Optional: "optimal trigger" computation
# ============================================================

def _try_import_scipy():
    try:
        from scipy.integrate import quad
        from scipy.optimize import brentq
        return quad, brentq
    except Exception:
        return None, None

def _mpmath_fallback():
    try:
        import mpmath as mp
        return mp
    except Exception:
        return None

def _F_integrals(eps: float, kappa: float, theta: float, sigma: float, rho: float, sign: int) -> Tuple[float, float]:
    quad, _ = _try_import_scipy()
    if quad is not None:
        q = math.sqrt(2.0 * kappa / (sigma * sigma))
        p = rho / kappa
        lin = (-sign) * q * (theta - eps)

        def integrand_F(u: float) -> float:
            return (u ** (p - 1.0)) * math.exp(lin * u - 0.5 * u * u)

        def integrand_d(u: float) -> float:
            return (u ** p) * math.exp(lin * u - 0.5 * u * u)

        F, _ = quad(integrand_F, 0.0, np.inf, limit=200)
        I, _ = quad(integrand_d, 0.0, np.inf, limit=200)
        Fp = (+q * I) if sign == +1 else (-q * I)
        return float(F), float(Fp)

    mp = _mpmath_fallback()
    if mp is None:
        raise ImportError("Need scipy or mpmath for optimal-trigger integrals.")
    q = mp.sqrt(2.0 * kappa / (sigma * sigma))
    p = rho / kappa
    lin = (-sign) * q * (theta - eps)
    F = mp.quad(lambda u: (u ** (p - 1.0)) * mp.e ** (lin * u - 0.5 * u * u), [0, mp.inf])
    I = mp.quad(lambda u: (u ** p) * mp.e ** (lin * u - 0.5 * u * u), [0, mp.inf])
    Fp = (+q * I) if sign == +1 else (-q * I)
    return float(F), float(Fp)

def _find_root_bracket(func, grid: np.ndarray) -> Optional[Tuple[float, float]]:
    vals = [func(x) for x in grid]
    for i in range(len(grid) - 1):
        if np.isnan(vals[i]) or np.isnan(vals[i + 1]):
            continue
        if vals[i] * vals[i + 1] < 0:
            return float(grid[i]), float(grid[i + 1])
    return None

def compute_optimal_exit_triggers(
    ou: OUParams,
    rho: float = 0.01,
    c: float = 0.0,
    search_width: float = 8.0,
    grid_points: int = 200,
) -> Dict[str, float]:
    kappa, theta, sigma = ou.kappa, ou.theta, ou.sigma
    ou_std = sigma / math.sqrt(2.0 * kappa) if kappa > 0 else float(np.std([0, 1]))

    def f_long_exit(eps: float) -> float:
        F, Fp = _F_integrals(eps, kappa, theta, sigma, rho, sign=+1)
        return (eps - c) * Fp - F

    def f_short_exit(eps: float) -> float:
        F, Fp = _F_integrals(eps, kappa, theta, sigma, rho, sign=-1)
        return (eps + c) * Fp - F

    lo, hi = theta - search_width * ou_std, theta + search_width * ou_std
    grid = np.linspace(lo, hi, grid_points)

    _, brentq = _try_import_scipy()
    if brentq is None:
        eps_exit_long = float(grid[np.argmin([abs(f_long_exit(x)) for x in grid])])
        eps_exit_short = float(grid[np.argmin([abs(f_short_exit(x)) for x in grid])])
    else:
        br_long = _find_root_bracket(f_long_exit, grid)
        br_short = _find_root_bracket(f_short_exit, grid)
        eps_exit_long = float(brentq(f_long_exit, *br_long)) if br_long else float(theta + 2 * ou_std)
        eps_exit_short = float(brentq(f_short_exit, *br_short)) if br_short else float(theta - 2 * ou_std)

    return {
        "eps_exit_long": eps_exit_long,
        "eps_exit_short": eps_exit_short,
        "eps_entry_long": eps_exit_short,
        "eps_entry_short": eps_exit_long,
        "ou_std": float(ou_std),
    }


# ============================================================
# 2) Rolling Bayesian Random-Walk regression (daily update)
# ============================================================

def _import_pymc():
    try:
        import pymc as pm
        return pm
    except Exception:
        import pymc3 as pm
        return pm

Inference = Literal["advi", "nuts"]

@dataclass
class RollingBayesHedge:
    alpha_hat: pd.Series
    beta_hat: pd.Series
    sigma_obs_hat: pd.Series

def _fit_rw_window_pymc(
    y: np.ndarray,
    x: np.ndarray,
    inference: Inference = "advi",
    advi_steps: int = 1500,
    draws: int = 300,
    tune: int = 300,
    target_accept: float = 0.9,
    warm_start: bool = True,
    prev_alpha0: Optional[float] = None,
    prev_beta0: Optional[float] = None,
    random_seed: int = 7,
    use_ols_init: bool = True,
) -> Tuple[float, float, float]:
    pm = _import_pymc()
    T = len(y)

    initvals = None
    if use_ols_init:
        # OLS init values for alpha0/beta0 (init only; priors unchanged)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_centered = x - x_mean
        denom = float(np.sum(x_centered ** 2))
        if denom > 1e-12 and np.isfinite(denom):
            beta_ols = float(np.sum(x_centered * (y - y_mean)) / denom)
        else:
            beta_ols = 0.0
        alpha_ols = float(y_mean - beta_ols * x_mean)
        if np.isfinite(alpha_ols) and np.isfinite(beta_ols):
            initvals = {"alpha0": alpha_ols, "beta0": beta_ols}

    mu_a0 = float(prev_alpha0) if (warm_start and prev_alpha0 is not None) else 0.0
    mu_b0 = float(prev_beta0) if (warm_start and prev_beta0 is not None) else 0.0
    s0 = 1.0 

    with pm.Model() as model:
        sigma_alpha = pm.Exponential("sigma_alpha", 50.0)
        sigma_beta = pm.Exponential("sigma_beta", 50.0)

        alpha0 = pm.Normal("alpha0", mu=mu_a0, sigma=s0)
        beta0 = pm.Normal("beta0", mu=mu_b0, sigma=s0)

        eta_a = pm.Normal("eta_a", mu=0.0, sigma=sigma_alpha, shape=T-1)
        eta_b = pm.Normal("eta_b", mu=0.0, sigma=sigma_beta, shape=T-1)

        alpha_path = pm.Deterministic("alpha", pm.math.concatenate([[alpha0], alpha0 + pm.math.cumsum(eta_a)]))
        beta_path  = pm.Deterministic("beta",  pm.math.concatenate([[beta0],  beta0  + pm.math.cumsum(eta_b)]))

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)

        mu = alpha_path + beta_path * x
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y)

        if inference == "nuts":
            sample_kwargs = dict(
                draws=draws, tune=tune, target_accept=target_accept,
                chains=2, cores=1, random_seed=random_seed, progressbar=False,
            )
            if initvals is not None:
                try:
                    trace = pm.sample(**sample_kwargs, initvals=initvals)
                except TypeError:
                    trace = pm.sample(**sample_kwargs, start=initvals)
            else:
                trace = pm.sample(**sample_kwargs)
        else:
            fit_kwargs = dict(n=advi_steps, method="advi", random_seed=random_seed, progressbar=False)
            if initvals is not None:
                try:
                    approx = pm.fit(**fit_kwargs, start=initvals)
                except TypeError:
                    approx = pm.fit(**fit_kwargs)
            else:
                approx = pm.fit(**fit_kwargs)
            trace = approx.sample(draws=draws, random_seed=random_seed)

    try:
        alpha_last = float(trace.posterior["alpha"].sel(alpha_dim_0=T-1).mean(("chain", "draw")).values)
        beta_last  = float(trace.posterior["beta"].sel(beta_dim_0=T-1).mean(("chain", "draw")).values)
        sig_hat    = float(trace.posterior["sigma_obs"].mean(("chain", "draw")).values)
    except Exception:
        alpha_last = float(np.mean(trace["alpha"][:, T-1]))
        beta_last  = float(np.mean(trace["beta"][:, T-1]))
        sig_hat    = float(np.mean(trace["sigma_obs"]))

    return alpha_last, beta_last, sig_hat

def _fit_single_window_job(args) -> Tuple[int, float, float, float]:
    (i, y_win, x_win, inference, advi_steps, draws, tune, target_accept, random_seed, use_ols_init) = args
    try:
        a, b, s = _fit_rw_window_pymc(
            y_win, x_win, inference=inference, advi_steps=advi_steps,
            draws=draws, tune=tune, target_accept=target_accept, warm_start=False, random_seed=random_seed,
            use_ols_init=use_ols_init
        )
        return i, a, b, s
    except Exception:
        return i, np.nan, np.nan, np.nan

def rolling_bayesian_rw_hedge_ratio(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 252,
    inference: Inference = "advi",
    advi_steps: int = 1500,
    draws: int = 300,
    tune: int = 300,
    target_accept: float = 0.9,
    update_every: int = 1,
    warm_start: bool = True,
    random_seed: int = 7,
    use_ols_init: bool = True,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> RollingBayesHedge:
    s1 = s1.dropna()
    s2 = s2.dropna()
    idx = s1.index.intersection(s2.index)
    s1 = s1.loc[idx].astype(float)
    s2 = s2.loc[idx].astype(float)

    y, x = s1.values, s2.values
    T = len(idx)
    if T <= window + 2:
        raise ValueError("Not enough data.")

    y_mean, y_std = float(y.mean()), float(y.std(ddof=1) + 1e-12)
    x_mean, x_std = float(x.mean()), float(x.std(ddof=1) + 1e-12)
    y_s, x_s = (y - y_mean) / y_std, (x - x_mean) / x_std

    alpha_out = pd.Series(index=idx, dtype=float, name="alpha_hat")
    beta_out  = pd.Series(index=idx, dtype=float, name="beta_hat")
    sig_out   = pd.Series(index=idx, dtype=float, name="sigma_obs_hat")

    indices = [i for i in range(window, T) if (i - window) % update_every == 0]
    
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    if n_jobs > 1:
        tasks = [(i, y_s[i-window:i], x_s[i-window:i], inference, advi_steps, draws, tune, target_accept, random_seed, use_ols_init) for i in indices]
        results = []
        with mp.Pool(processes=n_jobs) as pool:
            if has_tqdm and show_progress:
                it = pool.imap_unordered(_fit_single_window_job, tasks)
                for res in tqdm(it, total=len(tasks), desc=f"Parallel Fit ({n_jobs} jobs)", leave=False):
                    results.append(res)
            else:
                results = pool.map(_fit_single_window_job, tasks)
        
        for i, a_last, b_last, sig_hat in results:
            if not np.isnan(a_last):
                beta_orig = (y_std / x_std) * b_last
                alpha_orig = y_mean + y_std * a_last - beta_orig * x_mean
                alpha_out.iloc[i], beta_out.iloc[i], sig_out.iloc[i] = alpha_orig, beta_orig, sig_hat
    else:
        prev_a, prev_b = None, None
        it = tqdm(indices, desc="Sequential Fit", leave=False) if has_tqdm and show_progress else indices
        for i in it:
            a_last, b_last, sig_hat = _fit_rw_window_pymc(
                y_s[i-window:i], x_s[i-window:i], inference=inference, advi_steps=advi_steps,
                draws=draws, tune=tune, target_accept=target_accept, warm_start=warm_start, prev_alpha0=prev_a, prev_beta0=prev_b,
                random_seed=random_seed, use_ols_init=use_ols_init
            )
            prev_a, prev_b = a_last, b_last
            beta_orig = (y_std / x_std) * b_last
            alpha_orig = y_mean + y_std * a_last - beta_orig * x_mean
            alpha_out.iloc[i], beta_out.iloc[i], sig_out.iloc[i] = alpha_orig, beta_orig, sig_hat

    return RollingBayesHedge(alpha_hat=alpha_out.ffill().dropna(), beta_hat=beta_out.ffill().dropna(), sigma_obs_hat=sig_out.ffill().dropna())


# ============================================================
# 3) Backtest
# ============================================================

@dataclass
class BacktestResult:
    ledger: pd.DataFrame
    metrics: Dict[str, float]

def backtest_dynamic_pairs_rolling(
    s1: pd.Series, s2: pd.Series, alpha_hat: pd.Series, beta_hat: pd.Series,
    signal_mode: Literal["residual", "spread"] = "residual", ou_window: int = 252,
    method: Literal["bands", "optimal"] = "bands", outer_k: float = 1.0,
    inner_k: float = 0.1, rho: float = 0.01, c: float = 0.0,
    rehedge_daily: bool = False, force_flat_end: bool = True,
) -> BacktestResult:
    s1, s2 = s1.dropna().astype(float), s2.dropna().astype(float)
    idx = s1.index.intersection(s2.index).intersection(alpha_hat.index).intersection(beta_hat.index)
    s1, s2 = s1.loc[idx], s2.loc[idx]
    a, b = alpha_hat.loc[idx], beta_hat.loc[idx]
    sig = (s1 - (a + b * s2)) if signal_mode == "residual" else (s1 - b * s2)

    pos, m1, m2, cash, rows = 0, 0.0, 0.0, 0.0, []
    for t_i in range(len(idx)):
        t, y, x, z, beta_t = idx[t_i], float(s1.iloc[t_i]), float(s2.iloc[t_i]), float(sig.iloc[t_i]), float(b.iloc[t_i])
        pos_prev, m1_prev, m2_prev = pos, m1, m2

        if rehedge_daily and pos != 0:
            target_m2 = -beta_t * m1
            cash -= (target_m2 - m2) * x + abs(target_m2 - m2) * c
            m2 = target_m2

        if t_i < ou_window:
            entry_long = entry_short = exit_long = exit_short = np.nan
        else:
            hist = sig.iloc[t_i-ou_window:t_i]
            if method == "bands":
                mu, sd = float(hist.mean()), float(hist.std(ddof=1) + 1e-12)
                entry_long, entry_short = mu - outer_k * sd, mu + outer_k * sd
                exit_long, exit_short = mu - inner_k * sd, mu + inner_k * sd
            else:
                ou = fit_ou_discrete(hist)
                trig = compute_optimal_exit_triggers(ou, rho, c)
                entry_long, entry_short, exit_long, exit_short = trig["eps_entry_long"], trig["eps_entry_short"], trig["eps_exit_long"], trig["eps_exit_short"]

        if not np.isnan(entry_long):
            if pos == 0:
                if z <= entry_long: pos, m1, m2 = 1, 1.0, -beta_t
                elif z >= entry_short: pos, m1, m2 = -1, -1.0, beta_t
                if pos != 0: cash -= (m1 * y + m2 * x) + (abs(m1) + abs(m2)) * c
            elif (pos == 1 and z >= exit_long) or (pos == -1 and z <= exit_short):
                cash += (m1 * y + m2 * x) - (abs(m1) + abs(m2)) * c
                pos, m1, m2 = 0, 0.0, 0.0

        rows.append((t, z, beta_t, pos_prev, pos, m1_prev, m2_prev, m1, m2, cash, cash + m1 * y + m2 * x, entry_long, entry_short, exit_long, exit_short))

    ledger = pd.DataFrame(rows, columns=["time", "signal", "beta_used", "pos_prev", "pos", "m1_prev", "m2_prev", "m1", "m2", "cash", "book_value", "entry_long", "entry_short", "exit_long", "exit_short"]).set_index("time")
    if force_flat_end and len(ledger) > 0 and int(ledger["pos"].iloc[-1]) != 0:
        t_l = ledger.index[-1]
        m1_l, m2_l, cash_l = float(ledger["m1"].iloc[-1]), float(ledger["m2"].iloc[-1]), float(ledger["cash"].iloc[-1])
        cash_l += (m1_l * float(s1.loc[t_l]) + m2_l * float(s2.loc[t_l])) - (abs(m1_l) + abs(m2_l)) * c
        ledger.loc[t_l, ["cash", "m1", "m2", "pos", "book_value"]] = [cash_l, 0.0, 0.0, 0, cash_l]

    d_bv = ledger["book_value"].diff().dropna()
    sharpe = float(d_bv.mean() / d_bv.std(ddof=1) * math.sqrt(252.0)) if len(d_bv) > 2 and d_bv.std() > 0 else 0.0
    return BacktestResult(ledger, {"pnl": float(ledger["book_value"].iloc[-1] - ledger["book_value"].iloc[0]) if len(ledger) > 1 else 0.0, "sharpe": sharpe})

if __name__ == "__main__":
    mp.freeze_support()
    rng = np.random.default_rng(0)
    n = 800
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    s2 = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=idx)
    s1 = pd.Series(0.7 * s2 + 10 + rng.normal(0, 2, n), index=idx)
    hed = rolling_bayesian_rw_hedge_ratio(s1, s2, window=252, n_jobs=4)
    bt = backtest_dynamic_pairs_rolling(s1, s2, hed.alpha_hat, hed.beta_hat, ou_window=252, c=0.001)
    print(bt.metrics)
