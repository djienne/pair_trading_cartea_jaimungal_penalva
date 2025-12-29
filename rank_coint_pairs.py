import argparse
import concurrent.futures as cf
import itertools
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from coint_calibrate import calibrate_pair
from utils import load_config, get_dirs, get_symbol_path


def fetch_usdt_symbols(base_url: str) -> List[str]:
    url = f"{base_url}/fapi/v1/exchangeInfo"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    symbols = []
    for item in data.get("symbols", []):
        if (
            item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
            and item.get("status") == "TRADING"
        ):
            symbols.append(item["symbol"])
    return sorted(set(symbols))


def filter_available_symbols(symbols: List[str], feather_dir: str, interval: str) -> List[str]:
    available = []
    for symbol in symbols:
        path = get_symbol_path(symbol, interval, feather_dir)
        if os.path.exists(path):
            available.append(symbol)
    return available


def score_coint(df: pd.DataFrame) -> Dict:
    pvals = df["adf_pvalue"].dropna()
    if pvals.empty:
        return {}
    mask = df["adf_pvalue"].notna()
    pass_rate = float(df.loc[mask, "adf_pass"].mean())
    return {
        "median_p": float(pvals.median()),
        "mean_p": float(pvals.mean()),
        "min_p": float(pvals.min()),
        "last_p": float(pvals.iloc[-1]),
        "pass_rate": pass_rate,
        "n_windows": int(pvals.shape[0]),
    }


def run_pair_calibration(args: Tuple[str, str, Dict]) -> Optional[Dict]:
    y_symbol, x_symbol, config_run = args
    pair = {"y_symbol": y_symbol, "x_symbol": x_symbol}
    try:
        df = calibrate_pair(pair, config_run)
    except Exception as exc:
        return {
            "error": str(exc),
            "y_symbol": y_symbol,
            "x_symbol": x_symbol,
        }

    metrics = score_coint(df)
    if not metrics:
        return {
            "error": "no_pvalues",
            "y_symbol": y_symbol,
            "x_symbol": x_symbol,
        }
    if metrics.get("n_windows", 0) < 1000:
        return {
            "error": "insufficient_windows",
            "y_symbol": y_symbol,
            "x_symbol": x_symbol,
        }

    window = int(config_run.get("rolling_window_days", 30))
    return {
        "y_symbol": y_symbol,
        "x_symbol": x_symbol,
        "pair": f"{y_symbol}-{x_symbol}",
        "window": window,
        **metrics,
    }


def rank_pairs(
    config: Dict,
    symbols: List[str],
    max_pairs: int,
    log_adf: bool,
    max_workers: int,
) -> pd.DataFrame:
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 30))
    feather_dir, _, _ = get_dirs(config)

    available = filter_available_symbols(symbols, feather_dir, interval)
    missing = sorted(set(symbols) - set(available))
    if missing:
        print(f"Skipping {len(missing)} symbols without data: {', '.join(missing)}")

    if len(available) < 2:
        raise ValueError("Need at least two symbols with cached data to rank pairs.")

    config_run = dict(config)
    if not log_adf:
        config_run["log_adf_each_window"] = False

    combos = list(itertools.combinations(available, 2))
    if max_pairs is not None and max_pairs > 0:
        combos = combos[: int(max_pairs)]

    rows = []
    errors = 0
    if combos:
        workers = max(1, int(max_workers))
        workers = min(workers, len(combos))
        with cf.ProcessPoolExecutor(max_workers=workers) as executor:
            tasks = (
                (y_symbol, x_symbol, config_run) for y_symbol, x_symbol in combos
            )
            for result in executor.map(run_pair_calibration, tasks):
                if not result:
                    continue
                if "error" in result:
                    errors += 1
                    continue
                rows.append(result)

    if errors:
        print(f"Skipped {errors} pairs due to errors or missing p-values.")

    if not rows:
        raise ValueError("No valid cointegration results to rank.")

    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(["median_p", "mean_p", "min_p"]).reset_index(drop=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank USDT perpetual pairs by cointegration ADF p-values."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional list of symbols to evaluate (e.g., ETHUSDT UNIUSDT).",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit number of symbols (after fetch/sort).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit number of pairs to evaluate.",
    )
    parser.add_argument(
        "--log-adf",
        action="store_true",
        help="Log ADF pass/fail per window while calibrating.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    base_url = config.get("binance_base_url", "https://fapi.binance.com")

    if args.symbols:
        symbols = sorted(set(args.symbols))
    else:
        symbols = fetch_usdt_symbols(base_url)
        max_symbols = args.max_symbols or config.get("ranking_max_symbols")
        if max_symbols:
            symbols = symbols[: int(max_symbols)]

    max_workers = int(config.get("ranking_threads", 4))
    ranked = rank_pairs(
        config, symbols, args.max_pairs, args.log_adf, max_workers=max_workers
    )

    output_dir = config.get("output_dir") or os.path.join(
        config.get("data_dir", "data"), "output"
    )
    os.makedirs(output_dir, exist_ok=True)

    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 30))
    out_path = os.path.join(
        output_dir, f"coint_rankings_{interval}_w{window}.csv"
    )
    ranked.to_csv(out_path, index=False)
    print(f"Saved rankings to {out_path}")
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()