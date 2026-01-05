import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BACKTEST_RE = re.compile(r"^backtest_(.+)_(\w+)_w(\d+)\.feather$")


def parse_interval_to_periods(interval: str) -> Optional[int]:
    text = interval.strip().lower()
    if text.endswith("d"):
        try:
            days = int(text[:-1])
        except ValueError:
            return None
        return 365 // max(days, 1)
    if text.endswith("h"):
        try:
            hours = int(text[:-1])
        except ValueError:
            return None
        return int(365 * 24 / max(hours, 1))
    return None


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: Optional[int]) -> float:
    ret = returns.dropna()
    if ret.empty:
        return 0.0
    std = float(ret.std())
    if std == 0.0:
        return 0.0
    scale = np.sqrt(periods_per_year) if periods_per_year else 1.0
    return float(ret.mean() / std * scale)


def extract_pair_parts(pair_id: str) -> Tuple[str, str, str]:
    if "__" in pair_id:
        y, x = pair_id.split("__", 1)
        name = f"{y}-{x}"
        return name, y, x
    return pair_id, "", ""


def summarize_backtest(path: str) -> Optional[Dict[str, object]]:
    name = os.path.basename(path)
    m = BACKTEST_RE.match(name)
    if not m:
        return None

    pair_id, interval, window = m.group(1), m.group(2), int(m.group(3))

    try:
        df = pd.read_feather(path)
    except Exception:
        return None

    if "equity" not in df.columns or df["equity"].empty:
        return None

    equity = df["equity"].astype(float)
    start_equity = float(equity.iloc[0])
    final_equity = float(equity.iloc[-1])
    pnl = final_equity - start_equity
    return_pct = (final_equity / start_equity - 1.0) * 100 if start_equity else 0.0

    if "strategy_return" in df.columns:
        returns = df["strategy_return"].astype(float)
    else:
        returns = equity.pct_change().fillna(0.0)

    periods = parse_interval_to_periods(interval)
    sharpe = sharpe_ratio(returns, periods)
    mdd = max_drawdown(equity) * 100.0
    trades = int(df["turnover"].gt(0).sum()) if "turnover" in df.columns else 0

    pair_name, y_symbol, x_symbol = extract_pair_parts(pair_id)

    return {
        "pair_id": pair_id,
        "pair": pair_name,
        "y_symbol": y_symbol,
        "x_symbol": x_symbol,
        "interval": interval,
        "window": window,
        "start_equity": start_equity,
        "final_equity": final_equity,
        "pnl": pnl,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd,
        "trades": trades,
        "file": name,
    }


def collect_results(output_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for name in os.listdir(output_dir):
        if not name.endswith(".feather"):
            continue
        path = os.path.join(output_dir, name)
        result = summarize_backtest(path)
        if result:
            rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank backtest outputs stored in data/output."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "output"),
        help="Directory with backtest_*.feather files.",
    )
    parser.add_argument("--top", type=int, default=20, help="Rows to print.")
    parser.add_argument(
        "--sort",
        default="final_equity",
        choices=[
            "final_equity",
            "return_pct",
            "pnl",
            "sharpe",
            "max_drawdown_pct",
            "trades",
        ],
        help="Ranking metric (descending except max_drawdown_pct).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path (default: output_dir/backtest_ranking.csv).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        raise SystemExit(f"Output directory not found: {output_dir}")

    df = collect_results(output_dir)
    if df.empty:
        raise SystemExit("No valid backtest files found.")

    ascending = args.sort == "max_drawdown_pct"
    df_sorted = df.sort_values(args.sort, ascending=ascending).reset_index(drop=True)

    top = df_sorted.head(args.top)
    display_cols = [
        "pair",
        "interval",
        "window",
        "final_equity",
        "return_pct",
        "pnl",
        "sharpe",
        "max_drawdown_pct",
        "trades",
        "file",
    ]
    print(top[display_cols].to_string(index=False))

    csv_path = args.csv or os.path.join(output_dir, "backtest_ranking.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nSaved full ranking to {csv_path}")


if __name__ == "__main__":
    main()
