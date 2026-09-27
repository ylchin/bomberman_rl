"""
check_task1.py

Reports descriptive metrics for a Task 1 evaluate.py output CSV
(coin-heaven: no crates, no opponents).

Usage:
    python check_task1.py experiments/cnn_task1_full_eval.csv [more.csv ...]

    # or, with no arguments, it globs all task1 eval CSVs in experiments/:
    python check_task1.py
"""

import sys
import glob
import pandas as pd

DEFAULT_GLOB = "experiments/*task1*eval*.csv"

# Fallback used if a CSV doesn't have a round_total_coins column
# Coin-heaven scenario default.
FALLBACK_TOTAL_COINS = 50


def summarize(path):
    df = pd.read_csv(path)

    if "round_total_coins" in df.columns:
        denom = df["round_total_coins"].replace(0, pd.NA)
    else:
        print(f"  [!] {path}: no round_total_coins column, assuming "
              f"{FALLBACK_TOTAL_COINS} coins/round")
        denom = FALLBACK_TOTAL_COINS

    coin_fraction = (df["coins"] / denom).clip(upper=1.0)
    # Rounds with 0 coins spawned trivially count as fully collected.
    coin_fraction = coin_fraction.fillna(1.0)

    row = {
        "file": path,
        "n_rounds": len(df),
        "mean_coin_fraction": coin_fraction.mean(),
        "mean_steps_survived": df["steps_survived"].mean(),
        "pct_within_200_steps": (df["steps_survived"] <= 200).mean() * 100,
        "pct_within_400_steps": (df["steps_survived"] <= 400).mean() * 100,
        "mean_self_kill_rate": df["self_kill"].mean(),
        "mean_invalid_actions": df["invalid_actions"].mean(),
    }
    return row, df


def main(paths):
    if not paths:
        paths = sorted(glob.glob(DEFAULT_GLOB))
        if not paths:
            print(f"No files given and none matched {DEFAULT_GLOB!r}")
            return

    rows = []
    for path in paths:
        print(f"\n=== {path} ===")
        row, df = summarize(path)
        print(df[["seed", "coins", "steps_survived", "invalid_actions", "self_kill"]].describe())
        print(f"Mean coin fraction:        {row['mean_coin_fraction']:.3f}")
        print(f"Mean steps survived:       {row['mean_steps_survived']:.1f}")
        print(f"Rounds within 200 steps:   {row['pct_within_200_steps']:.1f}%")
        print(f"Rounds within 400 steps:   {row['pct_within_400_steps']:.1f}%")
        print(f"Mean self-kill rate:       {row['mean_self_kill_rate']:.3f}")
        print(f"Mean invalid actions:      {row['mean_invalid_actions']:.3f}")
        rows.append(row)

    if len(rows) > 1:
        summary = pd.DataFrame(rows).set_index("file")
        print("\n=== Summary across all files ===")
        print(summary.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1:])
