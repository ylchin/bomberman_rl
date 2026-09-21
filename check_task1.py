"""
check_task1.py

Checks a Task 1 evaluate.py output CSV against the exit criteria:
>=95% of coins collected, averaged over the held-out seeds, treating any
round that ran past 200 steps as not meeting the "within 200 steps" part.

Usage:
    python check_task1.py experiments/cnn_task1_full_eval.csv
"""

import sys
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "experiments/cnn_task1_full_eval.csv"

# Total coins spawned in coin-heaven. 
TOTAL_COINS = 50

df = pd.read_csv(CSV_PATH)
print(df[["seed", "coins", "steps_survived", "invalid_actions", "self_kill"]].describe())

df["coin_fraction"] = (df["coins"] / TOTAL_COINS).clip(upper=1.0)
df["within_200"] = df["steps_survived"] <= 200
df["criteria_score"] = df["coin_fraction"].where(df["within_200"], 0.0)

mean_score = df["criteria_score"].mean()
print(f"\nRounds finishing within 200 steps: {df['within_200'].sum()} / {len(df)}")
print(f"Mean coin fraction (all rounds):   {df['coin_fraction'].mean():.3f}")
print(f"Mean self-kill rate:               {df['self_kill'].mean():.3f}")
print(f"Exit-criteria score (avg):         {mean_score:.3f}")
print("PASS" if mean_score >= 0.95 else "FAIL", "(threshold: 0.95)")
