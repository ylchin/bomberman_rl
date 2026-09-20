"""
1. parse_training_log(): turns the per-episode summary lines already
     printed by train.py (e.g. "ep 3000: steps=54 reward=9.3 coins=14
     invalid=2 self_kill=0 eps=0.01") into a clean CSV -- so training
     progress can be plotted, not just eyeballed in a text log.

2. plot_training_curve() / plot_eval_comparison(): matplotlib figures
     from those CSVs. Saved as PNG files under experiments/plots/.

Usage:
    # after a training run, extract its curve:
    python plot_results.py parse-log \
        --log agent_code/our_agent/logs/our_agent.log \
        --out experiments/task2_training_curve.csv

    # plot one training curve:
    python plot_results.py plot-training \
        --csv experiments/task2_training_curve.csv \
        --out experiments/plots/task2_training_curve.png

    # compare several eval CSVs (e.g. before/after more training, or
    # different tasks) as a grouped bar chart:
    python plot_results.py plot-eval-comparison \
        --csv experiments/task1_eval.csv:100ep \
        --csv experiments/task1_eval_3000ep_full.csv:3000ep \
        --out experiments/plots/task1_comparison.png"""

import argparse
import csv
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt


LOG_LINE_RE = re.compile(
    r"ep (\d+): steps=(\d+) reward=([-\d.]+) coins=(\d+) invalid=(\d+) "
    r"self_kill=(\d+) eps=([\d.]+)"
)

TRAINING_CSV_COLUMNS = ["episode", "steps", "reward", "coins", "invalid", "self_kill", "eps"]


def parse_training_log(log_path: Path, out_csv: Path):
    #Extracts every "ep N: steps=... reward=... ..." summary line from a train.py log file into a CSV.
    #Only relies on that one line format --
    rows = []
    with open(log_path) as f:
        for line in f:
            m = LOG_LINE_RE.search(line)
            if m:
                ep, steps, reward, coins, invalid, self_kill, eps = m.groups()
                rows.append({
                    "episode": int(ep), "steps": int(steps), "reward": float(reward),
                    "coins": int(coins), "invalid": int(invalid),
                    "self_kill": int(self_kill), "eps": float(eps),
                })

    if not rows:
        raise ValueError(
            f"No 'ep N: steps=...' lines found in {log_path}. "
            "Check the log actually has training episodes, or that the log "
            "message format still matches LOG_LINE_RE."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} episodes from {log_path} -> {out_csv}")
    return out_csv


def _rolling_mean(values, window=50):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def plot_training_curve(csv_path: Path, out_png: Path, window=50):
    #Three-panel figure from a parsed training-log CSV:
    #1. reward per episode (raw + rolling mean) - the main "is it learning" signal
    #2. eps decay - context for panel 1 (less random action = more signal)
    #3. self-kill rate, rolling - Task 2+ specific, but harmless to always show
    
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    episodes = [int(r["episode"]) for r in rows]
    reward = [float(r["reward"]) for r in rows]
    eps = [float(r["eps"]) for r in rows]
    self_kill = [int(r["self_kill"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(episodes, reward, alpha=0.25, color="tab:blue", label="reward (raw)")
    axes[0].plot(episodes, _rolling_mean(reward, window), color="tab:blue",
                 label=f"reward (rolling mean, w={window})")
    axes[0].set_ylabel("episode reward")
    axes[0].legend(loc="best")
    axes[0].set_title(f"Training curve -- {csv_path.name}")

    axes[1].plot(episodes, eps, color="tab:orange")
    axes[1].set_ylabel("epsilon")

    axes[2].plot(episodes, _rolling_mean(self_kill, window), color="tab:red")
    axes[2].set_ylabel(f"self-kill rate\n(rolling, w={window})")
    axes[2].set_xlabel("episode")
    axes[2].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def plot_eval_comparison(csv_label_pairs, out_png: Path):
    #Grouped bar chart comparing mean score, mean coins, and self-kill rate across several eval CSVs
    #eg. checkpoints at different training amounts, or different tasks. csv_label_pairs: list of (Path, label)
    labels, mean_scores, mean_coins, self_kill_rates = [], [], [], []

    for path, label in csv_label_pairs:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        scores = [float(r["score"]) for r in rows]
        coins = [float(r["coins"]) for r in rows]
        kills_self = [int(r["self_kill"]) for r in rows]

        labels.append(label)
        mean_scores.append(st.mean(scores))
        mean_coins.append(st.mean(coins))
        self_kill_rates.append(100 * st.mean(kills_self))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, title, ylabel in [
        (axes[0], mean_scores, "Mean score", "score"),
        (axes[1], mean_coins, "Mean coins", "coins"),
        (axes[2], self_kill_rates, "Self-kill rate", "% of rounds"),
    ]:
        ax.bar(labels, values, color="tab:blue")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def convert_train_csv(csv_path: Path, out_csv: Path):
    #converts training-CSV schema to the TRAINING_CSV_COLUMNS schema plot_training_curve() expects
    #(episode,steps,reward,coins,invalid,self_kill,eps). Only renames/selects columns

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for r in rows:
        out_rows.append({
            "episode": r["episode"],
            "steps": r["steps"],
            "reward": r["total_reward"],
            "coins": r["coins"],
            "invalid": r["invalid"],
            "self_kill": r["killed_self"],
            "eps": r["epsilon"],
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Converted {len(out_rows)} rows from {csv_path} -> {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Parse training logs and plot results")
    sub = parser.add_subparsers(dest="command", required=True)

    p0 = sub.add_parser("convert-train-csv",
                         help="Adapt training-CSV column names to the schema "
                              "plot-training expects.")
    p0.add_argument("--csv", type=Path, required=True)
    p0.add_argument("--out", type=Path, required=True)

    p1 = sub.add_parser("parse-log")
    p1.add_argument("--log", type=Path, required=True)
    p1.add_argument("--out", type=Path, required=True)

    p2 = sub.add_parser("plot-training")
    p2.add_argument("--csv", type=Path, required=True)
    p2.add_argument("--out", type=Path, required=True)
    p2.add_argument("--window", type=int, default=50)

    p3 = sub.add_parser("plot-eval-comparison")
    p3.add_argument("--csv", action="append", required=True,
                     help="path:label, repeatable, e.g. --csv foo.csv:100ep --csv bar.csv:3000ep")
    p3.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "convert-train-csv":
        convert_train_csv(args.csv, args.out)
    elif args.command == "parse-log":
        parse_training_log(args.log, args.out)
    elif args.command == "plot-training":
        plot_training_curve(args.csv, args.out, window=args.window)
    elif args.command == "plot-eval-comparison":
        pairs = []
        for item in args.csv:
            path_str, _, label = item.partition(":")
            pairs.append((Path(path_str), label or Path(path_str).stem))
        plot_eval_comparison(pairs, args.out)


if __name__ == "__main__":
    main()
