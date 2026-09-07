"""
evaluate.py  

Headless evaluation harness: runs an agent for N seeded rounds against a
fixed set of opponents, in evaluation mode (no exploration, no training
updates), and writes one row per round to a CSV.

*** BEFORE FIRST USE ***
Run `python main.py play --help` in the actual repo and confirm/adjust:
  1. The exact flag names for: agent list, --no-gui, --seed, --n-rounds,
     and (importantly) whether there's a flag to disable exploration /
     force self.train=False for OUR agent specifically while still running
     rule_based_agent normally.
  2. Where per-round results actually get written. Some versions of this
     framework write a JSON summary to `results/<agent_name>-<timestamp>.json`
     per round; others only print to stdout/log. `_parse_round_output()`
     below is a best-effort parser for the JSON-file convention described
     in the assignment — swap it for a stdout/log parser if that's what
     your repo's main.py actually produces. Everything else in this file
     (the seed loop, CSV schema, aggregation) does not depend on that
     choice and does not need to change.

CSV schema (one row per round):
    seed, round, score, coins, kills, self_kill (bool), steps_survived,
    invalid_actions
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # adjust if evaluate.py lives elsewhere
RESULTS_DIR = REPO_ROOT / "results"

CSV_COLUMNS = [
    "seed", "round", "score", "coins", "kills",
    "self_kill", "steps_survived", "invalid_actions",
]


def run_single_round(agent_name, opponents, seed, no_gui=True, extra_args=None):
    """
    Launches one round of main.py with a fixed seed, agent under test as
    --my-agent, given opponents, evaluation mode (no training).
    Returns the path to whatever result artifact was produced (or None),
    for _parse_round_output to consume. Raises CalledProcessError on crash
    (which you WANT to see, not silently swallow, during eval).
    """
    cmd = [
        sys.executable, "main.py", "play",
        "--my-agent", agent_name,
        "--agents", *opponents,
        "--seed", str(seed),
        "--n-rounds", "1",
    ]
    if no_gui:
        cmd.append("--no-gui")
    if extra_args:
        cmd.extend(extra_args)

    before = set(RESULTS_DIR.glob("*")) if RESULTS_DIR.exists() else set()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    after = set(RESULTS_DIR.glob("*")) if RESULTS_DIR.exists() else set()

    new_files = after - before
    return max(new_files, key=lambda p: p.stat().st_mtime) if new_files else None


def _parse_round_output(result_path, agent_name):
    """
    TODO(verify): adjust to match the actual results/ file format in the
    repo. Assumed shape below (edit once confirmed):

        {
          "agents": {
            "<agent_name>": {
              "score": int,
              "coins": int,
              "kills": int,
              "self_kill": bool,
              "steps_survived": int,
              "invalid_actions": int
            },
            ...
          }
        }
    """
    if result_path is None:
        raise RuntimeError(
            "No result artifact found after round — confirm main.py's actual "
            "output location/format and update _parse_round_output()."
        )
    with open(result_path) as f:
        data = json.load(f)
    stats = data["agents"][agent_name]
    return {
        "score": stats.get("score", 0),
        "coins": stats.get("coins", 0),
        "kills": stats.get("kills", 0),
        "self_kill": bool(stats.get("self_kill", False)),
        "steps_survived": stats.get("steps_survived", 0),
        "invalid_actions": stats.get("invalid_actions", 0),
    }


def run_evaluation(agent_name, opponents, seeds, out_csv, no_gui=True, extra_args=None):
    """
    Runs len(seeds) rounds, writes one CSV row per round to out_csv.
    `opponents` is a list of agent names, e.g. ['rule_based_agent'] or
    ['rule_based_agent', 'rule_based_agent', 'rule_based_agent'].
    `seeds` should be the team's fixed held-out evaluation seed list
    (per the project plan's Section 04 protocol) — never seeds used in
    training.
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, seed in enumerate(seeds):
            result_path = run_single_round(agent_name, opponents, seed, no_gui=no_gui,
                                            extra_args=extra_args)
            stats = _parse_round_output(result_path, agent_name)
            row = {"seed": seed, "round": i, **stats}
            writer.writerow(row)
            print(f"[{i + 1}/{len(seeds)}] seed={seed} score={stats['score']} "
                  f"coins={stats['coins']} kills={stats['kills']} "
                  f"self_kill={stats['self_kill']}")

    print(f"Wrote {len(seeds)} rows to {out_path}")
    return out_path


def summarize(csv_path):
    """
    Quick console summary (mean +/- std for the key metrics) — full analysis
    and plots belong in a separate script that reads this CSV, but this is
    useful as a sanity check right after a run.
    """
    import statistics as stats_mod

    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    if n == 0:
        print("No rows to summarize.")
        return

    def col(name, cast=float):
        return [cast(r[name]) for r in rows]

    scores = col("score")
    kills = col("kills")
    self_kills = [r["self_kill"] == "True" for r in rows]

    print(f"n_rounds={n}")
    print(f"score:      mean={stats_mod.mean(scores):.2f}  std={stats_mod.pstdev(scores):.2f}")
    print(f"kills:      mean={stats_mod.mean(kills):.2f}")
    print(f"self_kill rate: {100 * sum(self_kills) / n:.1f}%")


if __name__ == "__main__":
    # Example usage — adjust agent_name / opponents / seed range per task.
    # Task 1 (coin-heaven) example:
    EVAL_SEEDS = list(range(1000, 1100))  # 100 held-out seeds, never used in training
    out = run_evaluation(
        agent_name="our_agent",
        opponents=[],  # Task 1: no opponents
        seeds=EVAL_SEEDS,
        out_csv="experiments/task1_baseline.csv",
        extra_args=["--scenario", "coin-heaven"],
    )
    summarize(out)
