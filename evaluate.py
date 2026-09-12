"""
evaluate.py

Headless evaluation harness (Person A). Runs N seeded rounds against a
fixed opponent lineup and writes one CSV row per round, so every claimed
improvement in the report is measured the same way (assignment Section 04:
held-out seeds, >=100 rounds, exploration off, original settings.py, no
auxiliary reward -- true game score only).

Confirmed against a real --save-stats output (2026-09-12):
    {
      "by_agent": {
        "<agent_name>": {
          "score": int, "steps": int, "invalid": int,
          "coins": int (absent if 0), "suicides": int (absent if 0),
          "bombs": int, "crates": int, "moves": int, "rounds": int, "time": float
        }, ...
      },
      "by_round": {"Round 01 (...)": {"coins": int, "kills": int, "steps": int, "suicides": int}}
    }
There is no per-agent "kills" field -- score = coins*1 + kills*5 (assignment
Sec. 3), so kills is derived as (score - coins) / 5.

CSV columns (fixed -- do not rename without updating downstream plotting):
    seed, round, agent_name, score, coins, kills, self_kill,
    steps_survived, invalid_actions

Usage:
    python evaluate.py --agent our_agent \
        --opponents rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --n-rounds 100 --seed-start 10000 \
        --out experiments/our_agent_vs_rulebased.csv
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STATS_DIR = REPO_ROOT / "results"

CSV_COLUMNS = [
    "seed", "round", "agent_name", "score", "coins",
    "kills", "self_kill", "steps_survived", "invalid_actions",
]


def run_single_seed(agent: str, opponents: list, scenario: str, seed: int) -> Path:
    """
    Runs one seeded, single-round game with --agents (agent first, then
    opponents) and --save-stats pointed at a seed-specific file, so we
    never have to guess which output file belongs to which run.
    """
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = STATS_DIR / f"eval_{agent}_seed{seed}.json"

    cmd = [
        sys.executable, "main.py", "play",
        "--agents", agent, *opponents,
        "--scenario", scenario,
        "--seed", str(seed),
        "--no-gui",
        "--n-rounds", "1",
        "--save-stats", str(stats_path),
    ]

    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"main.py exited with code {proc.returncode} for seed {seed}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if not stats_path.exists():
        raise RuntimeError(
            f"Expected stats file {stats_path} was not created.\nstdout:\n{proc.stdout}"
        )
    return stats_path


def parse_result_file(path: Path, agent_name: str, seed: int, round_num: int) -> dict:
    """Parses one --save-stats JSON file into our fixed CSV schema."""
    with open(path) as f:
        data = json.load(f)

    by_agent = data.get("by_agent", {})
    agent_data = by_agent.get(agent_name) or by_agent.get(f"{agent_name}_0")
    if agent_data is None:
        raise KeyError(
            f"agent '{agent_name}' not found in by_agent keys: {list(by_agent.keys())}"
        )

    score = agent_data.get("score", 0)
    coins = agent_data.get("coins", 0)      # key absent when 0
    kills = round((score - coins) / 5)      # score = coins*1 + kills*5 (assignment Sec.3)
    self_kill = int(agent_data.get("suicides", 0) > 0)  # key absent when 0

    return {
        "seed": seed,
        "round": round_num,
        "agent_name": agent_name,
        "score": score,
        "coins": coins,
        "kills": kills,
        "self_kill": self_kill,
        "steps_survived": agent_data.get("steps", 0),
        "invalid_actions": agent_data.get("invalid", 0),
    }


def run_evaluation(agent, opponents, scenario, n_rounds, seed_start, out_csv: Path,
                    keep_stats_files: bool = False) -> Path:
    """
    Runs n_rounds separate seeded games (seed_start, seed_start+1, ...) and
    writes every round's result as one CSV row. Re-running with the same
    seed_start reproduces the identical evaluation -- this IS the
    held-out-seed protocol: always evaluate this way, never compare runs
    on different seeds or eyeball a single game.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(n_rounds):
        seed = seed_start + i
        stats_path = run_single_seed(agent, opponents, scenario, seed)
        row = parse_result_file(stats_path, agent, seed, round_num=i)
        rows.append(row)
        print(f"[{i + 1}/{n_rounds}] seed={seed} score={row['score']} "
              f"coins={row['coins']} kills={row['kills']} self_kill={row['self_kill']}")
        if not keep_stats_files:
            stats_path.unlink()  # avoid littering results/ with hundreds of tiny files

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_csv}")
    return out_csv


def summarize(csv_path: Path):
    """Console summary: mean score, coins, kills, self-kill rate."""
    import statistics as st
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{csv_path} has no rows.")
        return

    scores = [float(r["score"]) for r in rows]
    self_kills = [int(r["self_kill"]) for r in rows]
    coins = [float(r["coins"]) for r in rows]
    kills = [float(r["kills"]) for r in rows]

    print(f"\n--- {csv_path.name} ({len(rows)} rounds) ---")
    print(f"mean score:      {st.mean(scores):.2f} (pop.std {st.pstdev(scores):.2f})")
    print(f"mean coins:      {st.mean(coins):.2f}")
    print(f"mean kills:      {st.mean(kills):.2f}")
    print(f"self-kill rate:  {100 * st.mean(self_kills):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Headless seeded evaluation harness")
    parser.add_argument("--agent", required=True, help="agent_code/ subfolder name to evaluate")
    parser.add_argument("--opponents", nargs="+", default=["rule_based_agent"] * 3)
    parser.add_argument("--scenario", default="classic",
                         choices=["empty", "coin-heaven", "loot-crate", "classic"])
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000,
                         help="Fixed, documented held-out seed range -- never reuse "
                              "seeds also used during training.")
    parser.add_argument("--out", type=Path, default=Path("experiments/eval_results.csv"))
    parser.add_argument("--keep-stats-files", action="store_true",
                         help="Keep the per-seed raw JSON in results/ instead of deleting after parsing.")
    args = parser.parse_args()

    csv_path = run_evaluation(
        agent=args.agent, opponents=args.opponents, scenario=args.scenario,
        n_rounds=args.n_rounds, seed_start=args.seed_start, out_csv=args.out,
        keep_stats_files=args.keep_stats_files,
    )
    summarize(csv_path)


if __name__ == "__main__":
    main()
