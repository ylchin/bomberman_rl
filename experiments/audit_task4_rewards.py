"""Attribute actual training rewards in a separate short Task 4 run.

Instruments train.py's reward calls without changing rewards, updates, or game
rules. The source checkpoint is only used for initialization; outputs require
a fresh run name. This measures learning with exploration, not evaluation.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Initialization checkpoint, relative to the repository root.")
    parser.add_argument("--run", required=True, help="New AGENT_RUN name.")
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    if args.n_rounds < 1:
        parser.error("--n-rounds must be positive")
    if not args.run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                           for c in args.run):
        parser.error("--run must contain only letters, digits, underscores or hyphens")
    os.chdir(ROOT)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = ROOT / "experiments" / args.run
    agent_dir = ROOT / "agent_code/our_agent"
    outputs = [output, agent_dir / f"training_{args.run}.csv",
               agent_dir / f"weights/q_linear_{args.run}.pkl",
               agent_dir / f"weights/q_linear_{args.run}_best.pkl",
               agent_dir / f"weights/candidates/linear_{args.run}"]
    if any(p.exists() for p in outputs):
        parser.error("Run outputs already exist; choose a new --run name")
    os.environ.update(
        AGENT_MODEL="linear", AGENT_PRESET="task4", AGENT_RUN=args.run,
        AGENT_SEED=str(args.seed), AGENT_INIT_CHECKPOINT=str(checkpoint),
        AGENT_SURVIVAL_FILTER="1", AGENT_BOMB_COLLISION_GUARD="1",
        AGENT_ESCAPE_COLLISION_GUARD="1",
        AGENT_OPTIMISTIC_FALLBACK="0", AGENT_COIN_PREFERENCE="0",
    )

    import main as game
    from agent_code.our_agent import train, rewards, config

    # Accumulate only calls made by the actual training pipeline. This includes
    # terminal event handling exactly as implemented, rather than reconstructing
    # a different reward sequence from an evaluation game.
    event_counts = Counter()
    event_totals = Counter()
    potential_stats = Counter()
    all_counts = Counter()
    all_totals = Counter()
    all_potential = Counter()
    episode_rows = []
    original_events = train.reward_from_events
    original_potential = train.potential_shaping
    original_terminal = train._terminal_potential_correction
    original_csv = train._write_csv_row

    def reward_from_events(events, logger=None):
        events = list(events)
        value = original_events(events, logger)
        for event in events:
            event_counts[event] += 1
            event_totals[event] += rewards.GAME_REWARDS.get(event, 0.0)
        return value

    def potential_shaping(old_state, new_state, gamma):
        value = original_potential(old_state, new_state, gamma)
        potential_stats["calls"] += 1
        potential_stats["net"] += value
        potential_stats["positive"] += max(0.0, value)
        potential_stats["negative"] += min(0.0, value)
        potential_stats["terminal_calls"] += int(new_state is None)
        return value

    def terminal_potential_correction(agent, last_state, already_recorded):
        value = original_terminal(agent, last_state, already_recorded)
        potential_stats["calls"] += 1
        potential_stats["net"] += value
        potential_stats["positive"] += max(0.0, value)
        potential_stats["negative"] += min(0.0, value)
        potential_stats["terminal_calls"] += 1
        potential_stats["terminal_net"] += value
        return value

    columns = ["episode", "coins", "kills", "game_score", "steps", "epsilon",
               "training_reward", "objective_reward", "other_event_reward",
               "potential_net", "potential_positive", "potential_negative",
               "potential_calls", "terminal_potential_calls",
               "event_counts", "event_rewards"]
    output.mkdir(parents=True)
    source_paths = [Path(__file__), ROOT / "settings.py", ROOT / "environment.py"]
    source_paths += [agent_dir / name for name in
                     ["train.py", "rewards.py", "features.py", "escape_planner.py",
                      "callbacks.py", "config.py", "q_linear.py", "replay_buffer.py"]]
    protocol = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": source_hash,
        "run": args.run, "rounds": args.n_rounds, "seed": args.seed,
        "training_config": config.TRAIN, "reward_table": rewards.GAME_REWARDS,
        "escape_reward_scale": rewards.ESCAPE_REWARD_SCALE,
        "coin_tie_reward": rewards.COIN_TIE_REWARD,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        "protocol": "Separate Task 4 training run with both guards on. Original training reward and updates; rewards observed, not modified. Opponents retain unseeded randomness.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with (output / "reward_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        def write_csv_row(agent, last_state):
            original_csv(agent, last_state)
            attributed = sum(event_totals.values()) + potential_stats["net"]
            if not math.isclose(attributed, agent.ep_reward, rel_tol=1e-9, abs_tol=1e-7):
                raise RuntimeError(f"Reward accounting mismatch: {attributed} vs {agent.ep_reward}")
            objective = event_totals["COIN_COLLECTED"] + event_totals["KILLED_OPPONENT"]
            row = dict(
                episode=agent.episode, coins=event_counts["COIN_COLLECTED"],
                kills=event_counts["KILLED_OPPONENT"],
                game_score=event_counts["COIN_COLLECTED"] + 5 * event_counts["KILLED_OPPONENT"],
                steps=last_state["step"], epsilon=agent.epsilon,
                training_reward=agent.ep_reward, objective_reward=objective,
                other_event_reward=sum(event_totals.values()) - objective,
                potential_net=potential_stats["net"],
                potential_positive=potential_stats["positive"],
                potential_negative=potential_stats["negative"],
                potential_calls=potential_stats["calls"],
                terminal_potential_calls=potential_stats["terminal_calls"],
                event_counts=json.dumps(dict(event_counts), sort_keys=True),
                event_rewards=json.dumps(dict(event_totals), sort_keys=True),
            )
            writer.writerow(row)
            stream.flush()
            episode_rows.append(row)
            all_counts.update(event_counts)
            all_totals.update(event_totals)
            all_potential.update(potential_stats)
            event_counts.clear()
            event_totals.clear()
            potential_stats.clear()

        train.reward_from_events = reward_from_events
        train.potential_shaping = potential_shaping
        train._terminal_potential_correction = terminal_potential_correction
        train._write_csv_row = write_csv_row
        try:
            game.main(["play", "--agents", "our_agent", "rule_based_agent",
                       "rule_based_agent", "rule_based_agent", "--scenario", "classic",
                       "--train", "1", "--no-gui", "--n-rounds", str(args.n_rounds),
                       "--seed", str(args.seed)])
        finally:
            train.reward_from_events = original_events
            train.potential_shaping = original_potential
            train._terminal_potential_correction = original_terminal
            train._write_csv_row = original_csv
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash
    n = len(episode_rows)
    assert n == args.n_rounds
    summary = {
        "episodes": n,
        "coin_tie_reward": rewards.COIN_TIE_REWARD,
        "mean_game_score": sum(r["game_score"] for r in episode_rows) / n,
        "mean_training_reward": sum(r["training_reward"] for r in episode_rows) / n,
        "mean_potential": {key: value / n for key, value in all_potential.items()},
        "events": {key: {"mean_count": all_counts[key] / n, "mean_reward": all_totals[key] / n}
                   for key in sorted(all_counts, key=lambda key: abs(all_totals[key]), reverse=True)},
        "limits": "Undiscounted per-episode attribution during learning and exploration; large reward contributions alone do not prove exploitation. Terminal shaping is observed as implemented, not corrected by this runner.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
