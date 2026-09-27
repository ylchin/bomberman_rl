"""Compare two frozen linear policies on identical safe, coin-visible states.

Sample both policies' trajectories, then analyze outside the timed act callback.
This diagnoses action preferences; it does not estimate a win-rate difference.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_code.our_agent.features import (
    ACTIONS, FEATURE_NAMES, state_to_features,
)
from agent_code.our_agent.escape_planner import survival_actions
from agent_code.our_agent.q_linear import LinearQ
from agent_code.our_agent.coin_navigation import safe_coin_actions


def compare_state(state, baseline, candidate):
    coin_ids, distance = safe_coin_actions(state)
    if not coin_ids:
        return None
    phi = state_to_features(state)
    mask = survival_actions(state, bomb_collision_guard=True, escape_collision_guard=True)
    allowed = LinearQ.available_actions(phi, mask)
    coin_ids = [a for a in coin_ids if allowed[a]]
    if not coin_ids:
        return None
    models = [baseline, candidate]
    values = [model.q_values(phi) for model in models]
    best = [np.flatnonzero(allowed & (q == q[allowed].max())).tolist() for q in values]
    fractions = [sum(a in coin_ids for a in actions) / len(actions) for actions in best]
    regression = fractions[0] == 1.0 and fractions[1] == 0.0
    improvement = fractions[0] == 0.0 and fractions[1] == 1.0
    row = {
        "step": state["step"], "coin_distance": distance,
        "coin_actions": [ACTIONS[a] for a in coin_ids],
        "baseline_greedy": [ACTIONS[a] for a in best[0]],
        "candidate_greedy": [ACTIONS[a] for a in best[1]],
        "baseline_coin_probability": fractions[0],
        "candidate_coin_probability": fractions[1],
        "strict_coin_preference_loss": regression,
        "strict_coin_preference_gain": improvement,
        "allowed_actions": [ACTIONS[a] for a in np.flatnonzero(allowed)],
    }
    example = None
    if regression:
        coin_action = best[0][0]
        alternative = best[1][0]
        contributions = [(model.W[alternative] - model.W[coin_action]) * phi for model in models]
        shift = contributions[1] - contributions[0]
        indices = np.argsort(-np.abs(shift))[:8]
        example = {
            **row, "state": state, "coin_reference": ACTIONS[coin_action],
            "candidate_alternative": ACTIONS[alternative],
            "baseline_alternative_minus_coin": float(contributions[0].sum()),
            "candidate_alternative_minus_coin": float(contributions[1].sum()),
            "baseline_q": dict(zip(ACTIONS, values[0].tolist())),
            "candidate_q": dict(zip(ACTIONS, values[1].tolist())),
            "largest_feature_shifts": [
                {"feature": FEATURE_NAMES[i], "value": float(phi[i]),
                 "baseline_contribution": float(contributions[0][i]),
                 "candidate_contribution": float(contributions[1][i]),
                 "shift_toward_alternative": float(shift[i])}
                for i in indices if shift[i] != 0
            ],
        }
    return row, example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path(
        "agent_code/our_agent/weights/candidates/linear_model1_safety_v2_seed1/ep_04000.pkl"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--n-rounds", type=int, default=10, help="Games per driving policy.")
    parser.add_argument("--seed-start", type=int, default=28000)
    parser.add_argument("--stride", type=int, default=5, help="Sample every N decisions.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n_rounds < 1 or args.stride < 1:
        parser.error("--n-rounds and --stride must be positive")
    paths = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    models = {name: LinearQ.load(path) for name, path in paths.items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    os.environ.update(AGENT_MODEL="linear", AGENT_SURVIVAL_FILTER="1",
                      AGENT_BOMB_COLLISION_GUARD="1", AGENT_ESCAPE_COLLISION_GUARD="1",
                      AGENT_COIN_PREFERENCE="0", AGENT_OPTIMISTIC_FALLBACK="0")
    import main as game
    from agent_code.our_agent import callbacks, config

    original_act = callbacks.act
    samples = []

    def act(agent, state):
        action = original_act(agent, state)
        if (state["step"] - 1) % args.stride == 0:
            samples.append((state, action))
        return action

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    protocol = {"checkpoints": {name: str(path) for name, path in paths.items()},
                "checkpoint_sha256": hashes, "seed_start": args.seed_start,
                "rounds_per_driver": args.n_rounds, "stride": args.stride,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), ROOT / "agent_code/our_agent/features.py",
                                            ROOT / "agent_code/our_agent/escape_planner.py",
                                            ROOT / "agent_code/our_agent/q_linear.py",
                                            ROOT / "agent_code/our_agent/coin_navigation.py",
                                            ROOT / "agent_code/our_agent/callbacks.py"]},
                "protocol": "Frozen policies; both guards enabled. Compare exact greedy tie sets on identical sampled states. Opponents retain unseeded randomness."}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, examples = [], []
    sampled_count = Counter()
    callbacks.act = act
    try:
        with (output / "states.jsonl").open("w") as states_file:
            for index in range(args.n_rounds):
                seed = args.seed_start + index
                for driver in (paths if index % 2 == 0 else reversed(paths)):
                    config.EVAL_WEIGHTS = [str(paths[driver])]
                    config.TRAIN["seed"] = seed
                    samples.clear()
                    game.main(["play", "--agents", "our_agent", "rule_based_agent",
                               "rule_based_agent", "rule_based_agent", "--scenario", "classic",
                               "--seed", str(seed), "--no-gui", "--n-rounds", "1"])
                    sampled_count[driver] += len(samples)
                    for state, action in samples:
                        result = compare_state(state, models["baseline"], models["candidate"])
                        if result is None:
                            continue
                        row, example = result
                        row.update(driver=driver, seed=seed, actual_driver_action=action)
                        rows.append(row)
                        states_file.write(json.dumps({"driver": driver, "seed": seed,
                                                      "state": state}, default=encode) + "\n")
                        if example is not None:
                            example.update(driver=driver, seed=seed, actual_driver_action=action)
                            examples.append(example)
                    states_file.flush()
                    for logger in logging.Logger.manager.loggerDict.values():
                        if isinstance(logger, logging.Logger):
                            for handler in list(logger.handlers):
                                handler.close()
                                logger.removeHandler(handler)
                print(f"Sampled {index + 1}/{args.n_rounds} pairs; {len(rows)} comparable states", flush=True)
    finally:
        callbacks.act = original_act
    for name, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]

    def summarize(items):
        n = len(items)
        return {"comparable_states": n,
                "baseline_mean_coin_probability": sum(r["baseline_coin_probability"] for r in items) / n if n else None,
                "candidate_mean_coin_probability": sum(r["candidate_coin_probability"] for r in items) / n if n else None,
                "strict_coin_preference_losses": sum(r["strict_coin_preference_loss"] for r in items),
                "strict_coin_preference_gains": sum(r["strict_coin_preference_gain"] for r in items),
                "candidate_actions_in_losses": dict(Counter(
                    a for r in items if r["strict_coin_preference_loss"] for a in r["candidate_greedy"]))}

    summary = {"sampled_states": dict(sampled_count), "overall": summarize(rows),
               "by_driver": {driver: summarize([r for r in rows if r["driver"] == driver]) for driver in paths},
               "limits": "Correlated, trajectory-selected states, not independent trials. Safe shortest-path coin seeking is a diagnostic reference, not always the best strategic action. Tie sets are compared without the history-dependent stuck-action override. Q values are uncalibrated across checkpoints; feature shifts describe ranking changes, not causes of the training regression."}
    examples.sort(key=lambda x: x["candidate_alternative_minus_coin"], reverse=True)
    summary["example_preference_losses"] = [
        {key: item[key] for key in [
            "driver", "seed", "step", "coin_actions", "baseline_greedy", "candidate_greedy",
            "baseline_alternative_minus_coin", "candidate_alternative_minus_coin",
            "largest_feature_shifts",
        ]}
        for item in examples[:3]
    ]
    (output / "examples.json").write_text(json.dumps(examples[:10], default=encode, indent=2) + "\n")
    if rows:
        with (output / "decisions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
