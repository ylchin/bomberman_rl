"""Compare saved weights on recorded coin opportunities, without running games.

Reports raw checkpoints and symmetry projections computed only in memory.
Does not update weights, save checkpoints, or estimate game performance.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_code.our_agent.coin_navigation import safe_coin_actions
from agent_code.our_agent.escape_planner import survival_actions
from agent_code.our_agent.features import ACTIONS, state_to_features
from agent_code.our_agent.q_linear import LinearQ
from experiments.analyze_task4_guard import decode_state
from experiments.average_linear_symmetry import project_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=ROOT / "experiments/task4_scoring_audit/states.jsonl")
    parser.add_argument("--reference", type=Path, default=ROOT / "agent_code/our_agent/weights/q_linear_safety_v2_symavg.pkl")
    args = parser.parse_args()
    weights_dir = ROOT / "agent_code/our_agent/weights"
    paths = {"reference": args.reference}
    for arm in ("off", "on"):
        for episode in (50, 100):
            paths[f"{arm}_ep{episode}"] = (
                weights_dir / f"candidates/linear_task4_coin_tie_{arm}_seed2/ep_{episode:05d}.pkl"
            )
    raw = {name: LinearQ.load(path).W for name, path in paths.items()}
    matrices = {"reference": raw["reference"]}
    weight_stats = {}
    for name, weights in raw.items():
        projected = project_weights(weights)
        delta = weights - raw["reference"]
        asymmetry = weights - projected
        weight_stats[name] = {
            "path": str(paths[name].resolve()),
            "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "drift_norm": float(np.linalg.norm(delta)),
            "asymmetry_norm": float(np.linalg.norm(asymmetry)),
        }
        if name != "reference":
            matrices[name] = weights
            matrices[name + "_projected"] = projected

    rows = defaultdict(list)
    seen = set()
    records = 0
    with args.states.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            state = decode_state(record["state"])
            key = (record["seed"], state["round"], state["step"])
            if key in seen:
                raise ValueError(f"Duplicate recorded decision: {key}")
            seen.add(key)
            records += 1
            coin_ids, _ = safe_coin_actions(state)
            if not coin_ids:
                continue
            phi = state_to_features(state)
            mask = survival_actions(state, bomb_collision_guard=True, escape_collision_guard=True)
            allowed = LinearQ.available_actions(phi, mask)
            coin_ids = [a for a in coin_ids if allowed[a]]
            if not coin_ids:
                continue
            for name, weights in matrices.items():
                q = weights @ phi
                best = np.flatnonzero(allowed & (q == q[allowed].max()))
                probs = np.zeros(len(ACTIONS))
                probs[best] = 1.0 / len(best)
                rows[name].append({
                    "seed": record["seed"],
                    "coin_probability": float(probs[coin_ids].sum()),
                    "wait_probability": float(probs[4]),
                    "bomb_probability": float(probs[5]),
                })
    if not rows:
        raise ValueError("No screened coin opportunities in saved states")

    def summarize(items):
        return {"states": len(items), **{
            key: sum(item[key] for item in items) / len(items)
            for key in ("coin_probability", "wait_probability", "bomb_probability")
        }}

    policies = {}
    for name, items in rows.items():
        by_seed = defaultdict(list)
        for item in items:
            by_seed[item["seed"]].append(item)
        policies[name] = {
            **summarize(items),
            "by_seed": {str(seed): summarize(group) for seed, group in sorted(by_seed.items())},
            "strict_coin_preference_losses_vs_reference": sum(
                base["coin_probability"] == 1 and item["coin_probability"] == 0
                for base, item in zip(rows["reference"], items)),
            "strict_coin_preference_gains_vs_reference": sum(
                base["coin_probability"] == 0 and item["coin_probability"] == 1
                for base, item in zip(rows["reference"], items)),
        }
    print(json.dumps({
        "states_path": str(args.states.resolve()),
        "states_sha256": hashlib.sha256(args.states.read_bytes()).hexdigest(),
        "recorded_decisions": records,
        "weights": weight_stats,
        "policies": policies,
        "limits": "Fixed historical trajectory sample, not new games or a win-rate estimate. Uses current features and both collision guards, exact greedy ties, no exploration or stuck-action override. Opportunities are screened shortest paths to visible coins, not proof of the best strategic action. Projected weights exist only in memory; projected-policy recovery would support a symmetry contribution on this sample, not establish game-performance recovery. Decisions within a game are correlated; inspect by_seed for consistency.",
    }, indent=2))


if __name__ == "__main__":
    main()
