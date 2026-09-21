"""Inspect nearby coin choices in saved states with no bombs or active fire."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_code.our_agent.features import FEATURE_NAMES, state_to_features
from agent_code.our_agent.q_linear import LinearQ
from experiments.analyze_task4_guard import decode_state
from experiments.audit_task4_scoring import encode, inspect_decision, summarize
from experiments.compare_coin_decisions import safe_coin_actions


def inspect_nearby(state, action, model, max_distance=6):
    """Return an exclusion reason or a decision with its ranking explanation."""
    if state["bombs"]:
        return "bomb_present", None, None
    if np.any(state["explosion_map"] > 0):
        return "active_fire_present", None, None
    coin_ids, distance = safe_coin_actions(state)
    if not coin_ids:
        return "no_clear_visible_coin_route", None, None
    if distance > max_distance:
        return "coin_beyond_distance_limit", None, None
    row, example = inspect_decision(state, action, model)
    if row["category"] != "safe_coin_opportunity":
        return "coin_moves_screened_out", None, None
    phi = state_to_features(state)
    directions = [FEATURE_NAMES[i].removeprefix("COIN_DIR_")
                  for i in range(7, 12) if phi[i] > 0.5]
    row["coin_feature_directions"] = directions
    row["feature_matches_clear_route"] = any(d in row["coin_actions"] for d in directions)
    pos = state["self"][3]
    row["nearest_opponent_manhattan"] = min(
        (abs(a[3][0] - pos[0]) + abs(a[3][1] - pos[1]) for a in state["others"]),
        default=None,
    )
    row["wait_strictly_above_all_coin_moves"] = (
        action == "WAIT" and row["chosen_minus_best_coin_q"] > 0
    )
    return "included", row, example


def nearby_summary(rows, minimum_waits=3):
    return {
        **summarize(rows, minimum_waits),
        "distance_counts": dict(Counter(r["coin_distance"] for r in rows)),
        "wait_strictly_above_all_coin_moves": sum(r["wait_strictly_above_all_coin_moves"] for r in rows),
        "by_coin_feature_alignment": {
            label: summarize([r for r in rows if r["feature_matches_clear_route"] == matches], minimum_waits)
            for label, matches in (("matches_clear_route", True), ("differs_from_clear_route", False))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Existing scoring audit with states.jsonl and protocol.json")
    parser.add_argument("--max-distance", type=int, default=6, help="Maximum clear walking distance, inclusive")
    parser.add_argument("--wait-streak", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, help="Optional relocated copy of the original checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_distance < 1 or args.wait_streak < 2:
        parser.error("--max-distance must be positive and --wait-streak at least 2")
    source = args.folder.resolve()
    protocol = json.loads((source / "protocol.json").read_text())
    checkpoint = (args.checkpoint or Path(protocol["checkpoint"])).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != protocol["checkpoint_sha256"]:
        parser.error("Checkpoint differs from the one that generated these states")
    model = LinearQ.load(checkpoint)
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, examples, counts, seeds = [], [], Counter(), set()
    with (source / "states.jsonl").open() as states_file, (output / "decisions.jsonl").open("w") as decisions:
        for line in states_file:
            if not line.strip():
                continue
            saved = json.loads(line)
            seeds.add(saved["seed"])
            state = decode_state(saved["state"])
            reason, row, example = inspect_nearby(state, saved["action"], model, args.max_distance)
            counts[reason] += 1
            if row is None:
                continue
            row["seed"] = saved["seed"]
            rows.append(row)
            decisions.write(json.dumps(row, default=encode) + "\n")
            if example is not None:
                examples.append({**row, **example})
    expected_seeds = set(range(protocol["seed_start"], protocol["seed_start"] + protocol["rounds"]))
    if seeds != expected_seeds:
        raise ValueError("Saved states do not cover the original audit's world seeds")
    # Favor WAIT examples for this investigation; retain other nonprogress
    # actions separately rather than labeling bombing or detours as mistakes.
    selected = []
    for is_wait in (True, False):
        selected.extend(sorted([e for e in examples if (e["action"] == "WAIT") == is_wait],
                               key=lambda e: e["chosen_minus_best_coin_q"], reverse=True)[:5])
    result = {
        "source_rounds": len(seeds), "source_decisions": sum(counts.values()),
        "filter": {"max_clear_walking_distance": args.max_distance,
                   "no_bombs_anywhere": True, "no_active_fire_anywhere": True,
                   "minimum_consecutive_waits": args.wait_streak},
        "filter_counts": dict(counts),
        "overall": nearby_summary(rows, args.wait_streak),
        "games_with_eligible_decisions": len({r["seed"] for r in rows}),
        "wait_examples": [{k: e[k] for k in (
            "seed", "step", "position", "coin_distance", "coin_actions", "greedy_actions",
            "coin_feature_directions", "feature_matches_clear_route", "nearest_opponent_manhattan",
            "chosen_minus_best_coin_q", "largest_ranking_contributions",
        )} for e in selected if e["action"] == "WAIT"],
        "limits": "Offline analysis of correlated saved decisions; no games or policy changes. Nearby means clear walking distance, not Manhattan distance. Opponent occupancy blocks coin paths, but future movement and bomb placements remain unknown. No bombs/fire does not guarantee safety. Feature agreement means the encoded coin direction matches any allowed shortest first move. Nonprogress is not necessarily a strategic mistake; six steps is an exploratory cutoff. Uses current feature/planner code with the original checkpoint, not a replay.",
    }
    sources = [Path(__file__), ROOT / "experiments/audit_task4_scoring.py",
               ROOT / "experiments/compare_coin_decisions.py",
               ROOT / "experiments/analyze_task4_guard.py"] + [
        ROOT / "agent_code/our_agent" / name for name in ("features.py", "escape_planner.py", "q_linear.py", "coin_navigation.py")]
    metadata = {"source_folder": str(source), "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash, "source_protocol": protocol,
                "filter": result["filter"],
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    for name, contents in (("summary.json", result), ("examples.json", selected), ("protocol.json", metadata)):
        (output / name).write_text(json.dumps(contents, default=encode, indent=2) + "\n")
    print(json.dumps(result, default=encode, indent=2))


if __name__ == "__main__":
    main()
