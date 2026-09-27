"""Record one frozen policy's coin choices and consecutive waits outside known danger."""

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_code.our_agent.features import ACTIONS, FEATURE_NAMES, danger_map, state_to_features
from agent_code.our_agent.escape_planner import survival_actions
from agent_code.our_agent.q_linear import LinearQ
from experiments.compare_coin_decisions import safe_coin_actions


def inspect_decision(state, action, model):
    safe = bool(danger_map(state)[state["self"][3]] == 0)
    row = {"step": state["step"], "position": list(state["self"][3]),
           "action": action, "outside_known_danger": safe,
           "visible_coins": len(state["coins"]), "coin_distance": None,
           "coin_actions": [], "greedy_actions": [], "greedy_coin_probability": None,
           "nonprogress_coin_choice": False, "chosen_minus_best_coin_q": None}
    if not safe:
        row["category"] = "in_known_danger"
        return row, None
    if not state["coins"]:
        row["category"] = "safe_no_visible_coins"
        return row, None
    coin_ids, distance = safe_coin_actions(state)
    row["coin_distance"] = distance
    if not coin_ids:
        row["category"] = "safe_no_clear_coin_route"
        return row, None
    phi = state_to_features(state)
    mask = survival_actions(state, bomb_collision_guard=True, escape_collision_guard=True)
    allowed = LinearQ.available_actions(phi, mask)
    coin_ids = [i for i in coin_ids if allowed[i]]
    if not coin_ids:
        row["category"] = "safe_coin_moves_screened_out"
        return row, None
    row["category"] = "safe_coin_opportunity"
    row["coin_actions"] = [ACTIONS[i] for i in coin_ids]
    q = model.q_values(phi)
    greedy = np.flatnonzero(allowed & (q == q[allowed].max()))
    row["greedy_actions"] = [ACTIONS[i] for i in greedy]
    row["greedy_coin_probability"] = sum(i in coin_ids for i in greedy) / len(greedy)
    chosen = ACTIONS.index(action)
    row["nonprogress_coin_choice"] = chosen not in coin_ids
    best_coin = max(coin_ids, key=lambda i: q[i])
    row["chosen_minus_best_coin_q"] = float(q[chosen] - q[best_coin])
    example = None
    if row["nonprogress_coin_choice"]:
        contributions = (model.W[chosen] - model.W[best_coin]) * phi
        indices = np.argsort(-np.abs(contributions))[:8]
        example = {"state": state, "best_coin_action": ACTIONS[best_coin],
                   "q_values": dict(zip(ACTIONS, q.tolist())),
                   "largest_ranking_contributions": [
                       {"feature": FEATURE_NAMES[i], "value": float(phi[i]),
                        "chosen_minus_coin_contribution": float(contributions[i])}
                       for i in indices if contributions[i] != 0]}
    return row, example


def wait_streaks(rows, minimum=3):
    """Consecutive WAIT decisions at the same tile, all outside known danger."""
    streaks, current = [], []

    def finish():
        if len(current) >= minimum:
            streaks.append({"seed": current[0]["seed"], "start_step": current[0]["step"],
                            "end_step": current[-1]["step"], "length": len(current),
                            "position": current[0]["position"],
                            "steps_with_coin_opportunity": sum(
                                r["category"] == "safe_coin_opportunity" for r in current),
                            "categories": dict(Counter(r["category"] for r in current))})
        current.clear()

    for row in rows:
        eligible = row["outside_known_danger"] and row["action"] == "WAIT"
        contiguous = not current or (row["seed"] == current[-1]["seed"]
                                     and row["step"] == current[-1]["step"] + 1
                                     and row["position"] == current[-1]["position"])
        if not eligible or not contiguous:
            finish()
        if eligible:
            current.append(row)
    finish()
    return streaks


def summarize(rows, minimum=3):
    opportunities = [r for r in rows if r["category"] == "safe_coin_opportunity"]
    missed = [r for r in opportunities if r["nonprogress_coin_choice"]]
    streaks = wait_streaks(rows, minimum)
    return {
        "decisions": len(rows), "categories": dict(Counter(r["category"] for r in rows)),
        "coin_opportunities": len(opportunities),
        "actual_nonprogress_coin_choices": len(missed),
        "actual_coin_progress_fraction": 1 - len(missed) / len(opportunities) if opportunities else None,
        "mean_greedy_coin_probability": sum(r["greedy_coin_probability"] for r in opportunities) / len(opportunities) if opportunities else None,
        "nonprogress_actions": dict(Counter(r["action"] for r in missed)),
        "safe_wait_decisions": sum(r["outside_known_danger"] and r["action"] == "WAIT" for r in rows),
        "safe_wait_categories": dict(Counter(r["category"] for r in rows
                                             if r["outside_known_danger"] and r["action"] == "WAIT")),
        "safe_wait_streaks": len(streaks),
        "safe_wait_streaks_with_coin_opportunities": sum(s["steps_with_coin_opportunity"] > 0 for s in streaks),
        "longest_safe_wait_streak": max((s["length"] for s in streaks), default=0),
        "longest_streak_examples": sorted(streaks, key=lambda s: s["length"], reverse=True)[:5],
        "longest_streaks_with_coin_opportunities": sorted(
            [s for s in streaks if s["steps_with_coin_opportunity"] > 0],
            key=lambda s: s["length"], reverse=True)[:5],
    }


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def build_summary(rows, game_rows, minimum):
    if not game_rows:
        raise ValueError("No completed games to summarize")
    return {
        "rounds": len(game_rows),
        "game_metrics": {key: sum(r[key] for r in game_rows) / len(game_rows)
                         for key in ("score", "coins", "kills", "self_kill", "win", "tie", "loss")},
        "overall": summarize(rows, minimum),
        "by_game": [{"seed": r["seed"], **summarize([s for s in rows if s["seed"] == r["seed"]], minimum)} for r in game_rows],
        "limits": "Correlated decisions from a small trajectory sample. Safe means outside currently known blast/fire paths, not guaranteed safety. Coin paths exclude all known danger and current occupancy; reachable refers only to visible coins under that conservative reference. Nonprogress includes strategic bombing or detours and is not proof of a mistake. Wait streaks without visible reachable coins are reported separately. Greedy ties exclude the stuck-action override; actual choices include it. No comparison or performance improvement is established.",
    }


def write_summary(output, summary):
    # Game positions can contain NumPy integers, including inside wait examples.
    # Use the same encoder as the saved decision rows for both file and stdout.
    serialized = json.dumps(summary, default=encode, indent=2) + "\n"
    (output / "summary.json").write_text(serialized)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_game"},
                     default=encode, indent=2))


def rebuild_summary(output):
    """Recover reporting from completed saved games; no model or games loaded."""
    protocol = json.loads((output / "protocol.json").read_text())
    game_rows = json.loads((output / "games.json").read_text())
    with (output / "decisions.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected = set(range(protocol["seed_start"], protocol["seed_start"] + protocol["rounds"]))
    if len(game_rows) != protocol["rounds"] or {r["seed"] for r in game_rows} != expected:
        raise ValueError("Saved game results do not cover all protocol rounds")
    if {r["seed"] for r in rows} != expected:
        raise ValueError("Saved decisions do not cover all protocol rounds")
    summary = build_summary(rows, game_rows, protocol["wait_streak_minimum"])
    write_summary(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("agent_code/our_agent/weights/q_linear_safety_v2_symavg.pkl"))
    parser.add_argument("--seed-start", type=int, default=32000)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--wait-streak", type=int, default=3)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--summarize-only", type=Path, metavar="EXISTING_DIR",
                        help="Rebuild summary.json from saved decisions and games; do not run games.")
    args = parser.parse_args()
    if args.summarize_only is not None:
        if args.out_dir is not None:
            parser.error("Use --summarize-only or --out-dir, not both")
        rebuild_summary(args.summarize_only.resolve())
        return
    if args.out_dir is None:
        parser.error("--out-dir is required when collecting games")
    if args.n_rounds < 1 or args.wait_streak < 2:
        parser.error("--n-rounds must be positive and --wait-streak must be at least 2")
    checkpoint = args.checkpoint.resolve()
    model = LinearQ.load(checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    os.environ.update(AGENT_MODEL="linear", AGENT_SURVIVAL_FILTER="1",
                      AGENT_BOMB_COLLISION_GUARD="1", AGENT_ESCAPE_COLLISION_GUARD="1",
                      AGENT_OPTIMISTIC_FALLBACK="0", AGENT_COIN_PREFERENCE="0",
                      AGENT_EVAL_CHECKPOINT=str(checkpoint))
    import main as game
    from evaluate import parse_result_file
    from agent_code.our_agent import callbacks, config
    config.MODEL = "linear"
    config.EVAL_WEIGHTS = [str(checkpoint)]
    config.TRAIN.update(survival_filter=True, bomb_collision_guard=True,
                        escape_collision_guard=True, optimistic_fallback=False, coin_preference=False)
    source_paths = [Path(__file__), ROOT / "experiments/compare_coin_decisions.py"] + [
        ROOT / "agent_code/our_agent" / name for name in
        ("callbacks.py", "config.py", "escape_planner.py", "features.py", "q_linear.py", "coin_navigation.py")]
    protocol = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                "seed_start": args.seed_start, "rounds": args.n_rounds,
                "wait_streak_minimum": args.wait_streak,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "policy": "Frozen linear checkpoint; both guards on; optimistic fallback off; no training.",
                "sampling": "Every own decision; analysis after each game. Opponents remain unseeded."}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    samples, all_rows, game_rows, examples = [], [], [], []
    original_act = callbacks.act

    def act(agent, state):
        action = original_act(agent, state)
        samples.append((state, action))
        return action

    callbacks.act = act
    try:
        with tempfile.TemporaryDirectory() as tmp, (output / "decisions.jsonl").open("w") as decisions, \
                (output / "states.jsonl").open("w") as snapshots:
            for index in range(args.n_rounds):
                seed = args.seed_start + index
                config.TRAIN["seed"] = seed
                samples.clear()
                stats = Path(tmp) / "stats.json"
                try:
                    game.main(["play", "--agents", "our_agent", "rule_based_agent", "rule_based_agent",
                               "rule_based_agent", "--scenario", "classic", "--seed", str(seed),
                               "--no-gui", "--n-rounds", "1", "--save-stats", str(stats)])
                finally:
                    for logger in logging.Logger.manager.loggerDict.values():
                        if isinstance(logger, logging.Logger):
                            for handler in list(logger.handlers):
                                handler.close()
                                logger.removeHandler(handler)
                game_rows.append(parse_result_file(stats, "our_agent", seed, index, 4))
                for state, action in samples:
                    row, example = inspect_decision(state, action, model)
                    row["seed"] = seed
                    all_rows.append(row)
                    decisions.write(json.dumps(row, default=encode) + "\n")
                    snapshots.write(json.dumps({"seed": seed, "state": state, "action": action}, default=encode) + "\n")
                    if example is not None:
                        examples.append({**row, **example})
                # Keep strongest ranking examples per action without retaining
                # every full state in memory. All snapshots remain on disk.
                examples = [e for action in ACTIONS for e in sorted(
                    [e for e in examples if e["action"] == action],
                    key=lambda e: e["chosen_minus_best_coin_q"], reverse=True)[:3]]
                decisions.flush()
                snapshots.flush()
                print(f"Inspected {index + 1}/{args.n_rounds} games; {len(all_rows)} decisions", flush=True)
    finally:
        callbacks.act = original_act
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    (output / "examples.json").write_text(json.dumps(examples, default=encode, indent=2) + "\n")
    (output / "games.json").write_text(json.dumps(game_rows, default=encode, indent=2) + "\n")
    write_summary(output, build_summary(all_rows, game_rows, args.wait_streak))


if __name__ == "__main__":
    main()
