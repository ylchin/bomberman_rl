"""Compare a policy flag or two checkpoints on paired world seeds.

Runs the normal main.py game loop in one process to avoid repeated import costs.
Each game creates fresh agents/worlds; opponent randomness remains unseeded,
as in evaluate.py. Only the selected flag or checkpoint differs between arms.
Rewards, game rules and opponents are unchanged.
"""

import argparse
from collections import deque
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def guard_settings(kind, enabled):
    """Explicit settings prevent inherited environment flags confounding arms."""
    return {
        "bomb_collision_guard": enabled if kind == "bomb" else True,
        "escape_collision_guard": enabled if kind == "escape" else kind in ("fallback", "coin", "checkpoint"),
        "optimistic_fallback": enabled if kind == "fallback" else False,
        "coin_preference": enabled if kind == "coin" else False,
    }


def arm_order(index, single_arm=None):
    return ((single_arm == "on",) if single_arm else
            ((False, True) if index % 2 == 0 else (True, False)))


def configure_arm(config, kind, enabled, seed, checkpoints):
    """Switch the imported config before fresh agents load their model."""
    checkpoint = checkpoints[enabled]
    config.MODEL = "linear"
    config.TRAIN.update(guard_settings(kind, enabled), survival_filter=True, seed=seed)
    config.EVAL_WEIGHTS = [str(checkpoint)]
    os.environ["AGENT_EVAL_CHECKPOINT"] = str(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=26000)
    parser.add_argument("--n-rounds", type=int, default=300)
    parser.add_argument("--guard-kind", choices=["bomb", "escape", "fallback", "coin", "checkpoint"], default="bomb",
                        help="Checkpoint compares two models with both guards on and both preferences off.")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "agent_code/our_agent/weights/candidates/linear_model1_safety_v2_seed1/ep_04000.pkl"))
    parser.add_argument("--candidate-checkpoint", type=Path,
                        help="Required for --guard-kind checkpoint; --checkpoint is the baseline.")
    parser.add_argument("--single-arm", choices=["off", "on"],
                        help="Capture just one configuration for diagnosis, not an A/B comparison.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n_rounds < 1:
        parser.error("--n-rounds must be positive")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    comparing_checkpoints = args.guard_kind == "checkpoint"
    if comparing_checkpoints != (args.candidate_checkpoint is not None):
        parser.error("--guard-kind checkpoint and --candidate-checkpoint must be supplied together")
    if comparing_checkpoints and args.single_arm:
        parser.error("Checkpoint comparison requires both arms; omit --single-arm")
    candidate = args.candidate_checkpoint.resolve() if comparing_checkpoints else checkpoint
    if not candidate.is_file():
        parser.error(f"Missing candidate checkpoint: {candidate}")
    checkpoints = {False: checkpoint, True: candidate}
    checkpoint_hashes = {arm: hashlib.sha256(path.read_bytes()).hexdigest()
                         for arm, path in checkpoints.items()}
    if comparing_checkpoints and checkpoint_hashes[False] == checkpoint_hashes[True]:
        parser.error("Baseline and candidate checkpoint files are identical")
    # Validate both schemas before creating output or starting hundreds of games.
    from agent_code.our_agent.q_linear import LinearQ
    for path in set(checkpoints.values()):
        LinearQ.load(path)
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    os.environ.update(AGENT_MODEL="linear", AGENT_SURVIVAL_FILTER="1",
                      AGENT_EVAL_CHECKPOINT=str(checkpoint))

    import numpy as np
    import environment
    import main as game
    from evaluate import CSV_COLUMNS, parse_result_file
    from agent_code.our_agent import callbacks, config
    from agent_code.our_agent.escape_planner import optimistic_fallback_actions, survival_actions
    from agent_code.our_agent.features import state_to_features
    from agent_code.our_agent.coin_navigation import COIN_PREFERENCE_MAX_DISTANCE

    history = deque(maxlen=10)
    deaths = []
    outcome = {}
    original_act = callbacks.act
    original_explosions = environment.BombeRLeWorld.evaluate_explosions

    def act(agent, state):
        action = original_act(agent, state)
        coin_preference_applied = bool(agent._coin_preference_applied)
        outcome["coin_preference_steps"] += int(coin_preference_applied)
        mask = survival_actions(
            state, bomb_collision_guard=config.TRAIN["bomb_collision_guard"],
            escape_collision_guard=config.TRAIN["escape_collision_guard"])
        legal = LinearQ.legal_actions(state_to_features(state))
        effective_mask = mask
        if not (mask & legal).any():
            outcome["fallback_steps"] += 1
        if config.TRAIN["optimistic_fallback"]:
            effective_mask = optimistic_fallback_actions(state, mask, legal)
            if not (mask & legal).any() and (effective_mask & legal).any():
                outcome["optimistic_fallback_steps"] += 1
        history.append({"state": state, "action": action,
                        "mask": mask, "effective_mask": effective_mask,
                        "pre_coin_action": agent._pre_coin_action,
                        "coin_preference_applied": coin_preference_applied})
        return action

    def explosions(world):
        me = next(a for a in world.agents if a.name == "our_agent")
        alive = not me.dead
        # Capture the final movement outcome before death removes the agent
        # from the observable state. This includes interceptions on the fatal turn.
        final_state = world.get_state_for_agent(me) if alive else None
        original_explosions(world)
        if alive and history:
            history[-1]["events"] = list(me.events)
        if alive and me.dead:
            outcome["died"] = 1
        if alive and me.dead and "KILLED_SELF" in me.events:
            deaths.append({"seed": seed, "guard": guard, "step": world.step,
                           "arm": arm_labels[guard], "checkpoint": str(checkpoints[guard]),
                           "history": list(history), "final_state": final_state})

    callbacks.act = act
    environment.BombeRLeWorld.evaluate_explosions = explosions

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    source_files = ["environment.py", "items.py", "settings.py", "evaluate.py",
                    "experiments/compare_task4_guard.py"] + [
        "agent_code/our_agent/" + name for name in
        ["callbacks.py", "config.py", "escape_planner.py", "features.py", "q_linear.py", "coin_navigation.py"]]
    arm_labels = {False: "baseline", True: "candidate"} if comparing_checkpoints else {False: "off", True: "on"}
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
        "seed_start": args.seed_start, "rounds_per_arm": args.n_rounds,
        "guard_kind": args.guard_kind,
        "comparison_kind": "checkpoint" if comparing_checkpoints else "flag",
        "arm_labels": {"off": arm_labels[False], "on": arm_labels[True]},
        "checkpoints": {arm_labels[a]: str(p) for a, p in checkpoints.items()},
        "checkpoint_hashes": {arm_labels[a]: h for a, h in checkpoint_hashes.items()},
        "primary_metric": "win" if comparing_checkpoints else None,
        "single_arm": args.single_arm,
        "coin_preference_max_distance": COIN_PREFERENCE_MAX_DISTANCE,
        "arm_settings": {arm: guard_settings(args.guard_kind, arm == "on")
                         for arm in ("off", "on")},
        "survival_definition": "survived=1 iff our_agent remains alive at round end; died includes all explosion deaths, not just self-kills",
        "protocol": "Fresh agents and worlds per game; unchanged unseeded opponents; original game settings; no training. Both-arm runs alternate arm order by seed; single-arm runs are diagnostics, not A/B comparisons.",
    }
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = {False: [], True: []}
    with (output / "guard_off.csv").open("w") as off, (output / "guard_on.csv").open("w") as on, \
            (output / "self_kills.jsonl").open("w") as traces, tempfile.TemporaryDirectory() as tmp:
        files = {False: off, True: on}
        columns = CSV_COLUMNS + ["survived", "died", "fallback_steps", "optimistic_fallback_steps", "coin_preference_steps"]
        writers = {key: csv.DictWriter(f, fieldnames=columns) for key, f in files.items()}
        for writer in writers.values():
            writer.writeheader()
        for index in range(args.n_rounds):
            seed = args.seed_start + index
            arms = arm_order(index, args.single_arm)
            for guard in arms:
                configure_arm(config, args.guard_kind, guard, seed, checkpoints)
                history.clear()
                deaths.clear()
                outcome.clear()
                outcome.update(died=0, fallback_steps=0, optimistic_fallback_steps=0, coin_preference_steps=0)
                stats = Path(tmp) / "stats.json"
                game.main(["play", "--agents", "our_agent", "rule_based_agent",
                           "rule_based_agent", "rule_based_agent", "--scenario", "classic",
                           "--seed", str(seed), "--no-gui", "--n-rounds", "1",
                           "--save-stats", str(stats)])
                row = parse_result_file(stats, "our_agent", seed, index, 4)
                row.update(outcome, survived=1 - outcome["died"])
                assert len(deaths) == row["self_kill"], "Death trace and score statistics disagree"
                assert row["self_kill"] <= row["died"], "Self-kill must also count as a death"
                writers[guard].writerow(row)
                files[guard].flush()
                rows[guard].append(row)
                for death in deaths:
                    traces.write(json.dumps(death, default=encode) + "\n")
                traces.flush()
                # main.py creates fresh file handlers on each setup. Close them
                # after the game so repeats do not accumulate handles/log writes.
                for logger in logging.Logger.manager.loggerDict.values():
                    if isinstance(logger, logging.Logger):
                        for handler in list(logger.handlers):
                            handler.close()
                            logger.removeHandler(handler)
            if (index + 1) % 10 == 0 or index + 1 == args.n_rounds:
                summaries = {arm_labels[g]: {k: round(sum(r[k] for r in rs) / len(rs), 4)
                                      for k in ["score", "self_kill", "survived", "win", "tie"]}
                             for g, rs in rows.items() if rs}
                unit = "games" if args.single_arm else "pairs"
                print(f"Completed {index + 1}/{args.n_rounds} {unit}: {summaries}", flush=True)
    for arm, path in checkpoints.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint_hashes[arm], "Checkpoint changed during evaluation"


if __name__ == "__main__":
    main()
