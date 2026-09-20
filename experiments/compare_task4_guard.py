"""Compare a safety guard on paired world seeds and capture self-kill histories.

Runs the normal main.py game loop in one process to avoid repeated import costs.
Each game creates fresh agents/worlds; opponent randomness remains unseeded,
as in evaluate.py. No policy, reward, game rule, or opponent is changed here.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=26000)
    parser.add_argument("--n-rounds", type=int, default=300)
    parser.add_argument("--guard-kind", choices=["bomb", "escape"], default="bomb",
                        help="For escape, keep the bomb guard enabled in both arms.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    checkpoint = ROOT / "agent_code/our_agent/weights/candidates/linear_model1_safety_v2_seed1/ep_04000.pkl"
    os.environ.update(AGENT_MODEL="linear", AGENT_SURVIVAL_FILTER="1",
                      AGENT_EVAL_CHECKPOINT=str(checkpoint))

    import numpy as np
    import environment
    import main as game
    from evaluate import CSV_COLUMNS, parse_result_file
    from agent_code.our_agent import callbacks, config
    from agent_code.our_agent.escape_planner import survival_actions

    history = deque(maxlen=10)
    deaths = []
    original_act = callbacks.act
    original_explosions = environment.BombeRLeWorld.evaluate_explosions

    def act(agent, state):
        action = original_act(agent, state)
        history.append({"state": state, "action": action,
                        "mask": survival_actions(
                            state, bomb_collision_guard=config.TRAIN["bomb_collision_guard"],
                            escape_collision_guard=config.TRAIN["escape_collision_guard"])})
        return action

    def explosions(world):
        me = next(a for a in world.agents if a.name == "our_agent")
        alive = not me.dead
        original_explosions(world)
        if alive and me.dead and "KILLED_SELF" in me.events:
            deaths.append({"seed": seed, "guard": guard, "step": world.step,
                           "history": list(history)})

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
        ["callbacks.py", "config.py", "escape_planner.py", "features.py", "q_linear.py"]]
    metadata = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
        "seed_start": args.seed_start, "rounds_per_arm": args.n_rounds,
        "guard_kind": args.guard_kind,
        "protocol": "Alternate guard off/on order by seed; fresh agents and worlds per game; unchanged unseeded opponents; original game settings; no training.",
    }
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = {False: [], True: []}
    with (output / "guard_off.csv").open("w") as off, (output / "guard_on.csv").open("w") as on, \
            (output / "self_kills.jsonl").open("w") as traces, tempfile.TemporaryDirectory() as tmp:
        files = {False: off, True: on}
        writers = {key: csv.DictWriter(f, fieldnames=CSV_COLUMNS) for key, f in files.items()}
        for writer in writers.values():
            writer.writeheader()
        for index in range(args.n_rounds):
            seed = args.seed_start + index
            for guard in ((False, True) if index % 2 == 0 else (True, False)):
                config.TRAIN["bomb_collision_guard"] = guard if args.guard_kind == "bomb" else True
                config.TRAIN["escape_collision_guard"] = guard if args.guard_kind == "escape" else False
                config.TRAIN["seed"] = seed
                history.clear()
                deaths.clear()
                stats = Path(tmp) / "stats.json"
                game.main(["play", "--agents", "our_agent", "rule_based_agent",
                           "rule_based_agent", "rule_based_agent", "--scenario", "classic",
                           "--seed", str(seed), "--no-gui", "--n-rounds", "1",
                           "--save-stats", str(stats)])
                row = parse_result_file(stats, "our_agent", seed, index, 4)
                assert len(deaths) == row["self_kill"], "Death trace and score statistics disagree"
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
                summaries = {str(g): {k: round(sum(r[k] for r in rs) / len(rs), 4)
                                      for k in ["score", "self_kill", "win", "tie"]}
                             for g, rs in rows.items()}
                print(f"Completed {index + 1}/{args.n_rounds} pairs: {summaries}", flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata["checkpoint_sha256"]


if __name__ == "__main__":
    main()
