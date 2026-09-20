"""Summarize the completed guard comparison and inspect its self-kill traces."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_code.our_agent.escape_planner import escape_route
from agent_code.our_agent.features import DIRECTION_VECTORS


def decode_state(raw):
    state = dict(raw)
    state["field"] = np.asarray(raw["field"])
    state["explosion_map"] = np.asarray(raw["explosion_map"])
    state["self"] = (*raw["self"][:3], tuple(raw["self"][3]))
    state["others"] = [(*a[:3], tuple(a[3])) for a in raw["others"]]
    state["bombs"] = [(tuple(p), t) for p, t in raw["bombs"]]
    state["coins"] = list(map(tuple, raw["coins"]))
    return state


def diagnose(death):
    history = death["history"]
    states = [decode_state(h["state"]) for h in history]
    interceptions = []
    risky_choices = []
    new_bombs_when_routes_lost = []
    placed_at = None
    for i, (h, state) in enumerate(zip(history, states)):
        pos = state["self"][3]
        if i + 1 < len(states):
            after = states[i + 1]
            if h["action"] == "BOMB" and pos in {p for p, _ in after["bombs"]}:
                placed_at = state["step"]
            if h["action"] in DIRECTION_VECTORS:
                dx, dy = DIRECTION_VECTORS[h["action"]]
                destination = (pos[0] + dx, pos[1] + dy)
                initially_open = (state["field"][destination] == 0
                                  and destination not in {a[3] for a in state["others"]}
                                  and destination not in {p for p, _ in state["bombs"]})
                if (initially_open and after["self"][3] == pos
                        and destination in {a[3] for a in after["others"]}):
                    interceptions.append(state["step"])
            if any(h["mask"]) and not any(history[i + 1]["mask"]):
                added = {p for p, _ in after["bombs"]} - {p for p, _ in state["bombs"]}
                if h["action"] == "BOMB":
                    added.discard(pos)
                if added:
                    new_bombs_when_routes_lost.append(state["step"])
        # The placement check promises that some conservative route exists.
        # Check whether later actions follow such a route while alternatives
        # remain. This is a diagnostic association, not a causal counterfactual.
        if placed_at is not None and state["step"] > placed_at and h["action"] != "BOMB":
            blocked = {a[3] for a in state["others"]}
            def robust(action):
                return escape_route(state, pos, blocked=blocked, first_action=action,
                                    deadline_margin=1, avoid_opponent_collisions=True)[0]
            if not robust(h["action"]):
                alternatives = [a for a in [*DIRECTION_VECTORS, "WAIT"] if robust(a)]
                if alternatives:
                    risky_choices.append({"step": state["step"], "chosen": h["action"],
                                          "alternatives": alternatives})
    return {
        "seed": death["seed"], "death_step": death["step"],
        "own_bomb_placement_step": placed_at,
        "observed_interceptions": interceptions,
        "riskier_escape_choices_with_alternatives": risky_choices,
        "new_enemy_bombs_when_routes_lost": new_bombs_when_routes_lost,
        "final_mask_empty": not any(history[-1]["mask"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder
    protocol = json.loads((folder / "protocol.json").read_text())
    rows = {}
    for arm in ["off", "on"]:
        with (folder / f"guard_{arm}.csv").open() as f:
            rows[arm] = list(csv.DictReader(f))
        assert len(rows[arm]) == protocol["rounds_per_arm"], "Comparison is incomplete"
    assert [r["seed"] for r in rows["off"]] == [r["seed"] for r in rows["on"]]
    n = len(rows["on"])
    rng = np.random.default_rng(20260920)
    resamples = rng.integers(n, size=(10000, n))
    metrics = {}
    for metric in ["score", "coins", "kills", "self_kill", "win", "tie", "loss", "score_margin"]:
        off, on = [np.asarray([float(r[metric]) for r in rows[arm]]) for arm in ["off", "on"]]
        differences = on - off
        metrics[metric] = {"off": float(off.mean()), "on": float(on.mean()),
                           "on_minus_off": float(differences.mean()),
                           "paired_bootstrap_95_interval": np.quantile(
                               differences[resamples].mean(axis=1), [0.025, 0.975]).tolist()}
    off = np.asarray([float(r["win"]) + float(r["tie"]) for r in rows["off"]])
    on = np.asarray([float(r["win"]) + float(r["tie"]) for r in rows["on"]])
    metrics["win_or_tie"] = {"off": float(off.mean()), "on": float(on.mean()),
                             "on_minus_off": float((on-off).mean()),
                             "paired_bootstrap_95_interval": np.quantile(
                                 (on-off)[resamples].mean(axis=1), [0.025, 0.975]).tolist()}
    deaths = [json.loads(line) for line in (folder / "self_kills.jsonl").read_text().splitlines()]
    diagnosis = [diagnose(d) for d in deaths if d["guard"]]
    assert len(diagnosis) == sum(int(r["self_kill"]) for r in rows["on"])
    summary = {"guard_kind": protocol.get("guard_kind", "bomb"),
               "rounds_per_arm": n, "metrics": metrics,
               "interval_method": "10000 paired world-seed bootstrap resamples; exploratory percentile 95% intervals. Opponent randomness remains unseeded; no multiple-comparison correction.",
               "guard_on_deaths": len(diagnosis),
               "diagnostic_counts_overlap": {
                   "observed_interception": sum(bool(d["observed_interceptions"]) for d in diagnosis),
                   "riskier_escape_choice_with_alternative": sum(bool(d["riskier_escape_choices_with_alternatives"]) for d in diagnosis),
                   "new_enemy_bomb_when_routes_lost": sum(bool(d["new_enemy_bombs_when_routes_lost"]) for d in diagnosis),
                   "final_mask_empty": sum(d["final_mask_empty"] for d in diagnosis)},
               "diagnostic_limit": "Last 10 decisions per self-kill; categories overlap and are observed associations, not proven causes."}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (folder / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
