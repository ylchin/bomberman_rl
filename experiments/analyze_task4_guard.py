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
from agent_code.our_agent.features import ACTIONS, DIRECTION_VECTORS, state_to_features
from agent_code.our_agent.q_linear import LinearQ


def metric_fields(rows):
    fields = ["score", "coins", "kills", "self_kill", "win", "tie", "loss", "score_margin"]
    # Old comparisons do not contain overall survival or fallback-use counts.
    # Never infer survival from the absence of a self-kill.
    for key in ["survived", "died", "steps_survived", "fallback_steps", "optimistic_fallback_steps", "coin_preference_steps"]:
        if rows and all(row.get(key, "") != "" for row in rows):
            fields.append(key)
    return fields


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
    fallback_steps = []
    bomb_requests_during_fallback = []
    bomb_placements_during_fallback = []
    actions_outside_nonempty_screen = []
    screens = [LinearQ.legal_actions(state_to_features(s)) & np.asarray(h["mask"], dtype=bool)
               for h, s in zip(history, states)]
    effective_screens = [
        LinearQ.legal_actions(state_to_features(s)) & np.asarray(h.get("effective_mask", h["mask"]), dtype=bool)
        for h, s in zip(history, states)
    ]
    optimistic_fallback_steps = []
    actions_outside_effective_screen = []
    placed_at = None
    for i, (h, state) in enumerate(zip(history, states)):
        pos = state["self"][3]
        fallback = not screens[i].any()
        if fallback:
            fallback_steps.append(state["step"])
            if effective_screens[i].any():
                optimistic_fallback_steps.append(state["step"])
            if h["action"] == "BOMB":
                bomb_requests_during_fallback.append(state["step"])
        elif not screens[i][ACTIONS.index(h["action"])]:
            actions_outside_nonempty_screen.append(state["step"])
        if effective_screens[i].any() and not effective_screens[i][ACTIONS.index(h["action"])]:
            actions_outside_effective_screen.append(state["step"])
        after = states[i + 1] if i + 1 < len(states) else (
            decode_state(death["final_state"]) if death.get("final_state") is not None else None
        )
        if after is not None:
            placed = ("BOMB_DROPPED" in h["events"] if "events" in h else
                      pos not in {p for p, _ in state["bombs"]} and pos in {p for p, _ in after["bombs"]})
            if h["action"] == "BOMB" and placed:
                placed_at = state["step"]
                if fallback:
                    bomb_placements_during_fallback.append(state["step"])
            if h["action"] in DIRECTION_VECTORS:
                dx, dy = DIRECTION_VECTORS[h["action"]]
                destination = (pos[0] + dx, pos[1] + dy)
                initially_open = (state["field"][destination] == 0
                                  and destination not in {a[3] for a in state["others"]}
                                  and destination not in {p for p, _ in state["bombs"]})
                if (initially_open and after["self"][3] == pos
                        and destination in {a[3] for a in after["others"]}):
                    interceptions.append(state["step"])
            if i + 1 < len(states) and screens[i].any() and not screens[i + 1].any():
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
        "fallback_steps": fallback_steps,
        "bomb_requests_during_fallback": bomb_requests_during_fallback,
        "confirmed_bomb_placements_during_fallback": bomb_placements_during_fallback,
        "actions_outside_nonempty_screen": actions_outside_nonempty_screen,
        "optimistic_fallback_steps": optimistic_fallback_steps,
        "actions_outside_effective_screen": actions_outside_effective_screen,
        "final_mask_empty": not bool(screens[-1].any()),
    }


def summarize_single_arm(folder, protocol):
    arm = protocol["single_arm"]
    with (folder / f"guard_{arm}.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != protocol["rounds_per_arm"]:
        raise ValueError("Diagnostic run is incomplete")
    deaths = [json.loads(line) for line in (folder / "self_kills.jsonl").read_text().splitlines()]
    diagnosis = [diagnose(d) for d in deaths if d["guard"] == (arm == "on")]
    assert len(diagnosis) == sum(int(r["self_kill"]) for r in rows)
    fields = ["observed_interceptions", "riskier_escape_choices_with_alternatives",
              "new_enemy_bombs_when_routes_lost", "fallback_steps",
              "bomb_requests_during_fallback", "confirmed_bomb_placements_during_fallback",
              "actions_outside_nonempty_screen", "optimistic_fallback_steps",
              "actions_outside_effective_screen", "final_mask_empty"]
    summary = {
        "checkpoint": protocol["checkpoint"], "rounds": len(rows),
        "guard_kind": protocol["guard_kind"], "single_arm": arm,
        "metrics": {key: sum(float(r[key]) for r in rows) / len(rows)
                    for key in metric_fields(rows)},
        "self_kill_count": len(diagnosis),
        "deaths_with_each_pattern_overlap": {key: sum(bool(d[key]) for d in diagnosis) for key in fields},
        "examples": diagnosis[:10],
        "limits": "Fresh diagnostic games with unseeded opponents, not a replay of previous deaths or an A/B comparison. Last 10 decisions per death; patterns overlap and do not establish causation. Empty screens near death may be consequences of an earlier trap. Final movement outcomes are captured, but new-enemy-bomb route-loss checks require another decision state.",
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (folder / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    return summary


def summarize_comparison(folder, protocol):
    rows = {}
    for arm in ["off", "on"]:
        with (folder / f"guard_{arm}.csv").open() as f:
            rows[arm] = list(csv.DictReader(f))
        if len(rows[arm]) != protocol["rounds_per_arm"]:
            raise ValueError("Comparison is incomplete")
    expected_seeds = list(range(protocol["seed_start"], protocol["seed_start"] + protocol["rounds_per_arm"]))
    for arm in ("off", "on"):
        if [int(r["seed"]) for r in rows[arm]] != expected_seeds:
            raise ValueError("Comparison seeds are missing, duplicated or out of order")
    n = len(rows["on"])
    if not n:
        raise ValueError("Comparison has no rounds")
    rng = np.random.default_rng(20260920)
    resamples = rng.integers(n, size=(10000, n))
    metrics = {}
    fields = metric_fields(rows["off"] + rows["on"])
    comparing_checkpoints = protocol.get("comparison_kind") == "checkpoint"
    if comparing_checkpoints:
        fields = ["win"] + [key for key in fields if key != "win"]
    for metric in fields:
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
    for arm in ("off", "on"):
        if sum(d["guard"] == (arm == "on") for d in deaths) != sum(int(r["self_kill"]) for r in rows[arm]):
            raise ValueError(f"Self-kill traces and {arm} results disagree")
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
                   "optimistic_fallback_used": sum(bool(d["optimistic_fallback_steps"]) for d in diagnosis),
                   "actions_outside_effective_screen": sum(bool(d["actions_outside_effective_screen"]) for d in diagnosis),
                   "final_mask_empty": sum(d["final_mask_empty"] for d in diagnosis)},
               "diagnostic_limit": "Last 10 decisions per self-kill; categories overlap and are observed associations, not proven causes."}
    if comparing_checkpoints:
        summary.update(
            comparison_kind="checkpoint", primary_metric="win",
            checkpoints=protocol["checkpoints"], checkpoint_hashes=protocol["checkpoint_hashes"],
            arm_settings={"baseline": protocol["arm_settings"]["off"],
                          "candidate": protocol["arm_settings"]["on"]},
            baseline_self_kill_count=sum(int(r["self_kill"]) for r in rows["off"]),
            candidate_self_kill_count=summary.pop("guard_on_deaths"),
            diagnostic_arm="candidate",
        )
        summary["metrics"] = {
            key: {"baseline": result["off"], "candidate": result["on"],
                  "candidate_minus_baseline": result["on_minus_off"],
                  "paired_bootstrap_95_interval": result["paired_bootstrap_95_interval"]}
            for key, result in metrics.items()
        }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (folder / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.folder / "protocol.json").read_text())
    summary = (summarize_single_arm(args.folder, protocol) if protocol.get("single_arm")
               else summarize_comparison(args.folder, protocol))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
