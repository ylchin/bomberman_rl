"""Inspect fallback choices in saved self-kill traces without running games."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.analyze_task4_guard import decode_state
from agent_code.our_agent.escape_planner import escape_route
from agent_code.our_agent.features import ACTIONS, BOMB_TIMER, state_to_features
from agent_code.our_agent.q_linear import LinearQ


def action_routes(state, legal):
    """Compare current occupancy with an optimistic future-occupancy model.

    Both searches force the candidate action and use exact known-bomb timing.
    Only initially legal actions are considered. The optimistic search removes
    opponent occupancy for subsequent movement, but retains bombs, walls,
    crates and fire. This is a possibility bound, not an opponent prediction.
    """
    pos = state["self"][3]
    opponents = {a[3] for a in state["others"]}
    result = {}
    for action, allowed in zip(ACTIONS, legal):
        if not allowed:
            continue
        proposed = state
        options = {"first_action": action}
        if action == "BOMB":
            proposed = dict(state, bombs=[*state["bombs"], (pos, BOMB_TIMER)])
            options = {"placement_turn": True}
        result[action] = {
            "static_opponent_route": bool(escape_route(
                proposed, pos, blocked=opponents, **options,
            )[0]),
            "optimistic_route": bool(escape_route(
                proposed, pos, blocked=(), **options,
            )[0]),
        }
    return result


def inspect_death(death):
    decisions = []
    previous = None
    previous_legal = None
    for frame in death["history"]:
        state = decode_state(frame["state"])
        legal = LinearQ.legal_actions(state_to_features(state))
        screen = legal & np.asarray(frame["mask"], dtype=bool)
        if not screen.any():
            newly_opened = []
            if (previous is not None
                    and state["step"] == previous["step"] + 1
                    and state["self"][3] == previous["self"][3]):
                newly_opened = [ACTIONS[i] for i in range(4)
                                if legal[i] and not previous_legal[i]]
            routes = action_routes(state, legal)
            chosen = frame["action"]
            # Missing means the recorded choice was outside the current legal
            # action set; don't silently classify it as a failed route.
            chosen_route = routes.get(chosen, {}).get("optimistic_route")
            alternatives = [a for a, r in routes.items()
                            if a != chosen and r["optimistic_route"]]
            decisions.append({
                "step": state["step"],
                "position": list(state["self"][3]),
                "chosen": chosen,
                "events": frame.get("events", []),
                "newly_opened_moves_while_stationary": newly_opened,
                "routes_by_legal_action": routes,
                "chosen_has_optimistic_route": chosen_route,
                "alternatives_with_optimistic_route": alternatives,
                "missed_possible_escape": chosen_route is False and bool(alternatives),
                "missed_newly_opened_exit": chosen_route is False and bool(
                    set(newly_opened) & set(alternatives)
                ),
            })
        previous, previous_legal = state, legal
    return {"seed": death["seed"], "death_step": death["step"],
            "fallback_decisions": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--seed", type=int, help="Inspect one world seed; default: all saved deaths")
    parser.add_argument("--arm", choices=("on", "off"), default="on")
    args = parser.parse_args()
    with (args.folder / "self_kills.jsonl").open() as stream:
        deaths = [json.loads(line) for line in stream if line.strip()]
    selected = [d for d in deaths if d["guard"] == (args.arm == "on")
                and (args.seed is None or d["seed"] == args.seed)]
    if args.seed is not None and not selected:
        parser.error(f"No saved self-kill for seed {args.seed} in arm {args.arm}")
    results = [inspect_death(d) for d in selected]
    decisions = [f for d in results for f in d["fallback_decisions"]]
    print(json.dumps({
        "arm": args.arm,
        "saved_deaths_inspected": len(results),
        "fallback_decisions_inspected": len(decisions),
        "decisions_with_missed_possible_escape": sum(d["missed_possible_escape"] for d in decisions),
        "decisions_with_missed_newly_opened_exit": sum(d["missed_newly_opened_exit"] for d in decisions),
        "deaths": results,
        "limits": (
            "Saved self-kill traces only, not all fallback decisions or all deaths. "
            "Uses the current feature and escape-planner code with the recorded safety mask. "
            "Each decision uses only bombs and fire visible at that decision; BOMB candidates "
            "also include their hypothetical own bomb and its placement turn. "
            "Optimistic routes keep the first action legal but ignore subsequent opponent "
            "occupancy, interception and future bomb placements. They may be impossible "
            "against real opponents. Missed possible escape is a diagnostic flag, "
            "not proof an alternative action would survive. No games are replayed."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
