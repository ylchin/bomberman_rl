"""Offline symmetry diagnosis on saved states; never starts games or training."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_code.our_agent.features import ACTIONS, FEATURE_NAMES, state_to_features
from agent_code.our_agent.escape_planner import survival_actions
from agent_code.our_agent.q_linear import LinearQ
from agent_code.our_agent.symmetry import apply_to_action, apply_to_features, apply_to_state, sym_transforms
from experiments.analyze_task4_guard import decode_state


def greedy_set(q, allowed):
    """Exact ties, matching LinearQ.act, without selecting a random winner."""
    return set(np.flatnonzero(allowed & (q == q[allowed].max())).tolist())


def allowed_actions(state, phi):
    mask = survival_actions(state, bomb_collision_guard=True, escape_collision_guard=True)
    return LinearQ.available_actions(phi, mask)


def inspect_state(state, models):
    phi = state_to_features(state)
    allowed = allowed_actions(state, phi)
    originals = {name: greedy_set(model.q_values(phi), allowed) for name, model in models.items()}
    result = {"feature_mismatches": [], "mask_mismatches": [], "comparisons": [],
              "orientation_actions": {name: Counter() for name in models}}
    for op in sym_transforms():
        transformed = apply_to_state(state, op)
        actual_phi = state_to_features(transformed)
        permuted_phi = apply_to_features(phi, op)
        actual_allowed = allowed_actions(transformed, actual_phi)
        # perm[a] is the transformed index corresponding to original action a.
        # Indexing a transformed action vector with perm maps it back.
        perm = np.asarray([apply_to_action(a, op) for a in range(len(ACTIONS))])
        changed = np.flatnonzero(~np.isclose(actual_phi, permuted_phi, rtol=0, atol=1e-6))
        same_mask = np.array_equal(actual_allowed[perm], allowed)
        if op != (False, 0):
            if len(changed):
                result["feature_mismatches"].append({
                    "op": op, "features": [FEATURE_NAMES[i] for i in changed]})
            if not same_mask:
                result["mask_mismatches"].append(op)
        for name, model in models.items():
            actual_q = model.q_values(actual_phi)
            raw_best = greedy_set(actual_q, actual_allowed)
            for action in raw_best:
                result["orientation_actions"][name][ACTIONS[action]] += 1.0 / len(raw_best)
            if op == (False, 0):
                continue
            mapped_best = greedy_set(actual_q[perm], actual_allowed[perm])
            # A pure feature permutation keeps the original allowed set after
            # mapping back. It isolates learned weights from BFS tie-breaking
            # and orientation-dependent feature extraction or action screening.
            feature_best = greedy_set(model.q_values(permuted_phi)[perm], allowed)
            original = originals[name]
            result["comparisons"].append({
                "model": name, "op": op,
                "raw_exact_match": mapped_best == original,
                "raw_disjoint": not bool(mapped_best & original),
                "feature_only_exact_match": feature_best == original,
                "feature_only_disjoint": not bool(feature_best & original),
                "same_feature_vector": len(changed) == 0,
                "same_action_mask": bool(same_mask),
                "original_greedy": [ACTIONS[a] for a in sorted(original)],
                "raw_greedy_mapped_back": [ACTIONS[a] for a in sorted(mapped_best)],
                "feature_only_greedy_mapped_back": [ACTIONS[a] for a in sorted(feature_best)],
                "changed_features": [FEATURE_NAMES[i] for i in changed],
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Output folder from compare_coin_decisions.py.")
    parser.add_argument("--out-dir", type=Path, help="Defaults to <folder>/symmetry; must be new.")
    args = parser.parse_args()
    folder = args.folder.resolve()
    output = args.out_dir.resolve() if args.out_dir else folder / "symmetry"
    protocol = json.loads((folder / "protocol.json").read_text())
    paths = {name: Path(path) for name, path in protocol["checkpoints"].items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    if hashes != protocol["checkpoint_sha256"]:
        parser.error("Checkpoint hashes differ from the state-capture run")
    models = {name: LinearQ.load(path) for name, path in paths.items()}
    samples = [json.loads(line) for line in (folder / "states.jsonl").read_text().splitlines() if line.strip()]
    if not samples:
        parser.error("No saved states found")
    output.mkdir(parents=True, exist_ok=False)
    comparisons = []
    feature_counts = Counter()
    mismatch_count = 0
    mask_count = 0
    orientation_actions = {name: Counter() for name in models}
    with (output / "comparisons.jsonl").open("w") as stream:
        for sample in samples:
            state = decode_state(sample["state"])
            result = inspect_state(state, models)
            mismatch_count += len(result["feature_mismatches"])
            mask_count += len(result["mask_mismatches"])
            for mismatch in result["feature_mismatches"]:
                feature_counts.update(mismatch["features"])
            for name in models:
                orientation_actions[name].update(result["orientation_actions"][name])
            for row in result["comparisons"]:
                row.update(driver=sample["driver"], seed=sample["seed"], step=state["step"])
                comparisons.append(row)
                stream.write(json.dumps(row) + "\n")

    def summarize(rows):
        n = len(rows)
        comparable = [r for r in rows if r["same_feature_vector"] and r["same_action_mask"]]
        return {
            "comparisons": n,
            **{key + "_rate": sum(r[key] for r in rows) / n if n else None
               for key in ["raw_exact_match", "raw_disjoint", "feature_only_exact_match", "feature_only_disjoint"]},
            "comparisons_without_feature_or_mask_changes": len(comparable),
            "raw_exact_match_rate_without_feature_or_mask_changes":
                sum(r["raw_exact_match"] for r in comparable) / len(comparable) if comparable else None,
        }

    summary = {
        "states": len(samples), "nonidentity_transforms_per_state": 7,
        "raw_feature_mismatches": mismatch_count, "raw_action_mask_mismatches": mask_count,
        "feature_mismatch_counts": dict(feature_counts.most_common()),
        "models": {name: summarize([r for r in comparisons if r["model"] == name]) for name in models},
        "by_driver": {driver: {name: summarize([r for r in comparisons if r["model"] == name and r["driver"] == driver])
                               for name in models} for driver in sorted({s["driver"] for s in samples})},
        "action_probabilities_over_all_8_orientations": {
            name: {action: counts[action] / (len(samples) * 8) for action in ACTIONS}
            for name, counts in orientation_actions.items()},
        "examples": [r for r in comparisons if not r["raw_exact_match"]][:4],
        "source_files_changed_since_capture": [p for p, h in protocol.get("source_sha256", {}).items()
                                               if not (ROOT / p).is_file() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h],
        "limits": "Offline, correlated states from two policies. Consistency is not a performance metric. Pure feature permutations can differ from freshly extracted features because of target/path tie-breaking; feature-only results isolate weight asymmetry on those inputs. The history-dependent stuck-action override is excluded. No random tie-breaking is sampled.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {"source_folder": str(folder), "checkpoint_sha256": hashes,
                  "states_sha256": hashlib.sha256((folder / "states.jsonl").read_bytes()).hexdigest(),
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [Path(__file__), ROOT / "agent_code/our_agent/symmetry.py",
                                              ROOT / "agent_code/our_agent/features.py",
                                              ROOT / "agent_code/our_agent/escape_planner.py",
                                              ROOT / "agent_code/our_agent/q_linear.py"]}}
    (output / "protocol.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
