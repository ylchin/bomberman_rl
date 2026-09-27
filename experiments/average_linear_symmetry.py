"""Create a separate linear checkpoint averaged over all eight board symmetries.

This changes weights only. It does not retrain, modify features, or alter masks.
For feature vector phi and action a, the projected model represents
mean_g Q_original(transform_features(phi, g), transform_action(a, g)).
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_code.our_agent.features import ACTIONS, FEATURE_DIM
from agent_code.our_agent.q_linear import LinearQ
from agent_code.our_agent.symmetry import apply_to_action, apply_to_features, sym_transforms


def project_weights(weights):
    """Average equivalent action/feature entries; never mutate the input.

    Each group orbit contains the distinct entries visited by all eight ops.
    Each member occurs equally often in the full group average. Assigning one
    mean per orbit also makes equivalent weights bit-identical.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(ACTIONS), FEATURE_DIM) or not np.isfinite(weights).all():
        raise ValueError("Expected finite weights with the current action/feature dimensions")
    permutations = []
    for op in sym_transforms():
        action_perm = [apply_to_action(a, op) for a in range(len(ACTIONS))]
        # Transforming feature labels yields new-index -> old-index. Invert it
        # to find the transformed coordinate of each original feature.
        feature_perm = np.argsort(apply_to_features(np.arange(FEATURE_DIM), op))
        permutations.append((action_perm, feature_perm))
    result = np.empty_like(weights)
    visited = np.zeros(weights.shape, dtype=bool)
    for action in range(len(ACTIONS)):
        for feature in range(FEATURE_DIM):
            if visited[action, feature]:
                continue
            orbit = sorted({(ap[action], int(fp[feature])) for ap, fp in permutations})
            mean = float(np.mean([weights[a, f] for a, f in orbit]))
            for a, f in orbit:
                result[a, f] = mean
                visited[a, f] = True
    return result


def create_checkpoint(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if source == output:
        raise ValueError("Output must differ from the source checkpoint")
    if output.exists() or report_path.exists():
        raise FileExistsError("Output or report already exists; choose a new filename")
    original_bytes = source.read_bytes()
    model = LinearQ.load(source)
    projected = project_weights(model.W)

    # Check the defining averaging identity before creating any output.
    rng = np.random.default_rng(0)
    max_error = 0.0
    for phi in rng.normal(size=(16, FEATURE_DIM)):
        ensemble = []
        for op in sym_transforms():
            mapped_actions = [apply_to_action(a, op) for a in range(len(ACTIONS))]
            ensemble.append(model.q_values(apply_to_features(phi, op))[mapped_actions])
        expected = np.mean(ensemble, axis=0)
        actual = projected @ phi
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        pickle.dump(dict(W=projected, feature_dim=FEATURE_DIM,
                         n_actions=len(ACTIONS), actions=ACTIONS), stream)
    np.testing.assert_array_equal(LinearQ.load(output).W, projected)
    if source.read_bytes() != original_bytes:
        raise RuntimeError("Source checkpoint changed during projection")
    report = {
        "source_checkpoint": str(source), "output_checkpoint": str(output),
        "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / "agent_code/our_agent/symmetry.py",
                                          ROOT / "agent_code/our_agent/features.py",
                                          ROOT / "agent_code/our_agent/q_linear.py"]},
        "max_ensemble_identity_error": max_error,
        "trained": False,
        "limitations": "Averages feature-permuted Q values, not raw-board predictions. Feature-extraction tie-breaking may still cause raw-board asymmetry. Floating-point arithmetic can affect exact ties. No game-performance improvement is established by this transformation.",
    }
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "agent_code/our_agent/weights/candidates/linear_model1_safety_v2_seed1/ep_04000.pkl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_checkpoint(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
