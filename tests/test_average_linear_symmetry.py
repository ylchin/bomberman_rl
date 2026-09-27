from pathlib import Path
import tempfile
import unittest

import numpy as np

from agent_code.our_agent.features import ACTIONS, FEATURE_DIM
from agent_code.our_agent.q_linear import LinearQ
from agent_code.our_agent.symmetry import apply_to_action, apply_to_features, sym_transforms
from experiments.average_linear_symmetry import create_checkpoint, project_weights


class AverageLinearSymmetryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.weights = self.rng.normal(size=(len(ACTIONS), FEATURE_DIM))

    def test_matches_explicit_eight_prediction_average(self):
        averaged = project_weights(self.weights)
        for phi in self.rng.normal(size=(12, FEATURE_DIM)):
            expected = np.zeros(len(ACTIONS))
            for op in sym_transforms():
                q = self.weights @ apply_to_features(phi, op)
                expected += q[[apply_to_action(a, op) for a in range(len(ACTIONS))]] / 8
            np.testing.assert_allclose(averaged @ phi, expected, atol=1e-12, rtol=1e-12)

    def test_projected_values_transform_with_actions(self):
        averaged = project_weights(self.weights)
        phi = self.rng.normal(size=FEATURE_DIM)
        for op in sym_transforms():
            q = averaged @ apply_to_features(phi, op)
            mapped = q[[apply_to_action(a, op) for a in range(len(ACTIONS))]]
            np.testing.assert_allclose(mapped, averaged @ phi, atol=1e-12, rtol=1e-12)

    def test_projection_is_idempotent_and_preserves_source(self):
        before = self.weights.copy()
        averaged = project_weights(self.weights)
        np.testing.assert_array_equal(self.weights, before)
        np.testing.assert_allclose(project_weights(averaged), averaged, atol=1e-12, rtol=1e-12)

    def test_checkpoint_roundtrip_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source.pkl", Path(folder) / "average.pkl"
            model = LinearQ()
            model.W = self.weights.copy()
            model.save(source)
            original = source.read_bytes()
            report = create_checkpoint(source, target)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(report["trained"])
            np.testing.assert_array_equal(LinearQ.load(target).W, project_weights(self.weights))
            with self.assertRaises(FileExistsError):
                create_checkpoint(source, target)
            with self.assertRaises(ValueError):
                create_checkpoint(source, source)


if __name__ == "__main__":
    unittest.main()
