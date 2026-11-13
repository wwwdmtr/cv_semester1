"""Контракт последовательности, абляция порядка и train-only CNN."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rehab26_temporal as temporal


def tiny_train():
    rng = np.random.default_rng(19)
    labels = np.repeat(np.arange(3), 3)
    values = rng.normal(scale=0.03, size=(9, 64, 7))
    values += labels[:, None, None] * 3.0
    return values, labels, np.full(9, 1 / 3)


class TemporalSequenceTests(unittest.TestCase):
    def test_resampling_preserves_known_linear_curve_and_endpoints(self):
        signals = np.arange(7)[None, :] + np.array([0., 2.])[:, None]
        result = temporal.resample_signal_array(signals)
        expected = np.arange(7)[None, :] + np.linspace(0, 2, 64)[:, None]
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result[[0, -1]], signals)

    def test_resampling_uses_same_geometry_and_does_not_mutate(self):
        poses = np.random.default_rng(8).normal(size=(13, 33, 4))
        original = poses.copy()
        poses.flags.writeable = False
        result = temporal.resample_signals(poses)
        expected = temporal.resample_signal_array(temporal.extract_signals(poses))
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(poses, original)
        poses2 = original.copy()
        poses2[:, :, 3] = np.nan
        np.testing.assert_array_equal(temporal.resample_signals(poses2), result)

    def test_shuffle_is_per_record_deterministic_and_preserves_frame_vectors(self):
        x = np.arange(2 * 64 * 7).reshape(2, 64, 7).astype(float)
        original = x.copy()
        keys = [("rep1", "cam1"), ("rep2", "cam2")]
        shuffled = temporal.transform_order(x, keys, shuffled=True)
        np.testing.assert_array_equal(temporal.transform_order(x, keys), x)
        np.testing.assert_array_equal(shuffled, temporal.transform_order(x, keys, shuffled=True))
        np.testing.assert_array_equal(shuffled[::-1], temporal.transform_order(x[::-1], keys[::-1], shuffled=True))
        np.testing.assert_array_equal(x, original)
        for i in range(2):
            order = np.argsort(shuffled[i, :, 0])
            np.testing.assert_array_equal(shuffled[i, order], x[i])
            self.assertFalse(np.array_equal(shuffled[i], x[i]))
        self.assertFalse(np.array_equal(shuffled[0, :, 0] % (64 * 7), shuffled[1, :, 0] % (64 * 7)))

    def test_same_physical_id_shares_permutation_between_camera_observations(self):
        first = np.arange(64 * 7).reshape(64, 7).astype(float)
        x = np.stack([first, first + 10_000])
        result = temporal.transform_order(x, ["physical_rep", "physical_rep"], shuffled=True)
        np.testing.assert_array_equal(result[0], result[1] - 10_000)

    def test_preparation_aborts_all_rows_without_filtering(self):
        records = [
            {"poses": np.zeros((1, 33, 4)), "physical_id": "bad1", "camera": "cam1"},
            {"poses": np.zeros((1, 33, 4)), "physical_id": "bad2", "camera": "cam2"},
        ]
        with self.assertRaises(ValueError) as context:
            temporal.prepare_sequences(records)
        self.assertEqual([r["index"] for r in context.exception.failures], [0, 1])
        for value in (np.zeros((1, 7)), np.zeros((5, 6)), np.full((2, 7), np.nan)):
            with self.assertRaises(ValueError):
                temporal.resample_signal_array(value)
        with self.assertRaises(ValueError):
            temporal.transform_order(np.zeros((1, 64, 7)), [])

    def test_balanced_effective_loss_mass_is_equal_per_class(self):
        y = np.array([0, 0, 0, 1, 2])
        w = np.array([1., 2., 3., 4., 5.])
        effective, class_weights = temporal.effective_training_weights(y, w, True)
        np.testing.assert_allclose([effective[y == k].sum() for k in range(3)], [1 / 3] * 3)
        np.testing.assert_allclose(class_weights, w.sum() / (3 * np.array([6., 4., 5.])))
        unweighted, classes = temporal.effective_training_weights(y, w, False)
        np.testing.assert_allclose(unweighted, w / w.sum())
        np.testing.assert_array_equal(classes, np.ones(3))


class TemporalTrainingTests(unittest.TestCase):
    def test_train_scaling_loss_learning_seed_ensemble_and_roundtrip(self):
        x, y, weights = tiny_train()
        with patch.object(temporal, "TRAIN_EPOCHS", 2):
            model = temporal.fit_temporal(x, y, weights, {"balanced": False})
            repeated = temporal.fit_temporal(x, y, weights, {"balanced": False})
        # Means/scales derive from train only, and querying a shifted holdout
        # cannot refit the scaler or affect subsequent original predictions.
        np.testing.assert_allclose(model.mean_, x.mean(axis=(0, 1)), atol=1e-13)
        np.testing.assert_allclose(model.scale_, x.std(axis=(0, 1)), atol=1e-13)
        self.assertEqual(model.seeds_, (0, 1, 2))
        self.assertEqual(model.training_config_["epochs"], 2)
        self.assertFalse(model.training_config_["validation_or_test_used"])
        self.assertTrue(all(row["final"] < row["initial"] for row in model.training_losses_))
        self.assertEqual(len(model.training_log_), 3)
        self.assertTrue(all(len(row["epoch_losses"]) == 2 for row in model.training_log_))
        scores = model.predict_proba(x)
        np.testing.assert_array_equal(scores, repeated.predict_proba(x))
        np.testing.assert_array_equal(scores, model.member_probabilities(x).mean(axis=0))
        self.assertEqual(scores.shape, (len(x), 3))
        self.assertEqual(scores.dtype, np.float64)
        np.testing.assert_allclose(scores.sum(axis=1), np.ones(len(x)), atol=1e-14)
        model.predict_proba(x + 1000)
        np.testing.assert_array_equal(scores, model.predict_proba(x))
        np.testing.assert_array_equal(model.classes_, [0, 1, 2])
        np.testing.assert_array_equal(model.predict(x), scores.argmax(axis=1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "temporal.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict_proba(x), scores)
            np.testing.assert_array_equal(restored.mean_, model.mean_)
            np.testing.assert_array_equal(restored.predict(x), model.predict(x))


if __name__ == "__main__":
    unittest.main()
