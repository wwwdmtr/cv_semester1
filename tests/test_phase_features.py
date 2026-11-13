"""Проверки фиксированного 42D-рецепта и изоляции внешних людей."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_phase_features as evaluation
import rehab26_phase_features as phase
from rehab26_features import (
    FEATURE_NAMES as BASE_NAMES, FEATURE_VERSION as BASE_VERSION,
    extract_features as extract_global_features,
)


def pose_sequence(frames=10, seed=3):
    result = np.random.default_rng(seed).normal(size=(frames, 33, 4))
    result[:, :, 3] = 0.9
    return result


def synthetic_data():
    records, features = [], []
    for person_index, person in enumerate("ABCDE"):
        for label in range(3):
            for camera_index, camera in enumerate(("cam1", "cam2", "cam3")):
                poses = pose_sequence(seed=100 * person_index + 10 * label + camera_index)
                records.append({
                    "person": person, "physical_id": f"{person}:{label}", "label": label,
                    "camera": camera, "orientation": person_index * 10.0,
                    "view_angle": 45.0 * camera_index - 10.0 * person_index, "poses": poses,
                })
                features.append(extract_global_features(poses))
    return records, np.stack(features), np.array([row["label"] for row in records])


class PhaseFeatureTests(unittest.TestCase):
    def test_seven_numeric_differences_and_fixed_signal_order(self):
        # Bottom rows 8,9; edges 0,1,8,9. Expected differences are
        # independently calculable for constant, linear, squared signals.
        index = np.arange(10, dtype=float)
        signals = np.column_stack([
            10 - index, 10 - index, np.full(10, 3.0), index,
            index ** 2, -index, np.zeros(10),
        ])
        poses = pose_sequence()
        with patch.object(phase, "extract_signals", return_value=signals):
            result = phase.extract_features(poses)
        np.testing.assert_allclose(result[35:], [-4, -4, 0, 4, 36, -4, 0], atol=1e-12)
        np.testing.assert_array_equal(result[:35], extract_global_features(poses))

    def test_metadata_and_exact_baseline_prefix_for_multiple_lengths(self):
        self.assertEqual(len(phase.FEATURE_NAMES), 42)
        self.assertEqual(len(set(phase.FEATURE_NAMES)), 42)
        self.assertEqual(phase.FEATURE_NAMES[:35], BASE_NAMES)
        self.assertNotEqual(phase.FEATURE_VERSION, BASE_VERSION)
        self.assertEqual(phase.FEATURE_METADATA["dimension"], 42)
        self.assertFalse(phase.FEATURE_METADATA["visibility_used_in_predictors"])
        self.assertFalse(phase.FEATURE_METADATA["labels_used_in_predictors"])
        for frames in (2, 3, 10, 23):
            poses = pose_sequence(frames)
            result = phase.extract_features(poses)
            self.assertEqual(result.shape, (42,))
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_array_equal(result[:35], extract_global_features(poses))

    def test_ties_include_all_frames_and_edge_union_is_unique(self):
        tied = np.ones((10, 7))
        bottom, edges = phase.phase_masks(tied)
        self.assertEqual(bottom.sum(), 10)
        self.assertEqual(np.flatnonzero(edges).tolist(), [0, 1, 8, 9])
        bottom, edges = phase.phase_masks(tied[:2])
        self.assertEqual(bottom.sum(), 2)
        self.assertEqual(edges.sum(), 2)
        self.assertTrue(np.all(bottom & edges))

    def test_source_is_not_modified_and_visibility_is_ignored(self):
        poses = pose_sequence()
        original = poses.copy()
        poses.flags.writeable = False
        result = phase.extract_features(poses)
        np.testing.assert_array_equal(poses, original)
        changed = original.copy()
        changed[:, :, 3] = np.nan  # As for baseline, visibility is not a predictor.
        np.testing.assert_array_equal(phase.extract_features(changed), result)

    def test_reversal_is_invariant_but_changing_edge_membership_can_matter(self):
        poses = pose_sequence(20)
        result = phase.extract_features(poses)
        np.testing.assert_allclose(result, phase.extract_features(poses[::-1]), atol=1e-12)
        permutation = np.arange(len(poses))
        permutation[[0, 9]] = permutation[[9, 0]]
        changed = phase.extract_features(poses[permutation])
        np.testing.assert_array_equal(result[:35], changed[:35])
        self.assertFalse(np.allclose(result[35:], changed[35:]))

    def test_invalid_inputs_are_rejected(self):
        for shape in ((1, 33, 4), (10, 25, 4), (10, 33, 3), (33, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                phase.extract_features(np.zeros(shape))
        poses = pose_sequence()
        poses[0, 25, 0] = np.nan
        with self.assertRaises(ValueError):
            phase.extract_features(poses)
        poses = pose_sequence()
        poses[:, 26, :3] = poses[:, 24, :3]
        with self.assertRaises(ValueError):
            phase.extract_features(poses)
        with self.assertRaises(ValueError):
            phase.phase_masks(np.zeros((10, 6)))


class PhaseEvaluationTests(unittest.TestCase):
    def test_failures_are_logged_and_experiment_aborts_without_filtering(self):
        records, x, _ = synthetic_data()
        records[1]["poses"][0, 25, 0] = np.nan
        records[2]["poses"] = np.zeros((1, 33, 4))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaises(ValueError):
                evaluation.prepare_phase_matrix(records, x, output)
            failures = json.loads((output / "phase_feature_failures.json").read_text())
            self.assertEqual([row["observation_index"] for row in failures], [1, 2])
            self.assertFalse((output / "phase_feature_matrix_metadata.json").exists())

    def test_inconsistent_baseline_prefix_aborts_and_is_logged(self):
        records, x, _ = synthetic_data()
        x[0, 0] += 0.001
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaises(ValueError):
                evaluation.prepare_phase_matrix(records, x, output)
            failures = json.loads((output / "phase_feature_failures.json").read_text())
            self.assertIn("baseline_prefix_not_exactly_equal", failures[0]["reason"])

    def test_synthetic_nested_run_saves_two_conditions_and_ten_distinct_bundles(self):
        records, x, y = synthetic_data()
        feature_metadata = {"version": BASE_VERSION, "names": BASE_NAMES, "details": {}}
        spec = {"id": "logreg_C1_unweighted", "family": "logreg", "C": 1.0, "balanced": False}
        original_fit = evaluation.fit_pipeline
        original_search = evaluation.search_candidates
        fit_people = []

        def tracked_search(features, labels, rows, specs):
            # Exactly four outer-training people are passed into the existing
            # nested search; its own person_folds splits them again.
            self.assertEqual(len({row["person"] for row in rows}), 4)
            return original_search(features, labels, rows, specs)

        def tracked_fit(candidate, features, labels, weights):
            fit_people.append((len(features), features.shape[1], weights.sum()))
            return original_fit(candidate, features, labels, weights)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output, models = Path(tmp) / "reports", Path(tmp) / "models"
            with patch.object(evaluation, "candidates", return_value=[spec]), \
                    patch.object(evaluation, "search_candidates", side_effect=tracked_search), \
                    patch.object(evaluation, "fit_pipeline", side_effect=tracked_fit):
                metrics = evaluation.run(records, x, y, output, models, feature_metadata, {})
            self.assertEqual(len(metrics["outer_folds"]), 10)
            self.assertEqual(len(fit_people), 10)
            self.assertTrue(all(count == 36 and np.isclose(weight, 12) for count, _, weight in fit_people))
            self.assertEqual({width for _, width, _ in fit_people}, {35, 42})
            search = json.loads((output / "inner_search.json").read_text())
            self.assertEqual(len(search), 10)
            self.assertTrue(all(row["selection_complete_before_outer_prediction"] for row in search))
            for row in search:
                self.assertNotIn(row["outer_test_person"], row["train_people"])
                self.assertEqual(len(row["search"][0]["folds"]), 4)
            bundles = [joblib.load(path) for path in models.rglob("*.joblib")]
            self.assertEqual(len(bundles), 10)
            for bundle in bundles:
                self.assertNotIn(bundle["held_out_person"], bundle["training_people"])
                expected = BASE_VERSION if bundle["method"] == "global35" else phase.FEATURE_VERSION
                self.assertEqual(bundle["feature_version"], expected)
                self.assertEqual(bundle["pipeline"].n_features_in_, bundle["input_feature_dimension"])
                self.assertFalse(bundle["deployment_replacement"])
            versions = {row["method"]: row["feature_version"] for row in metrics["summaries"]}
            self.assertEqual(versions, {"global35": BASE_VERSION, "phase_relative42": phase.FEATURE_VERSION})
            matrix = json.loads((output / "phase_feature_matrix_metadata.json").read_text())
            self.assertEqual(matrix["shape"], [45, 42])
            self.assertTrue(matrix["baseline_prefix_exact"])
            self.assertEqual(len(matrix["feature_matrix_sha256"]), 64)
            runtime = json.loads((output / "runtime.json").read_text())
            self.assertTrue(runtime["roundtrip_verified"])
            self.assertEqual(runtime["saved_final_bundles"], 0)
            with self.assertRaises(ValueError):
                evaluation.run(records, x, y, output, models, feature_metadata, {})


if __name__ == "__main__":
    unittest.main()
