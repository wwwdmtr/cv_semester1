"""Изоляция людей, парность абляции и фиксированное объединение без fit сети."""

import contextlib
import csv
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
import evaluate_temporal_comparison as evaluation


class ToyProbabilityModel:
    """Детерминированный mock: синтетические данные содержат код категории."""

    classes_ = np.array([0, 1, 2])
    training_log_ = [{"seed": 0, "epoch_losses": [1., .8], "initial": 1., "final": .7}]

    def __init__(self, offset=0):
        self.offset = offset

    def predict_proba(self, x):
        values = x[:, 1] if x.ndim == 2 else x[:, 0, 1]
        labels = (values.astype(int) + self.offset) % 3
        result = np.full((len(x), 3), .1)
        result[np.arange(len(x)), labels] = .8
        return result

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)


def synthetic_data():
    records, features = [], []
    for person_index, person in enumerate("ABCDE"):
        for label in range(3):
            for camera in ("cam1", "cam2", "cam3"):
                records.append({"person": person, "physical_id": f"{person}:{label}",
                                "label": label, "camera": camera, "view_angle": 30.,
                                "poses": np.ones((2, 33, 4))})
                row = np.zeros(35)
                row[:2] = person_index, label
                features.append(row)
    x = np.stack(features)
    sequence = np.repeat(x[:, None, :7], 64, axis=1)
    return records, x, np.array([row["label"] for row in records]), sequence


def search_row(balanced, f1, fpr=.1, recall=.9):
    return {"candidate": {"id": f"t_{balanced}", "family": "temporal_cnn", "balanced": balanced},
            "mean_macro_f1": f1, "mean_false_feedback_rate": fpr,
            "mean_error_recall": recall, "guardrails_met": fpr <= .2 and recall >= .8}


class TemporalComparisonTests(unittest.TestCase):
    def test_fusion_is_fixed_probability_mean_and_uses_fitted_objects(self):
        _, x, _, sequence = synthetic_data()
        geometry, temporal = ToyProbabilityModel(), ToyProbabilityModel(offset=1)
        model = evaluation.FusionClassifier(geometry, temporal)
        self.assertIs(model.geometry_model, geometry)
        self.assertIs(model.temporal_model, temporal)
        expected = (geometry.predict_proba(x) + temporal.predict_proba(sequence)) / 2
        np.testing.assert_array_equal(model.predict_proba((x, sequence)), expected)
        np.testing.assert_array_equal(model.predict((x, sequence)), expected.argmax(axis=1))
        with self.assertRaises(ValueError):
            evaluation.FusionClassifier(geometry, temporal, .25)
        with self.assertRaises(ValueError):
            model.predict_proba((x[:-1], sequence))

    def test_shared_choice_is_symmetric_and_applies_guardrails_before_f1(self):
        ordered = [search_row(False, .9, .3, .9), search_row(True, .7, .1, .9)]
        shuffled = [search_row(False, .9, .3, .9), search_row(True, .5, .1, .9)]
        combined, chosen = evaluation.shared_temporal_selection(ordered, shuffled)
        self.assertTrue(chosen["balanced"])
        self.assertAlmostEqual(combined[1]["mean_macro_f1"], .6)
        _, reverse = evaluation.shared_temporal_selection(shuffled, ordered)
        self.assertEqual(chosen, reverse)
        with self.assertRaises(ValueError):
            evaluation.shared_temporal_selection(ordered, shuffled[:1])

    def test_sequence_errors_abort_and_are_logged_without_exclusion(self):
        records, _, _, _ = synthetic_data()
        calls = []

        def preparation(poses):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("synthetic_failure")
            return np.ones((64, 7))

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(evaluation.temporal, "resample_signals", side_effect=preparation):
                with self.assertRaises(ValueError):
                    evaluation.prepare_sequence_matrix(records, Path(tmp))
            self.assertEqual(len(calls), len(records))
            failures = json.loads((Path(tmp) / "temporal_feature_failures.json").read_text())
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["observation_index"], 1)
            self.assertFalse((Path(tmp) / "temporal_matrix_metadata.json").exists())

    def test_preparation_passes_physical_ids_and_proves_value_preservation(self):
        records, _, _, _ = synthetic_data()
        observations = np.arange(64 * 7).reshape(64, 7).astype(float)
        actual_transform = evaluation.temporal.transform_order
        observed_ids = []

        def transform(x, ids, **kwargs):
            observed_ids.extend(ids)
            return actual_transform(x, ids, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(evaluation.temporal, "resample_signals", return_value=observations), \
                    patch.object(evaluation.temporal, "transform_order", side_effect=transform):
                ordered, shuffled, _, metadata = evaluation.prepare_sequence_matrix(records, Path(tmp))
        self.assertEqual(observed_ids, [row["physical_id"] for row in records])
        np.testing.assert_array_equal(shuffled[0], shuffled[1])
        np.testing.assert_array_equal(np.sort(ordered, axis=1), np.sort(shuffled, axis=1))
        self.assertTrue(metadata["identical_sorted_values_per_channel"])
        self.assertEqual(metadata["observations_excluded"], 0)

    def test_nested_mock_run_excludes_people_reuses_components_and_roundtrips(self):
        records, x, y, sequence = synthetic_data()
        geometry_spec = {"id": "logreg_C1_unweighted", "family": "logreg", "C": 1., "balanced": False}
        temporal_spec = {"id": "temporal_cnn_unweighted", "family": "temporal_cnn", "balanced": False}
        fit_calls = []

        def fitting(spec, features, labels, weights):
            values = features if features.ndim == 2 else features[:, 0, :]
            people = sorted(set(values[:, 0].astype(int)))
            self.assertIn(len(people), (3, 4))
            np.testing.assert_array_equal(values[:, 1].astype(int), labels)
            self.assertAlmostEqual(weights.sum(), 3 * len(people))
            fit_calls.append((spec["family"], people))
            return ToyProbabilityModel()

        def temporal_fitting(features, labels, weights, spec):
            return fitting(spec, features, labels, weights)

        metadata = {"version": evaluation.BASE_FEATURE_VERSION, "names": [str(i) for i in range(35)]}
        sequence_metadata = {"version": evaluation.temporal.FEATURE_VERSION, "names": [], "details": {}}
        matrix_metadata = {"shape": list(sequence.shape), "observations_excluded": 0}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output, models = Path(tmp) / "reports", Path(tmp) / "models"
            with patch.object(evaluation, "geometry_candidates", return_value=[geometry_spec]), \
                    patch.object(evaluation, "temporal_candidates", return_value=[temporal_spec]), \
                    patch.object(evaluation, "prepare_sequence_matrix", return_value=(sequence, sequence.copy(), sequence_metadata, matrix_metadata)), \
                    patch.object(evaluation, "fit_pipeline", side_effect=fitting), \
                    patch.object(evaluation.temporal, "fit_temporal", side_effect=temporal_fitting):
                metrics = evaluation.run(records, x, y, output, models, metadata, {})
            self.assertEqual(len(fit_calls), 75)  # 5 * (4 inner + 1 outer) * 3 components
            self.assertEqual(len(metrics["outer_folds"]), 20)
            self.assertEqual({row["method"] for row in metrics["summaries"]}, set(evaluation.METHODS))
            searches = json.loads((output / "inner_search.json").read_text())
            for entry in searches:
                self.assertNotIn(entry["outer_test_person"], entry["train_people"])
                self.assertTrue(entry["selection_complete_before_outer_prediction"])
                self.assertEqual(entry["chosen"]["temporal_ordered"], entry["chosen"]["temporal_shuffled"])
                self.assertEqual(entry["fusion_alpha"], .5)
                self.assertFalse(entry["fusion_weight_tuned"])
                for search in entry["searches"].values():
                    for fold in search[0]["folds"]:
                        self.assertNotIn(fold["validation_person"], fold["train_people"])
                        self.assertNotIn(entry["outer_test_person"], fold["train_people"])
                        self.assertEqual(len(fold["train_people"]), 3)
            bundles = [joblib.load(path) for path in models.rglob("*.joblib")]
            self.assertEqual(len(bundles), 20)
            for bundle in bundles:
                self.assertNotIn(bundle["held_out_person"], bundle["training_people"])
                self.assertFalse(bundle["deployment_replacement"])
                self.assertEqual(bundle["feature_version"], evaluation.feature_versions()[bundle["method"]])
            with (output / "fold_predictions.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4 * len(records))
            self.assertEqual(len({(row["method"], row["physical_id"], row["camera"]) for row in rows}), len(rows))
            with (output / "inner_predictions.csv").open(newline="") as stream:
                inner_rows = list(csv.DictReader(stream))
            self.assertEqual(len(inner_rows), 4 * len(records) * 3)
            self.assertTrue(all(row["outer_test_person"] != row["person"] for row in inner_rows))
            runtime = json.loads((output / "runtime.json").read_text())
            self.assertEqual(runtime["fusion_additional_fits"], 0)
            self.assertEqual(runtime["saved_final_bundles"], 0)
            self.assertTrue(runtime["roundtrip_verified"])
            with self.assertRaises(ValueError):
                evaluation.run(records, x, y, output, models, metadata, {})


if __name__ == "__main__":
    unittest.main()
