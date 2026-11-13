"""Проверки train-only переноса между синхронными ракурсами на синтетике."""

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
import evaluate_multiview_transfer as transfer
from rehab26_baseline import fit_pipeline, person_folds, physical_view_weights


def synthetic_data(people="ABCDE"):
    records, features = [], []
    generator = np.random.default_rng(34)
    for person_index, person in enumerate(people):
        for label in range(3):
            for camera_index, camera in enumerate(transfer.CAMERAS):
                records.append({
                    "person": person, "physical_id": f"{person}:{label}", "label": label,
                    "camera": camera, "orientation": 10.0 * person_index,
                    "view_angle": (0.0, 45.0, 90.0)[camera_index] - 10.0 * person_index,
                })
                feature = generator.normal(0, 0.15, 35)
                feature[label] += 2
                feature[3] += person_index / 10
                features.append(feature)
    return records, np.stack(features), np.array([row["label"] for row in records])


class MultiviewTransferTests(unittest.TestCase):
    def setUp(self):
        self.records, self.x, self.y = synthetic_data()

    def test_record_contract_rejects_missing_duplicate_or_inconsistent_camera(self):
        transfer.validate_records(self.records, self.x, self.y)
        with self.assertRaises(ValueError):
            transfer.validate_records(self.records[:-1], self.x[:-1], self.y[:-1])
        for field, value in (("camera", "cam2"), ("person", "other"), ("label", 2)):
            records = [dict(row) for row in self.records]
            records[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                transfer.validate_records(records, self.x, self.y)

    def test_split_contract_rejects_same_person_or_repetition(self):
        with self.assertRaises(ValueError):
            transfer.assert_disjoint(self.records, [0], [3])
        rows = [{"person": "A", "physical_id": "same"},
                {"person": "B", "physical_id": "same"}]
        with self.assertRaises(ValueError):
            transfer.assert_disjoint(rows, [0], [1])

    def test_zero_gate_is_supervised_and_unit_gate_is_uniform_distillation(self):
        teacher = np.tile([0.2, 0.3, 0.5], (len(self.y), 1))
        zero = transfer.mix_targets(self.y, teacher, np.zeros_like(teacher))
        unit = transfer.mix_targets(self.y, teacher, np.ones_like(teacher))
        np.testing.assert_array_equal(zero, np.eye(3)[self.y])
        np.testing.assert_allclose(unit, 0.5 * np.eye(3)[self.y] + 0.5 * teacher)

    def test_class_gate_uses_entire_vector_before_renormalization(self):
        result = transfer.mix_targets([1], [[0.2, 0.3, 0.5]], [[0.1, 0.2, 0.8]])
        unnormalized = np.array([[0.01, 0.9 + 0.03, 0.2]])
        np.testing.assert_allclose(result, unnormalized / unnormalized.sum())

    def test_invalid_probabilities_and_gates_are_rejected(self):
        invalid = ([[0.2, 0.3, 0.4]], [[0.2, float("nan"), 0.8]], [[-0.1, 0.1, 1.0]])
        for teacher in invalid:
            with self.subTest(teacher=teacher), self.assertRaises(ValueError):
                transfer.mix_targets([0], teacher, [[1, 1, 1]])
        with self.assertRaises(ValueError):
            transfer.mix_targets([0], [[0.2, 0.3, 0.5]], [[1, 2, 1]])

    def test_beta_smoothed_gates_use_physical_weights_and_actual_angles(self):
        rows = [dict(row, view_angle=0.0) for row in self.records[:9]]
        y = self.y[:9]
        proba = np.eye(3)[y]
        table = transfer.estimate_gate_table(rows, y, proba)
        frontal = table["frontal_abs_le_30"]
        class_one = frontal["classes"]["1"]
        self.assertAlmostEqual(class_one["positive_weight"], 1.0)
        self.assertAlmostEqual(class_one["negative_weight"], 2.0)
        self.assertAlmostEqual(class_one["smoothed_tpr"], 2 / 3)
        self.assertAlmostEqual(class_one["smoothed_tnr"], 3 / 4)
        self.assertAlmostEqual(class_one["gate"], 2 / 3 + 3 / 4 - 1)
        self.assertEqual(class_one["positive_observations"], 3)
        self.assertEqual(class_one["positive_physical_repetitions"], 1)
        self.assertEqual(frontal["physical_repetition_count"], 3)
        self.assertEqual(table["profile_abs_ge_60"]["gate_vector"], [0, 0, 0])
        self.assertEqual(frontal["gate_vector"][0], min(frontal["gate_vector"][1:]))

    def test_missing_positive_or_negative_support_forces_gate_to_zero(self):
        rows = [dict(row, view_angle=0.0) for row in self.records[:3]]
        y = np.array([0, 0, 0])
        table = transfer.estimate_gate_table(rows, y, np.eye(3)[y])
        self.assertEqual(table["frontal_abs_le_30"]["gate_vector"], [0, 0, 0])
        rows = [dict(row, view_angle=0.0) for row in self.records[3:6]]
        y = np.array([1, 1, 1])
        table = transfer.estimate_gate_table(rows, y, np.eye(3)[y])
        self.assertEqual(table["frontal_abs_le_30"]["gate_vector"], [0, 0, 0])

    def test_hard_class_expansion_matches_baseline_and_scaler_weight(self):
        weights = physical_view_weights(self.records)
        spec = dict(transfer.TEACHER)
        baseline = fit_pipeline(spec, self.x, self.y, weights)
        student = transfer.fit_soft_student(spec, self.x, self.y, weights, np.eye(3)[self.y])
        np.testing.assert_array_equal(baseline.predict_proba(self.x), student.predict_proba(self.x))
        scaler = student.named_steps["scaler"]
        self.assertAlmostEqual(scaler.n_samples_seen_, len(self.records) / 3)
        np.testing.assert_allclose(scaler.mean_, np.average(self.x, axis=0, weights=weights))
        np.testing.assert_array_equal(student.named_steps["classifier"].class_weight,
                                      baseline.named_steps["classifier"].class_weight)

    def test_soft_expansion_preserves_scaler_and_pre_class_weight_mass(self):
        weights = physical_view_weights(self.records)
        targets = 0.5 * np.eye(3)[self.y] + 0.5 / 3
        student = transfer.fit_soft_student(transfer.TEACHER, self.x, self.y, weights, targets)
        scaler = student.named_steps["scaler"]
        self.assertAlmostEqual(scaler.n_samples_seen_, weights.sum())
        expected = {label: float(weights.sum() / (3 * weights[self.y == label].sum())) for label in range(3)}
        self.assertEqual(student.named_steps["classifier"].class_weight, expected)
        np.testing.assert_allclose(student.predict_proba(self.x).sum(axis=1), 1)

    def test_targets_crossfit_people_and_all_views_share_teacher_mean(self):
        cache = transfer.TrainingTargetCache(self.records, self.x, self.y)
        indices = np.arange(36)  # Four training people; E is completely absent.
        result = cache.get(indices)
        self.assertEqual(result["audit"]["subset_people"], list("ABCD"))
        for split in result["audit"]["teacher_crossfit_splits"]:
            self.assertFalse(set(split["teacher_train_people"]) & set(split["target_people"]))
            self.assertEqual(len(split["teacher_train_people"]), 3)
            self.assertNotIn("E", split["teacher_train_people"])
            self.assertFalse(set(split["teacher_train_physical_ids"]) & set(split["target_physical_ids"]))
        for index in range(0, len(indices), 3):
            expected = result["single_view_probabilities"][index:index + 3].mean(axis=0)
            np.testing.assert_allclose(result["teacher_probabilities"][index:index + 3], np.tile(expected, (3, 1)))
        diagnostics = result["audit"]["teacher_diagnostics"]
        self.assertEqual(set(diagnostics), {"single_view", "mean_three_views"})
        self.assertEqual(set(diagnostics["mean_three_views"]["per_target_person"]), set("ABCD"))
        self.assertAlmostEqual(diagnostics["mean_three_views"]["pooled_repeat_weighted"]["counts_repeat_weighted"]["all"], 12)

    def test_exact_subset_cache_and_input_order_are_preserved(self):
        cache = transfer.TrainingTargetCache(self.records, self.x, self.y)
        indices = np.arange(27)
        first = cache.get(indices)
        self.assertIs(first, cache.get(indices[::-1]))
        self.assertEqual(len(cache.targets), 1)
        forward = cache.for_condition(indices, "distill_class_view_gate")
        backward = cache.for_condition(indices[::-1], "distill_class_view_gate")
        np.testing.assert_array_equal(forward, backward[::-1])
        cache.get(np.arange(9, 36))
        self.assertEqual(len(cache.targets), 2)
        self.assertNotEqual(cache.audits[0]["subset_id"], cache.audits[1]["subset_id"])

    def test_control_targets_and_global_gate_are_defined_without_test_data(self):
        cache = transfer.TrainingTargetCache(self.records, self.x, self.y)
        indices = np.arange(27)
        np.testing.assert_allclose(cache.for_condition(indices, "label_smoothing"),
                                   0.5 * np.eye(3)[self.y[indices]] + 0.5 / 3)
        self.assertEqual(len(cache.targets), 0)
        result = cache.get(indices)
        expected_gate = np.average(result["gates"].mean(axis=1), weights=physical_view_weights(self.records[:27]))
        self.assertAlmostEqual(result["global_gate"], expected_gate)
        expected = ((1 - 0.5 * expected_gate) * np.eye(3)[self.y[indices]]
                    + 0.5 * expected_gate * result["teacher_probabilities"])
        np.testing.assert_allclose(cache.for_condition(indices, "distill_global_gate"), expected)

    def test_nested_inner_teachers_exclude_validation_and_outer_people(self):
        cache = transfer.TrainingTargetCache(self.records, self.x, self.y)
        outer = person_folds(self.records, required_people=5)[-1]  # Hold out E.
        with patch.object(transfer, "student_candidates", return_value=[dict(transfer.TEACHER)]):
            transfer.search_students(cache, outer["train"], "distill_class_view_gate")
        self.assertEqual(len(cache.audits), 4)
        for audit in cache.audits:
            self.assertEqual(len(audit["subset_people"]), 3)
            self.assertNotIn("E", audit["subset_people"])
            for split in audit["teacher_crossfit_splits"]:
                self.assertEqual(len(split["teacher_train_people"]), 2)
                self.assertNotIn("E", split["teacher_train_people"])
                self.assertFalse(set(split["teacher_train_people"]) & set(split["target_people"]))

    def test_synthetic_run_writes_all_conditions_bundles_and_incremental_selection(self):
        # This is a tiny synthetic integration test with one candidate, not the real pilot.
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output, models = Path(tmp) / "reports", Path(tmp) / "models"
            with patch.object(transfer, "student_candidates", return_value=[dict(transfer.TEACHER)]):
                result = transfer.run(self.records, self.x, self.y, output, models)
            self.assertEqual({row["method"] for row in result["summaries"]}, set(transfer.METHODS))
            self.assertEqual(len(result["outer_folds"]), 25)
            self.assertEqual(len(list(models.rglob("*.joblib"))), 25)
            runtime = json.loads((output / "runtime.json").read_text())
            self.assertTrue(runtime["roundtrip_verified"])
            self.assertEqual(runtime["crossfit_target_subsets"], 15)
            search = json.loads((output / "inner_search.json").read_text())
            self.assertTrue(all(row["selection_complete_before_outer_prediction"] for row in search))
            with (output / "training_targets.csv").open() as stream:
                import csv
                training_targets = list(csv.DictReader(stream))
            self.assertEqual(len(training_targets), 10 * 27 + 5 * 36)
            for path in models.rglob("*distill_global_gate*.joblib"):
                self.assertTrue(joblib.load(path)["transfer_contract"]["angle_used_only_for_training_gate"])
            bundle = joblib.load(next(models.rglob("*.joblib")))
            self.assertEqual(bundle["pipeline"].n_features_in_, 35)
            self.assertEqual(bundle["feature_version"], "rehab26_global_geometry_v1")
            with self.assertRaises(ValueError):
                transfer.run(self.records, self.x, self.y, output, models)


if __name__ == "__main__":
    unittest.main()
