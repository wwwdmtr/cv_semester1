"""Проверки анализа согласованности синхронных ракурсов."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_view_consistency import classification_metrics, paired_view_metrics


def row(person, physical_id, camera, truth, prediction, weight=1 / 3):
    return {
        "person": person,
        "physical_id": physical_id,
        "camera": camera,
        "true_label": truth,
        "predicted_class": prediction,
        "view_weight": weight,
    }


class ViewConsistencyTests(unittest.TestCase):
    def test_paired_metrics_count_physical_repetition_once(self):
        rows = [
            row("A", "first", "cam1", 0, 0),
            row("A", "first", "cam2", 0, 1),
            row("A", "first", "cam3", 0, 0),
            row("B", "second", "cam1", 2, 2),
            row("B", "second", "cam2", 2, 2),
            row("B", "second", "cam3", 2, 2),
        ]
        result = paired_view_metrics(rows)
        self.assertEqual(result["physical_repetitions"], 2)
        self.assertEqual(result["view_disagreement_count"], 1)
        self.assertEqual(result["view_disagreement_rate"], .5)
        self.assertEqual(result["all_views_correct_count"], 1)
        self.assertEqual(result["no_view_correct_count"], 0)
        self.assertEqual(result["view_count_distribution"], {"3": 2})

    def test_classification_metrics_distinguish_detection_and_error_type(self):
        rows = [
            row("A", "correct", "cam1", 0, 1, 1),
            row("A", "valgus", "cam1", 1, 2, 1),
            row("A", "depth", "cam1", 2, 0, 1),
        ]
        result = classification_metrics(rows)
        self.assertEqual(result["false_feedback_rate"], 1.)
        self.assertEqual(result["error_recall"], .5)
        self.assertEqual(result["specific_error_type_recall"], 0.)

    def test_inconsistent_true_labels_between_cameras_are_rejected(self):
        rows = [
            row("A", "same", "cam1", 0, 0),
            row("A", "same", "cam2", 1, 1),
        ]
        with self.assertRaises(ValueError):
            paired_view_metrics(rows)


if __name__ == "__main__":
    unittest.main()
