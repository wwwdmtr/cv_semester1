"""Проверки объединения синхронных предсказаний камер."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_multiview_fusion import aggregate_multiview, parse_scores


def row(physical_id, camera, truth, prediction, scores, person="A"):
    return {
        "person": person,
        "physical_id": physical_id,
        "camera": camera,
        "true_label": truth,
        "predicted_class": prediction,
        "candidate_id": "same_fold_model",
        "parsed_scores": np.asarray(scores, dtype=float),
    }


class MultiviewFusionTests(unittest.TestCase):
    def test_mean_scores_and_majority_are_calculated_once_per_repetition(self):
        rows = [
            row("first", "cam1", 1, 1, [.1, .8, .1]),
            row("first", "cam2", 1, 1, [.2, .7, .1]),
            row("first", "cam3", 1, 0, [.6, .3, .1]),
            row("second", "cam1", 0, 0, [.6, .2, .2]),
            row("second", "cam2", 0, 1, [.3, .5, .2]),
            row("second", "cam3", 0, 2, [.2, .3, .5]),
        ]
        result = aggregate_multiview(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["majority_vote"], 1)
        self.assertEqual(result[0]["mean_scores"], 1)
        # Ничья 1–1–1 разрешается средними оценками: класс 0 имеет максимум.
        self.assertEqual(result[1]["majority_vote"], 0)
        self.assertEqual(result[1]["mean_scores"], 0)

    def test_missing_camera_is_rejected(self):
        rows = [
            row("first", "cam1", 0, 0, [.8, .1, .1]),
            row("first", "cam2", 0, 0, [.8, .1, .1]),
        ]
        with self.assertRaises(ValueError):
            aggregate_multiview(rows)

    def test_inconsistent_true_label_is_rejected(self):
        rows = [
            row("first", "cam1", 0, 0, [.8, .1, .1]),
            row("first", "cam2", 1, 1, [.1, .8, .1]),
            row("first", "cam3", 0, 0, [.8, .1, .1]),
        ]
        with self.assertRaises(ValueError):
            aggregate_multiview(rows)

    def test_invalid_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_scores("[0.5, NaN, 0.5]")
        with self.assertRaises(ValueError):
            parse_scores("[0.5, 0.5]")


if __name__ == "__main__":
    unittest.main()
