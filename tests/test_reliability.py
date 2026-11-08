"""Проверки метрик надёжности и покрытия."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_reliability import coverage_curve, ranking_metrics, visibility_features


def row(truth, prediction, confidence, visibility):
    return {
        "true_label": truth,
        "predicted_class": prediction,
        "wrong": int(truth != prediction),
        "false_correction": int(prediction != 0 and truth != prediction),
        "false_feedback_correct": int(truth == 0 and prediction != 0),
        "confidence": confidence,
        "overall_visibility": visibility,
        "phase_visibility": visibility,
        "phase_min_visibility": visibility,
    }


class ReliabilityTests(unittest.TestCase):
    def test_low_confidence_can_perfectly_rank_synthetic_errors(self):
        rows = [
            row(0, 0, .9, .5),
            row(1, 1, .8, .5),
            row(0, 1, .2, .5),
            row(2, 0, .1, .5),
        ]
        result = ranking_metrics(rows, "wrong", metrics=("confidence",))
        self.assertEqual(result["confidence"]["auroc"], 1.0)

    def test_coverage_counts_rejection_as_not_detected(self):
        rows = [
            row(0, 0, .9, .9),
            row(1, 1, .8, .8),
            row(2, 2, .2, .2),
            row(0, 1, .1, .1),
        ]
        result = coverage_curve(rows, "confidence", coverages=(.5,))[0]
        self.assertEqual(result["retained_observations"], 2)
        self.assertEqual(result["overall_error_recall_counting_rejection_as_not_detected"], .5)
        self.assertEqual(result["error_recall_among_retained_errors"], 1.0)

    def test_visibility_features_are_finite_and_use_bottom_phase(self):
        poses = np.zeros((20, 33, 4), dtype=float)
        poses[:, :, 3] = .9
        phase = np.sin(np.linspace(0, np.pi, 20))
        poses[:, 11, :3] = [-.2, 1.0, 0]
        poses[:, 12, :3] = [.2, 1.0, 0]
        poses[:, 23, :3] = [-.15, .6, 0]
        poses[:, 24, :3] = [.15, .6, 0]
        poses[:, 25, 0] = -.14
        poses[:, 26, 0] = .14
        poses[:, 25, 1] = .3 + .1 * phase
        poses[:, 26, 1] = .3 + .1 * phase
        poses[:, 27, :3] = [-.16, 0, 0]
        poses[:, 28, :3] = [.16, 0, 0]
        result = visibility_features(poses, predicted_class=1)
        self.assertGreater(result["bottom_frames"], 0)
        self.assertAlmostEqual(result["phase_visibility"], .9)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
