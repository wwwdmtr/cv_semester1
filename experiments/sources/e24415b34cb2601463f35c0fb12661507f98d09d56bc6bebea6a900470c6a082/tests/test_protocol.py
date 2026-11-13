"""Проверки предпосылок эксперимента: группы, отбор и инвариантность."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ec3d_baseline import geometric_features, select_squats, subject_folds


class ProtocolTests(unittest.TestCase):
    def test_subjects_never_cross_partitions(self):
        sequences = [{"subject": person} for person in "ABCD" for _ in range(3)]
        test_indices = []
        for fold in subject_folds(sequences):
            groups = [{sequences[i]["subject"] for i in fold[part]} for part in ("train", "val", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertEqual(set(fold["train"] + fold["val"] + fold["test"]), set(range(12)))
            test_indices.extend(fold["test"])
        self.assertEqual(sorted(test_indices), list(range(12)))

    def test_undocumented_instruction_cannot_enter_training(self):
        sequences = [{"action": action, "instruction": label} for action in ("SQUAT", "Plank") for label in (1, 2, 3, 4, 5, 10)]
        selected = select_squats(sequences)
        self.assertEqual([s["instruction"] for s in selected], [1, 2, 3, 4, 5])
        self.assertTrue(all(s["action"] == "SQUAT" for s in selected))

    def test_geometry_ignores_camera_translation_scale_and_rigid_rotation(self):
        poses = np.random.default_rng(7).normal(size=(24, 25, 3))
        rotation, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(3, 3)))
        transformed = 3.7 * (poses @ rotation) + [20, -10, 8]
        np.testing.assert_allclose(geometric_features(poses), geometric_features(transformed), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
