"""Проверки безопасных преобразований 3D-поз."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_safe_augmentations import mirror_pose, resample_pose


class SafeAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.poses = np.arange(12 * 33 * 4, dtype=float).reshape(12, 33, 4)
        self.poses[:, :, 3] = np.linspace(0.2, 1.0, 12)[:, None]

    def test_mirror_twice_restores_source_without_mutation(self):
        source = self.poses.copy()
        mirrored = mirror_pose(self.poses)
        restored = mirror_pose(mirrored)
        np.testing.assert_allclose(restored, source)
        np.testing.assert_array_equal(self.poses, source)

    def test_temporal_resampling_preserves_endpoints_and_format(self):
        result = resample_pose(self.poses, 1.15)
        self.assertEqual(result.shape, (14, 33, 4))
        np.testing.assert_allclose(result[0], self.poses[0])
        np.testing.assert_allclose(result[-1], self.poses[-1])
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_resampling_factor_is_rejected(self):
        with self.assertRaises(ValueError):
            resample_pose(self.poses, 0)


if __name__ == "__main__":
    unittest.main()
