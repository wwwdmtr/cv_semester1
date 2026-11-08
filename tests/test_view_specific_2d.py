"""Проверки фиксированных фронтальных и профильных 2D-признаков."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_view_specific_2d import extract_physical_features


def poses(frames=20):
    result = np.zeros((frames, 33, 4), dtype=float)
    result[:, :, 3] = 1.0
    phase = np.linspace(0.0, 1.0, frames)
    bend = 0.05 * np.sin(np.pi * phase)
    for shoulder, hip, knee, ankle, side in ((11, 23, 25, 27, -1), (12, 24, 26, 28, 1)):
        result[:, shoulder, :2] = [0.5 + side * 0.10, 0.20]
        result[:, hip, :2] = [0.5 + side * 0.06, 0.40]
        result[:, knee, 0] = 0.5 + side * (0.05 - bend)
        result[:, knee, 1] = 0.60
        result[:, ankle, :2] = [0.5 + side * 0.09, 0.82]
    return result


class ViewSpecific2DTests(unittest.TestCase):
    def test_feature_vector_is_finite_and_fixed_length(self):
        result = extract_physical_features(poses(), poses())
        self.assertEqual(result.shape, (119,))
        self.assertTrue(np.isfinite(result).all())

    def test_features_ignore_image_translation_and_uniform_scale(self):
        front, profile = poses(), poses()
        expected = extract_physical_features(front, profile)
        changed_front, changed_profile = front.copy(), profile.copy()
        changed_front[:, :, :2] = changed_front[:, :, :2] * 1.7 + np.array([0.2, -0.3])
        changed_profile[:, :, :2] = changed_profile[:, :, :2] * 1.7 + np.array([0.2, -0.3])
        actual = extract_physical_features(changed_front, changed_profile)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_unsynchronized_sequences_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_physical_features(poses(20), poses(19))


if __name__ == "__main__":
    unittest.main()
