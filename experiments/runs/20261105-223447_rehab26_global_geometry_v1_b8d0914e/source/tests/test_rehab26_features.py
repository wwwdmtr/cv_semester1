"""Проверки контракта 3D-признаков: анатомия, инвариантность и плохой вход."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rehab26_features import FEATURE_NAMES, FEATURE_VERSION, extract_features


def asymmetric_pose(n_frames=4):
    """Правая нога прямая, левое колено согнуто на 90°, длина корпуса 1."""
    pose = np.zeros((n_frames, 33, 4), dtype=float)
    pose[:, :, 3] = .95
    joints = {
        11: (-.2, 1, 0), 12: (.2, 1, 0),
        23: (-.2, 0, 0), 24: (.2, 0, 0),
        25: (-.2, -1, 0), 26: (.2, -1, 0),
        27: (.8, -1, 0), 28: (.2, -2, 0),
    }
    for joint, xyz in joints.items():
        pose[:, joint, :3] = xyz
    return pose


class Rehab26FeatureTests(unittest.TestCase):
    def test_anatomical_indices_and_quantile_major_order(self):
        pose = asymmetric_pose()
        hip_angle = np.arccos(-1 / np.sqrt(1.04))
        expected_signals = np.array([
            np.pi, np.pi / 2, hip_angle, hip_angle,
            .4, np.sqrt(1.36), np.sqrt(2.5),
        ])
        features = extract_features(pose)
        self.assertEqual(FEATURE_VERSION, "rehab26_global_geometry_v1")
        self.assertEqual(features.shape, (35,))
        self.assertEqual(len(FEATURE_NAMES), 35)
        self.assertEqual(len(set(FEATURE_NAMES)), 35)
        self.assertEqual(FEATURE_NAMES[0], "right_knee_angle_q0")
        self.assertEqual(FEATURE_NAMES[1], "left_knee_angle_q0")
        self.assertEqual(FEATURE_NAMES[7], "right_knee_angle_q0.1")
        np.testing.assert_allclose(features.reshape(5, 7), np.tile(expected_signals, (5, 1)), atol=1e-12)

    def test_geometry_is_independent_of_rigid_position_rotation_and_uniform_size(self):
        rng = np.random.default_rng(42)
        pose = rng.normal(size=(20, 33, 4))
        pose[:, :, 3] = .9
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        changed = pose.copy()
        changed[:, :, :3] = 2.8 * (pose[:, :, :3] @ rotation) + [100, -50, 20]
        np.testing.assert_allclose(extract_features(pose), extract_features(changed), atol=1e-12)

    def test_baseline_does_not_use_visibility(self):
        pose = asymmetric_pose()
        changed = pose.copy()
        changed[:, :, 3] = np.linspace(0, 1, changed.shape[1])[None, :]
        np.testing.assert_array_equal(extract_features(pose), extract_features(changed))

    def test_no_mutation_of_source_coordinates(self):
        pose = asymmetric_pose()
        original = pose.copy()
        pose.flags.writeable = False
        extract_features(pose)
        np.testing.assert_array_equal(pose, original)

    def test_global_quantiles_do_not_claim_temporal_order(self):
        pose = np.random.default_rng(11).normal(size=(20, 33, 4))
        pose[:, :, 3] = .9
        np.testing.assert_allclose(extract_features(pose), extract_features(pose[::-1]), atol=1e-12)

    def test_invalid_shape_and_insufficient_frames_rejected(self):
        for bad in (np.zeros((1, 33, 4)), np.zeros((8, 25, 4)), np.zeros((8, 33, 3)), np.zeros((33, 4))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                extract_features(bad)

    def test_nonfinite_coordinates_rejected(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            pose = asymmetric_pose()
            pose[1, 25, 0] = bad_value
            with self.subTest(value=bad_value), self.assertRaises(ValueError):
                extract_features(pose)

    def test_zero_torso_or_leg_segment_rejected(self):
        torso = asymmetric_pose()
        torso[:, 11, :3] = torso[:, 23, :3]
        torso[:, 12, :3] = torso[:, 24, :3]
        leg = asymmetric_pose()
        leg[:, 26, :3] = leg[:, 24, :3]
        for name, bad in (("torso", torso), ("leg", leg)):
            with self.subTest(segment=name), self.assertRaises(ValueError):
                extract_features(bad)


if __name__ == "__main__":
    unittest.main()
