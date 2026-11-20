"""Проверки автомата границ приседания на заранее известных сигналах."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rehab26_repetition_detector import (
    RepetitionDetectorConfig,
    detect_repetitions,
    detect_repetitions_from_signals,
    find_stable_stance_start_from_signals,
)


FPS = 20.0


def signals_with_repetitions(peaks_deg, *, incomplete_last=False):
    knee_deg = [175.0] * 20
    distance = [1.50] * 20
    expected_bottoms = []
    for index, peak in enumerate(peaks_deg):
        descent = np.linspace(0, peak, 12)
        ascent = np.linspace(peak, 0, 12)[1:]
        flexion = np.concatenate([descent, ascent])
        compression_peak = max(0.07, peak / 300.0)
        compression = flexion / peak * compression_peak
        if incomplete_last and index == len(peaks_deg) - 1:
            flexion = descent
            compression = compression[:len(descent)]
        start = len(knee_deg)
        knee_deg.extend((175.0 - flexion).tolist())
        distance.extend((1.50 - compression).tolist())
        expected_bottoms.append(start + int(np.argmax(flexion)))
        if not (incomplete_last and index == len(peaks_deg) - 1):
            knee_deg.extend([175.0] * 12)
            distance.extend([1.50] * 12)
    return np.deg2rad(knee_deg), np.asarray(distance), expected_bottoms


def poses_from_knee_angles(knee_angle_radians):
    """Симметричный 3D-скелет: сгибание коленей одновременно сжимает высоту."""
    flexion = np.pi - np.asarray(knee_angle_radians)
    poses = np.zeros((len(flexion), 33, 4), dtype=float)
    poses[:, :, 3] = .95
    for joint, xyz in {
        11: (-.2, 1, 0), 12: (.2, 1, 0),
        23: (-.2, 0, 0), 24: (.2, 0, 0),
        25: (-.2, -1, 0), 26: (.2, -1, 0),
    }.items():
        poses[:, joint, :3] = xyz
    poses[:, 27, 0] = -.2 - np.sin(flexion)
    poses[:, 27, 1] = -1 - np.cos(flexion)
    poses[:, 28, 0] = .2 + np.sin(flexion)
    poses[:, 28, 1] = -1 - np.cos(flexion)
    return poses


class RepetitionDetectorTests(unittest.TestCase):
    def test_one_repetition_has_start_bottom_and_end(self):
        knee, distance, bottoms = signals_with_repetitions([70])
        result = detect_repetitions_from_signals(knee, distance, FPS)
        self.assertEqual(len(result.repetitions), 1)
        repetition = result.repetitions[0]
        self.assertLess(repetition.start_frame, repetition.bottom_frame)
        self.assertLess(repetition.bottom_frame, repetition.end_frame)
        self.assertLessEqual(abs(repetition.bottom_frame - bottoms[0]), 2)
        self.assertGreater(repetition.peak_knee_flexion_deg, 60)

    def test_world_pose_geometry_reaches_the_same_detector(self):
        knee, _, _ = signals_with_repetitions([60])
        # Синтетический генератор сигналов использует стойку 175°, геометрия — 180°.
        flexion = np.deg2rad(175.0) - knee
        poses = poses_from_knee_angles(np.pi - flexion)
        result = detect_repetitions(poses, FPS)
        self.assertEqual(len(result.repetitions), 1)
        self.assertGreater(result.repetitions[0].peak_knee_flexion_deg, 50)

    def test_two_repetitions_are_not_merged(self):
        knee, distance, _ = signals_with_repetitions([65, 55])
        result = detect_repetitions_from_signals(knee, distance, FPS)
        self.assertEqual(len(result.repetitions), 2)
        self.assertLess(result.repetitions[0].end_frame, result.repetitions[1].start_frame)

    def test_shallow_but_real_motion_is_not_filtered_as_correct_depth(self):
        knee, distance, _ = signals_with_repetitions([25])
        result = detect_repetitions_from_signals(knee, distance, FPS)
        self.assertEqual(len(result.repetitions), 1)
        self.assertLess(result.repetitions[0].peak_knee_flexion_deg, 35)

    def test_small_motion_and_unfinished_repetition_are_not_results(self):
        knee, distance, _ = signals_with_repetitions([16])
        small = detect_repetitions_from_signals(knee, distance, FPS)
        self.assertEqual(len(small.repetitions), 0)
        self.assertEqual(small.rejected_candidates[0]["reason"], "small_motion")

        knee, distance, _ = signals_with_repetitions([60], incomplete_last=True)
        incomplete = detect_repetitions_from_signals(knee, distance, FPS)
        self.assertEqual(len(incomplete.repetitions), 0)
        self.assertEqual(incomplete.rejected_candidates[0]["reason"], "incomplete_at_video_end")

    def test_unstable_initial_stance_is_rejected(self):
        knee, distance, _ = signals_with_repetitions([60])
        knee[:15] -= np.deg2rad(np.linspace(0, 20, 15))
        with self.assertRaisesRegex(ValueError, "unstable_calibration_knee"):
            detect_repetitions_from_signals(knee, distance, FPS)

    def test_search_skips_stable_bottom_and_finds_first_standing_window(self):
        knee, distance, _ = signals_with_repetitions([60])
        bottom_frames = 18
        knee = np.concatenate([np.deg2rad(np.full(bottom_frames, 90.0)), knee])
        distance = np.concatenate([np.full(bottom_frames, 1.10), distance])
        start = find_stable_stance_start_from_signals(knee, distance, FPS)
        self.assertGreaterEqual(start, bottom_frames - 2)
        result = detect_repetitions_from_signals(
            knee,
            distance,
            FPS,
            calibration_start_frame=start,
        )
        self.assertEqual(len(result.repetitions), 1)
        self.assertEqual(result.calibration_start_frame, start)

    def test_search_reports_when_no_standing_pose_exists(self):
        knee = np.deg2rad(np.full(60, 90.0))
        distance = np.full(60, 1.1)
        with self.assertRaisesRegex(ValueError, "stable_stance_not_found"):
            find_stable_stance_start_from_signals(knee, distance, FPS)

    def test_invalid_threshold_order_is_rejected(self):
        knee, distance, _ = signals_with_repetitions([60])
        config = RepetitionDetectorConfig(
            end_knee_flexion_deg=15,
            start_knee_flexion_deg=12,
        )
        with self.assertRaisesRegex(ValueError, "end_knee < start_knee"):
            detect_repetitions_from_signals(knee, distance, FPS, config)


if __name__ == "__main__":
    unittest.main()
