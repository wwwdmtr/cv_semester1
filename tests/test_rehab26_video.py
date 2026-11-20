"""Проверки видеоконтракта без установки MediaPipe и OpenCV."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rehab26_video import (
    VideoPoseResult,
    interpolate_short_gaps,
    resolve_manual_bounds,
    world_landmarks_to_array,
)
import predict_rehab26_video as video_cli


class FakeLandmark:
    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def valid_pose_sequence(n_frames=8):
    values = np.arange(n_frames * 33 * 4, dtype=float).reshape(n_frames, 33, 4)
    values[:, :, 3] = .9
    return values


class Rehab26VideoTests(unittest.TestCase):
    def test_world_landmarks_keep_xyz_and_visibility(self):
        landmarks = [FakeLandmark(i, i + .1, i + .2, .8) for i in range(33)]
        array = world_landmarks_to_array(landmarks)
        self.assertEqual(array.shape, (33, 4))
        np.testing.assert_allclose(array[7], [7, 7.1, 7.2, .8])

    def test_wrong_landmark_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid_landmark_count"):
            world_landmarks_to_array([FakeLandmark(0, 0, 0, 1)] * 32)

    def test_frame_and_time_bounds_are_inclusive_and_validated(self):
        self.assertEqual(resolve_manual_bounds(100, 25, start_frame=10, end_frame=40), (10, 40))
        self.assertEqual(resolve_manual_bounds(100, 25, start_sec=.4, end_sec=1.64), (10, 40))
        self.assertIsNone(resolve_manual_bounds(100, 25))
        with self.assertRaisesRegex(ValueError, "mixed_bounds"):
            resolve_manual_bounds(100, 25, start_frame=1, end_frame=2, start_sec=0, end_sec=1)
        with self.assertRaisesRegex(ValueError, "invalid_bounds"):
            resolve_manual_bounds(100, 25, start_frame=20, end_frame=100)

    def test_short_internal_gap_is_interpolated_and_reported(self):
        poses = valid_pose_sequence()
        expected = poses[3].copy()
        poses[3] = np.nan
        repaired, report = interpolate_short_gaps(poses, max_gap_frames=1, min_detection_coverage=.8)
        np.testing.assert_allclose(repaired[3], expected)
        self.assertEqual(report["interpolated_frames"], 1)
        self.assertEqual(report["interpolated_gaps"][0]["length"], 1)

    def test_boundary_missing_long_gap_and_low_coverage_are_rejected(self):
        boundary = valid_pose_sequence()
        boundary[0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing_boundary_pose"):
            interpolate_short_gaps(boundary, min_detection_coverage=.8)

        long_gap = valid_pose_sequence(10)
        long_gap[3:6] = np.nan
        with self.assertRaisesRegex(ValueError, "pose_gap_too_long"):
            interpolate_short_gaps(long_gap, max_gap_frames=2, min_detection_coverage=.6)

        low_coverage = valid_pose_sequence(10)
        low_coverage[2:8] = np.nan
        with self.assertRaisesRegex(ValueError, "insufficient_pose_coverage"):
            interpolate_short_gaps(low_coverage, max_gap_frames=8, min_detection_coverage=.8)

    def test_missing_classifier_is_recorded_without_retraining(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            pose_model = root / "pose.task"
            output = root / "output"
            video.write_bytes(b"test-video")
            pose_model.write_bytes(b"test-pose-model")
            extracted = VideoPoseResult(
                poses=valid_pose_sequence(6),
                detected=np.ones(6, dtype=bool),
                fps=30.0,
                width=640,
                height=480,
                frame_count=6,
                mediapipe_version="test",
                opencv_version="test",
            )
            argv = [
                "predict_rehab26_video.py", str(video),
                "--pose-model", str(pose_model),
                "--model", str(root / "missing.joblib"),
                "--output-dir", str(output),
                "--start-frame", "0", "--end-frame", "5",
            ]
            with patch.object(video_cli, "extract_world_poses", return_value=extracted), \
                    patch.object(sys, "argv", argv), \
                    self.assertRaisesRegex(FileNotFoundError, "Классификатор не найден"):
                video_cli.main()
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "classifier_missing")
            self.assertTrue((output / "repetition_world_poses.npy").is_file())
            self.assertIn("не переобучает", metadata["technical_rejection_reason"])


if __name__ == "__main__":
    unittest.main()
