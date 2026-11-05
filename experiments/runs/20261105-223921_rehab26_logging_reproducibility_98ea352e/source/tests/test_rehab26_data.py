"""Проверки группировки камер и границ без больших внешних данных."""

import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_rehab26
from rehab26_data import load_repetitions


def annotation(first, last, error="", orientation="0"):
    return {
        "video_id": "ex6_Test", "person": "Test", "first_frame": str(first),
        "last_frame": str(last), "person_orientation": orientation,
        "correctness_score": "5" if error else "1", "error_description": error,
        "additional_errors": "", "repetition_number": "1",
    }


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.folder = self.root / "mediapipe/ex6"
        self.folder.mkdir(parents=True)
        self.array = np.random.default_rng(10).normal(size=(25, 33, 4))
        self.array[:, :, 3] = .8
        for camera in ("cam1", "cam2", "cam3"):
            np.save(self.folder / f"ex6_Test_{camera}_3d.npy", self.array)

    def tearDown(self):
        self.tmp.cleanup()

    def test_camera_copies_share_physical_key_and_total_weight_one(self):
        rows = [annotation(1, 5), annotation(6, 10, "ERR_KNEES_INSIDE", "10")]
        with patch("rehab26_data.read_annotations", return_value=rows):
            records, audit = load_repetitions(self.root)
        self.assertEqual(len(records), 6)
        self.assertEqual(len({r["physical_id"] for r in records}), 2)
        self.assertEqual(audit["selected_physical_repetitions"], 2)
        for pid in {r["physical_id"] for r in records}:
            views = [r for r in records if r["physical_id"] == pid]
            self.assertAlmostEqual(sum(r["view_weight"] for r in views), 1.)
            self.assertEqual({r["camera"] for r in views}, {"cam1", "cam2", "cam3"})
            self.assertTrue(all(r["n_frames"] == 5 for r in views))
        np.testing.assert_array_equal(records[0]["poses"], self.array[1:6])

    def test_out_of_bounds_camera_is_logged_without_truncation(self):
        np.save(self.folder / "ex6_Test_cam3_3d.npy", self.array[:15])
        rows = [annotation(1, 5), annotation(12, 16, "ERR_SHALLOW_SQUAT")]
        with patch("rehab26_data.read_annotations", return_value=rows):
            records, audit = load_repetitions(self.root)
        self.assertEqual(len(records), 5)
        self.assertEqual(len(audit["excluded_observations"]), 1)
        self.assertTrue(all(r["n_frames"] == 5 for r in records))
        self.assertEqual([r["view_weight"] for r in records if r["label"] == 2], [.5, .5])

    def test_other_errors_never_become_correct_negatives(self):
        rows = [annotation(1, 5), annotation(6, 10, "ERR_LIFTED_HEELS")]
        with patch("rehab26_data.read_annotations", return_value=rows):
            selected, audit = load_repetitions(self.root)
            all_records, _ = load_repetitions(self.root, include_other_errors=True)
        self.assertEqual(len(selected), 3)
        self.assertEqual(audit["excluded_other_error_physical_repetitions"], 1)
        unsupported = [r for r in all_records if r["label"] is None]
        self.assertEqual(len(unsupported), 3)
        self.assertTrue(all(r["label_name"].startswith("unsupported:") for r in unsupported))

    def test_low_visibility_is_reported_not_filtered(self):
        low = self.array.copy()
        low[:, :, 3] = .01
        np.save(self.folder / "ex6_Test_cam1_3d.npy", low)
        with patch("rehab26_data.read_annotations", return_value=[annotation(1, 5)]):
            records, audit = load_repetitions(self.root)
        self.assertEqual(len(records), 3)
        self.assertEqual(audit["files"][0]["fraction_visibility_below_half"], 1.)


class DownloadTests(unittest.TestCase):
    def test_extract_checks_crc_and_resumes_verified_file(self):
        buffer = io.BytesIO()
        name = "mediapipe/ex6/ex6_Test_cam1_3d.npy"
        content = b"trusted test bytes" * 100
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name, content)
        raw = buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            info = archive.getinfo(name)
        member = {"filename": name, "method": info.compress_type, "crc32": info.CRC,
                  "compressed_bytes": info.compress_size, "uncompressed_bytes": info.file_size,
                  "offset": info.header_offset}
        with tempfile.TemporaryDirectory() as tmp, patch.object(download_rehab26, "ARCHIVE_SIZE", len(raw)):
            with patch.object(download_rehab26, "request_bytes", side_effect=lambda url, start, end: raw[start:end + 1]):
                saved = download_rehab26.extract_member(member, Path(tmp))
            with patch.object(download_rehab26, "request_bytes", side_effect=AssertionError("resume fetched network")):
                resumed = download_rehab26.extract_member(member, Path(tmp))
            self.assertFalse(saved["reused_verified_file"])
            self.assertTrue(resumed["reused_verified_file"])
            self.assertEqual((Path(tmp) / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
