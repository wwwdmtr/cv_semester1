import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_final_release import check_splits


class FinalVerificationTests(unittest.TestCase):
    def write_splits(self, folds):
        directory = tempfile.TemporaryDirectory()
        Path(directory.name, "splits.json").write_text(json.dumps(folds), encoding="utf-8")
        return directory

    def test_clean_person_split(self):
        directory = self.write_splits([{
            "test_person": "Anna",
            "train_people": ["Dima"],
            "train_physical_ids": ["ex6_Dima:1-2"],
            "test_physical_ids": ["ex6_Anna:1-2"],
        }])
        self.addCleanup(directory.cleanup)
        self.assertFalse(check_splits(directory.name)["leakage_detected"])

    def test_test_person_in_training_is_rejected(self):
        directory = self.write_splits([{
            "test_person": "Anna",
            "train_people": ["Anna", "Dima"],
            "train_physical_ids": ["ex6_Anna:1-2"],
            "test_physical_ids": ["ex6_Anna:1-2"],
        }])
        self.addCleanup(directory.cleanup)
        self.assertTrue(check_splits(directory.name)["leakage_detected"])


if __name__ == "__main__":
    unittest.main()
