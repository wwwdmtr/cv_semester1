import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_rehab26_demo import EXAMPLES, parse_scores, prepare_examples


class Rehab26DemoTest(unittest.TestCase):
    def test_parse_scores_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            parse_scores("[0.5, 0.5]")

    def test_fixed_examples_require_all_cameras(self):
        spec = EXAMPLES[0]
        rows = [
            {
                "method": "inner_selected",
                "person": "test",
                "physical_id": spec["physical_id"],
                "camera": camera,
                "true_label": spec["expected_true"],
                "predicted_class": spec["expected_fused"],
                "confidence_score": 0.8,
                "class_scores": str([1.0 if index == spec["expected_fused"] else 0.0 for index in range(3)]),
            }
            for camera in ("cam1", "cam2")
        ]
        with self.assertRaises(ValueError):
            prepare_examples(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
