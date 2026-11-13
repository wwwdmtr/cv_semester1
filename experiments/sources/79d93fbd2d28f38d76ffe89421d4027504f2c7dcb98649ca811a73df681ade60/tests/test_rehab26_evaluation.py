"""Проверки группировки, веса камер и выбора без внешнего теста."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rehab26_baseline import classification_metrics, person_folds, physical_view_weights, select_candidate


class EvaluationTests(unittest.TestCase):
    def test_camera_copies_stay_with_their_person(self):
        records = [{"person": person, "physical_id": f"{person}_{rep}", "camera": camera}
                   for person in "ABCDE" for rep in range(2) for camera in ("cam1", "cam2", "cam3")]
        tested = []
        for fold in person_folds(records, required_people=5):
            train = {records[i]["physical_id"] for i in fold["train"]}
            test = {records[i]["physical_id"] for i in fold["test"]}
            self.assertFalse(train & test)
            self.assertEqual(len(fold["test"]), 6)
            tested.extend(fold["test"])
        self.assertEqual(sorted(tested), list(range(len(records))))

    def test_repetition_has_one_weight_even_with_different_camera_counts(self):
        records = [{"physical_id": "A"}] * 3 + [{"physical_id": "B"}]
        weights = physical_view_weights(records)
        np.testing.assert_allclose(weights, [1/3, 1/3, 1/3, 1])
        # A correct repetition is criticised in all cameras; B is correctly accepted.
        metrics = classification_metrics([0, 0, 0, 0], [1, 1, 1, 0], weights)
        self.assertAlmostEqual(metrics["false_feedback_rate"], .5)
        self.assertEqual(metrics["counts_observations"]["false_feedback"], 3)
        self.assertAlmostEqual(metrics["counts_repeat_weighted"]["false_feedback"], 1.)

    def test_wrong_error_type_is_detected_error_but_not_correct_remark(self):
        result = classification_metrics([0, 1, 2], [0, 2, 1])
        self.assertEqual(result["error_recall"], 1.)
        self.assertEqual(result["specific_error_type_recall"], 0.)
        self.assertEqual(result["specific_remark_precision"], 0.)

    def test_selection_honours_guardrails_before_f1_and_never_needs_outer_scores(self):
        bad = {"candidate": {"id": "bad", "family": "logreg"}, "mean_macro_f1": .95,
               "mean_false_feedback_rate": .4, "mean_error_recall": .99, "guardrails_met": False}
        good = {"candidate": {"id": "good", "family": "svm"}, "mean_macro_f1": .8,
                "mean_false_feedback_rate": .1, "mean_error_recall": .9, "guardrails_met": True}
        self.assertEqual(select_candidate([bad, good])["candidate"]["id"], "good")
        self.assertEqual(select_candidate([bad])["candidate"]["id"], "bad")

    def test_no_correct_examples_gives_undefined_false_alarm_rate(self):
        result = classification_metrics([1, 2], [1, 0])
        self.assertIsNone(result["false_feedback_rate"])
        self.assertEqual(result["error_recall"], .5)
        self.assertEqual(len(result["per_class_f1"]), 3)


if __name__ == "__main__":
    unittest.main()
