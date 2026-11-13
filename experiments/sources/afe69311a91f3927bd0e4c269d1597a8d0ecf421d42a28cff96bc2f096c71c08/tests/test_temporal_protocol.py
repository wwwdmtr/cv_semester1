"""Проверка независимости двух критериев и замораживания протокола."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_temporal_comparison import PROTOCOL_HEADING, frozen_protocol, hypothesis_checks


def metrics(values):
    summaries, folds = [], []
    for method, score in values.items():
        row = {"method": method, "macro_f1_mean": score, "false_feedback_rate_mean": .15,
               "error_recall_mean": .8, "specific_error_type_recall_mean": .75}
        summaries.append(row)
        for person in "ABCDE":
            folds.append({"method": method, "test_person": person, "macro_f1": score})
    return {"summaries": summaries, "outer_folds": folds}


class TemporalProtocolTests(unittest.TestCase):
    def test_gain_over_baseline_does_not_establish_temporal_order_effect(self):
        payload = metrics({"global35": .65, "temporal_ordered": .70,
                           "temporal_shuffled": .70, "geometry_temporal_fusion": .72})
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            result = hypothesis_checks(payload, missing, missing)
        self.assertFalse(result["H17"]["practical_criterion_met"])
        self.assertTrue(result["H18"]["practical_criterion_met"])

    def test_fusion_must_improve_both_components(self):
        payload = metrics({"global35": .65, "temporal_ordered": .70,
                           "temporal_shuffled": .68, "geometry_temporal_fusion": .70})
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            result = hypothesis_checks(payload, missing, missing)
        self.assertTrue(result["H17"]["practical_criterion_met"])
        self.assertFalse(result["H18"]["practical_criterion_met"])

    def test_protocol_ends_before_later_results_and_requires_unique_heading(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "README.md"
            original = PROTOCOL_HEADING + "\n\nrecipe\n"
            journal.write_text("old\n" + original)
            before = frozen_protocol(journal)
            journal.write_text("old\n" + original + "\n## results\n\n<a id='results'></a>\nresult\n")
            self.assertEqual(frozen_protocol(journal), before)
            journal.write_text(original + original)
            with self.assertRaises(ValueError):
                frozen_protocol(journal)


if __name__ == "__main__":
    unittest.main()
