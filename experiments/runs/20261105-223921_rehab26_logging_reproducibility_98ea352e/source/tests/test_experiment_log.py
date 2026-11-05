"""Защита истории опытов: архивы, перезапись и сохранение неудачных запусков."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_log import Experiment, import_existing, require_open_report, verify


METRICS = {"summaries": [{
    "method": "inner_selected", "macro_f1_mean": .65, "macro_f1_std": .1,
    "false_feedback_rate_mean": .2, "error_recall_mean": .75,
}]}


class ExperimentLogTests(unittest.TestCase):
    def test_import_is_a_real_snapshot_and_does_not_claim_original_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_reports, old_models = root / "old_reports", root / "old_models"
            old_reports.mkdir()
            old_models.mkdir()
            (old_reports / "metrics.json").write_text(json.dumps(METRICS))
            (old_models / "model.joblib").write_bytes(b"model-parameters")
            with redirect_stdout(io.StringIO()):
                run = import_existing(old_reports, old_models, "old", "REHAB26", root=root)
            (old_reports / "metrics.json").write_text("changed afterwards")
            (old_models / "model.joblib").write_bytes(b"changed afterwards")
            self.assertEqual(json.loads((run.report_dir / "metrics.json").read_text()), METRICS)
            self.assertEqual((run.models_dir / "model.joblib").read_bytes(), b"model-parameters")
            self.assertIsNone(run.record["started_at_utc"])
            self.assertEqual(run.record["source_snapshot"]["timing"], "at_import_not_original_training")
            self.assertGreater(verify(run.directory, root), 0)

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "already_used"
            old.mkdir()
            (old / "metrics.json").write_text("keep")
            with self.assertRaises(FileExistsError):
                Experiment("new", "REHAB26", root=root, output=old)
            self.assertEqual((old / "metrics.json").read_text(), "keep")

    def test_names_are_unique_and_preparation_is_not_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = Experiment("same", "REHAB26", root=root)
            second = Experiment("same", "REHAB26", root=root)
            self.assertNotEqual(first.run_id, second.run_id)
            with first.capture("prepared"):
                pass
            with second.capture("prepared"):
                pass
            self.assertEqual(first.record["status"], "prepared")
            self.assertEqual(first.record["results"], [])

    def test_failure_retains_error_log_and_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = Experiment("failing", "REHAB26", root=root)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "expected failure"):
                    with run.capture():
                        print("step completed")
                        (run.report_dir / "partial.json").write_text('{"fold":1}')
                        raise RuntimeError("expected failure")
            record = json.loads((run.directory / "run.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("expected failure", (run.directory / "run.log").read_text())
            self.assertIn("step completed", (run.directory / "run.log").read_text())
            self.assertGreater(verify(run.directory, root), 0)

    def test_malformed_partial_metrics_does_not_hide_original_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = Experiment("partial", "REHAB26", root=root)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "interrupted calculation"):
                    with run.capture():
                        (run.report_dir / "metrics.json").write_text("{")
                        raise RuntimeError("interrupted calculation")
            self.assertEqual(run.record["status"], "failed")
            self.assertIn("metrics_read_error", run.record)
            self.assertEqual(run.record["error"]["message"], "interrupted calculation")

    def test_changed_artifact_detected_and_closed_report_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = Experiment("complete", "REHAB26", root=root)
            with run.capture():
                require_open_report(run.report_dir)
                (run.report_dir / "metrics.json").write_text(json.dumps(METRICS))
            with self.assertRaises(FileExistsError):
                require_open_report(run.report_dir)
            (run.report_dir / "metrics.json").write_text("modified")
            with self.assertRaisesRegex(ValueError, "Артефакт"):
                verify(run.directory, root)


if __name__ == "__main__":
    unittest.main()
