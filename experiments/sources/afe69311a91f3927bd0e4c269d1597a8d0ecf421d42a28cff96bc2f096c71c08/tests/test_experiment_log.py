"""Защита истории опытов: архивы, перезапись и сохранение неудачных запусков."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_log import Experiment, RESEARCH_NOTES_MARKER, import_existing, metric_rows, refresh_index, require_open_report, verify


METRICS = {"summaries": [{
    "method": "inner_selected", "macro_f1_mean": .65, "macro_f1_std": .1,
    "false_feedback_rate_mean": .2, "error_recall_mean": .75,
}]}


class ExperimentLogTests(unittest.TestCase):
    def test_optional_requirements_are_snapshotted_and_torch_version_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optional = root / "requirements-temporal.txt"
            optional.write_text("-r requirements.txt\ntorch==2.5.1\n")
            run = Experiment("optional_deps", "REHAB26", root=root)
            with run.capture("prepared"):
                pass
            self.assertEqual((run.source_dir / optional.name).read_text(), optional.read_text())
            self.assertIn("torch", json.loads((run.directory / "environment.json").read_text()))
            verify(run.directory, root=root)

    def test_per_method_feature_versions_survive_comparison_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "reports"
            report.mkdir()
            summaries = [dict(METRICS["summaries"][0], method="global35", feature_version="global_v1"),
                         dict(METRICS["summaries"][0], method="phase42", feature_version="phase_v1")]
            (report / "metrics.json").write_text(json.dumps({"summaries": summaries}))
            rows = metric_rows(report, "REHAB26")
            self.assertEqual([row["feature_version"] for row in rows], ["global_v1", "phase_v1"])
            run = root / "experiments/runs/first"
            run.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({
                "run_id": "first", "label": "first", "dataset": "REHAB26", "status": "completed",
                "reports": "reports", "feature_version": "comparison", "results": rows,
            }))
            refresh_index(root)
            import csv
            with (root / "experiments/comparison.csv").open(newline="") as stream:
                comparison = list(csv.DictReader(stream))
            self.assertEqual([row["feature_version"] for row in comparison], ["global_v1", "phase_v1"])

    def test_refresh_index_preserves_research_notes_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refresh_index(root)
            index = root / "experiments" / "README.md"
            marker = RESEARCH_NOTES_MARKER.encode("utf-8")
            self.assertTrue(index.read_bytes().endswith(marker + b"\n"))
            notes = marker + "\n## Решение до опыта\r\n\nТекст | [ссылка](../docs/plan.md#цель).\n\n".encode("utf-8")
            index.write_bytes(index.read_bytes().split(marker)[0] + notes)
            for iteration in range(3):
                if iteration == 1:
                    run = root / "experiments" / "runs" / "first"
                    run.mkdir(parents=True)
                    (run / "run.json").write_text(json.dumps({
                        "run_id": "first", "label": "first", "dataset": "REHAB26",
                        "status": "prepared", "reports": "reports/first", "results": [],
                    }), encoding="utf-8")
                refresh_index(root)
                content = index.read_bytes()
                self.assertEqual(content.count(marker), 1)
                self.assertEqual(content[content.index(marker):], notes)
            self.assertIn("[first]", index.read_text(encoding="utf-8"))

    def test_identical_source_snapshot_is_stored_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "tests").mkdir()
            (root / "docs").mkdir()
            (root / "scripts" / "train.py").write_text("print('same source')\n")
            (root / "tests" / "test_train.py").write_text("# same test\n")
            (root / "requirements.txt").write_text("numpy==2.2.6\n")
            (root / "docs" / "experiment-protocol.md").write_text("protocol v1\n")

            first = Experiment("first", "REHAB26", root=root)
            second = Experiment("second", "REHAB26", root=root)

            self.assertEqual(first.source_dir, second.source_dir)
            self.assertEqual(first.record["source_snapshot"]["snapshot_id"], second.record["source_snapshot"]["snapshot_id"])
            self.assertFalse((first.directory / "source").exists())
            self.assertFalse((second.directory / "source").exists())
            snapshots = [path for path in (root / "experiments" / "sources").iterdir() if path.is_dir()]
            self.assertEqual(snapshots, [first.source_dir])
            self.assertEqual((first.source_dir / "scripts" / "train.py").read_text(), "print('same source')\n")

    def test_changed_source_creates_new_snapshot_without_changing_old_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            source = root / "scripts" / "train.py"
            source.write_text("version = 1\n")
            first = Experiment("first", "REHAB26", root=root)
            source.write_text("version = 2\n")
            second = Experiment("second", "REHAB26", root=root)

            self.assertNotEqual(first.source_dir, second.source_dir)
            self.assertEqual((first.source_dir / "scripts" / "train.py").read_text(), "version = 1\n")
            self.assertEqual((second.source_dir / "scripts" / "train.py").read_text(), "version = 2\n")

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
