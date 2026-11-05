"""Локальный журнал: отдельный каталог, исходный код и артефакты каждого запуска."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def metric_rows(report_dir, dataset):
    path = Path(report_dir) / "metrics.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows = []
    for value in payload.get("summaries", []):
        rows.append({
            "dataset": dataset, "task": "three_categories", "method": value["method"],
            "macro_f1": value["macro_f1_mean"], "macro_f1_std": value["macro_f1_std"],
            "f1_aggregation": "mean_of_people", "false_feedback_rate": value["false_feedback_rate_mean"],
            "error_recall": value["error_recall_mean"], "rates_aggregation": "mean_of_people",
        })
    for value in payload.get("summary", []):
        rows.append({
            "dataset": dataset, "task": value["task"], "method": value["model"],
            "macro_f1": value["fold_macro_f1_mean"], "macro_f1_std": value["fold_macro_f1_std"],
            "f1_aggregation": "mean_of_people", "false_feedback_rate": value.get("false_feedback_rate"),
            "error_recall": None, "rates_aggregation": "pooled_predictions",
        })
    return rows


class Experiment:
    def __init__(self, label, dataset, *, hypothesis="", note="", parent_run=None,
                 output=None, models_dir=None, root=REPO_ROOT, imported=False):
        self.root = Path(root).resolve()
        stamp = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")[:60] or "experiment"
        self.run_id = f"{stamp}_{slug}_{uuid4().hex[:8]}"
        self.directory = self.root / "experiments" / "runs" / self.run_id
        self.report_dir = Path(output).resolve() if output is not None else self.directory / "reports"
        self.models_dir = Path(models_dir).resolve() if models_dir is not None else self.root / "models" / "experiments" / self.run_id
        if (self.report_dir == self.models_dir or self.report_dir in self.models_dir.parents
                or self.models_dir in self.report_dir.parents):
            raise ValueError("Каталоги отчётов и моделей должны быть раздельными")
        for path in (self.report_dir, self.models_dir):
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise FileExistsError(f"Результаты уже существуют: {path}. Выберите новый пустой каталог; перезапись запрещена.")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.record = {
            "run_id": self.run_id, "label": label, "dataset": dataset, "hypothesis": hypothesis,
            "note": note, "parent_run": parent_run, "status": "running", "recorded_at_utc": now(),
            "started_at_utc": None if imported else now(), "imported_existing_results": imported,
            "reports": self.relative(self.report_dir), "models": self.relative(self.models_dir),
            "evidence": "exploratory", "results": [],
        }
        self._capture_source(imported)
        self._save()

    def relative(self, path):
        try:
            return Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(Path(path).resolve())

    def _save(self):
        write_json(self.directory / "run.json", self.record)
        refresh_index(self.root)

    def _capture_source(self, imported):
        paths = [*self.root.glob("scripts/*.py"), *self.root.glob("tests/*.py")]
        paths += [self.root / "requirements.txt", self.root / "docs" / "experiment-protocol.md"]
        for path in paths:
            if path.is_file():
                target = self.directory / "source" / path.relative_to(self.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        versions = {"python": platform.python_version(), "platform": platform.platform()}
        for package in ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        write_json(self.directory / "environment.json", versions)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=self.root, capture_output=True, text=True)
        self.record["source_snapshot"] = {
            "timing": "at_import_not_original_training" if imported else "before_execution",
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "git_worktree_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
            "directory": self.relative(self.directory / "source"),
        }

    @contextmanager
    def capture(self, success_status="completed"):
        with (self.directory / "run.log").open("a", encoding="utf-8", buffering=1) as stream:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                try:
                    yield self
                except BaseException as exc:
                    traceback.print_exc()
                    self.record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    self.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
                    raise
                else:
                    self.finish(success_status)

    def finish(self, status):
        if self.record["status"] != "running":
            raise ValueError("Закрытый запуск не изменяется")
        self.record["status"] = status
        self.record["finished_at_utc"] = now()
        self.record["results"] = metric_rows(self.report_dir, self.record["dataset"])
        config_path = self.report_dir / "run_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            self.record["original_training_started_at_utc"] = config.get("created_utc")
            self.record["protocol_version"] = config.get("protocol_version")
            self.record["feature_version"] = config.get("feature_metadata", {}).get("version")
            ds = config.get("dataset", {})
            self.record["dataset_fingerprints"] = {k: ds[k] for k in ("annotations_sha256", "feature_matrix_sha256", "n_observations", "n_physical_repetitions") if k in ds}
            matches = {}
            for filename, expected in config.get("scripts_sha256", {}).items():
                source = self.directory / "source" / "scripts" / filename
                matches[filename] = source.exists() and digest(source) == expected
            self.record["snapshot_matches_recorded_training_scripts"] = matches or None
        artifacts = []
        for group, directory in (("reports", self.report_dir), ("models", self.models_dir), ("source", self.directory / "source")):
            if directory.exists():
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        artifacts.append({"group": group, "path": self.relative(path), "bytes": path.stat().st_size, "sha256": digest(path)})
        for filename in ("environment.json", "run.log"):
            path = self.directory / filename
            if path.exists():
                artifacts.append({"group": "journal", "path": self.relative(path), "bytes": path.stat().st_size, "sha256": digest(path)})
        write_json(self.directory / "artifacts.json", artifacts)
        self.record["artifact_manifest_sha256"] = digest(self.directory / "artifacts.json")
        self.record["artifact_count"] = len(artifacts)
        self._save()


def refresh_index(root=REPO_ROOT):
    root = Path(root)
    directory = root / "experiments"
    runs = [json.loads(p.read_text()) for p in sorted((directory / "runs").glob("*/run.json"))]
    directory.mkdir(parents=True, exist_ok=True)
    text = """# Журнал экспериментов

Каждый запуск хранится отдельно: настройки, исходный код, версии библиотек, предсказания, промежуточные внутренние проверки, итоговые метрики, графики и ссылки на сохранённые модели. Неудачные и прерванные запуски остаются в журнале. Каталог моделей находится вне Git, остальные результаты можно сохранить вместе с репозиторием.

Разные датасеты, категории и процедуры проверки не образуют честного сравнения улучшений. В CSV сохранены признаки задачи и способы усреднения; исходный EC3D и REHAB26 нужно рассматривать раздельно. F1 не является процентом правильных ответов. Архивирование старых результатов не создаёт отсутствовавшую в момент их обучения информацию.

| Запуск | Данные | Статус | Что проверяли | Результаты |
|---|---|---|---|---|
"""
    comparison = []
    for run in runs:
        location = f"runs/{run['run_id']}/run.json"
        hypothesis = (run.get("hypothesis") or run.get("note") or "—").replace("|", "/").replace("\n", " ")
        results = run.get("results", [])
        primary = next((r for r in results if r["method"] == "inner_selected"), None)
        if primary is None:
            primary = next((r for r in results if r["task"] == "instruction" and r["method"] == "geometry_logreg"), None)
        score = f"Macro-F1 {primary['macro_f1']:.3f}" if primary is not None else "—"
        text += f"| [{run['label']}]({location}) | {run['dataset']} | {run['status']} | {hypothesis} | {score} |\n"
        for result in results:
            comparison.append({"run_id": run["run_id"], "label": run["label"], "status": run["status"], **result})
    text += "\nПолная таблица всех методов: [comparison.csv](comparison.csv). Подробности ведения: [инструкция](../docs/experiment-log.md).\n"
    (directory / "README.md").write_text(text, encoding="utf-8")
    fields = ["run_id", "label", "status", "dataset", "task", "method", "macro_f1", "macro_f1_std", "f1_aggregation", "false_feedback_rate", "error_recall", "rates_aggregation"]
    with (directory / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)


def import_existing(reports, models, label, dataset, note="", root=REPO_ROOT):
    if not (Path(reports) / "metrics.json").exists():
        raise FileNotFoundError("Для импорта нужны существующие metrics.json")
    if not Path(models).is_dir():
        raise FileNotFoundError("Каталог уже сохранённых моделей не найден")
    experiment = Experiment(label, dataset, note=note, root=root, imported=True)
    with experiment.capture("imported_completed"):
        shutil.copytree(reports, experiment.report_dir, dirs_exist_ok=True)
        shutil.copytree(models, experiment.models_dir, dirs_exist_ok=True)
        print(f"Импортированы прежние результаты из {reports}; повторное обучение не запускалось.")
    return experiment


def verify(run_dir, root=REPO_ROOT):
    run_dir, root = Path(run_dir), Path(root)
    record = json.loads((run_dir / "run.json").read_text())
    if digest(run_dir / "artifacts.json") != record["artifact_manifest_sha256"]:
        raise ValueError("Изменился список артефактов")
    artifacts = json.loads((run_dir / "artifacts.json").read_text())
    for item in artifacts:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Артефакт отсутствует или изменён: {path}")
    return len(artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    archive = sub.add_parser("import-existing")
    archive.add_argument("--reports", required=True, type=Path)
    archive.add_argument("--models", required=True, type=Path)
    archive.add_argument("--label", required=True)
    archive.add_argument("--dataset", required=True)
    archive.add_argument("--note", default="")
    check = sub.add_parser("verify")
    check.add_argument("run_dir", type=Path)
    sub.add_parser("index")
    args = parser.parse_args()
    if args.command == "import-existing":
        result = import_existing(args.reports, args.models, args.label, args.dataset, args.note)
        print(f"Архив: {result.directory}")
    elif args.command == "verify":
        print(f"Проверено {verify(args.run_dir)} неизменённых артефактов")
    else:
        refresh_index()


if __name__ == "__main__":
    main()
