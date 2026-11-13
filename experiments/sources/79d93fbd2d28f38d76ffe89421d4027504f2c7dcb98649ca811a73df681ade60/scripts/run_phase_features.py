"""Сравнить 35 и 42 признака в новом архиве по заранее записанному протоколу."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from experiment_log import Experiment, digest
from rehab26_baseline import prepare, write_json


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HEADING = "## 2026-11-14 · запись 12 · протокол R2 до обучения: 35 против 42 признаков"


def frozen_protocol(journal):
    text = journal.read_text(encoding="utf-8")
    if text.count(PROTOCOL_HEADING) != 1:
        raise ValueError("Нужна ровно одна зафиксированная запись протокола R2")
    start = text.index(PROTOCOL_HEADING)
    end = text.find("\n## ", start + len(PROTOCOL_HEADING))
    return text[start:end if end >= 0 else None].rstrip() + "\n"


def make_report(directory):
    payload = json.loads((directory / "metrics.json").read_text())
    rows = {row["method"]: row for row in payload["summaries"]}
    baseline, phase = rows["global35"], rows["phase_relative42"]
    folds = {(row["method"], row["test_person"]): row for row in payload["outer_folds"]}
    people = sorted({row["test_person"] for row in payload["outer_folds"]})
    wins = sum(folds[("phase_relative42", person)]["macro_f1"] >
               folds[("global35", person)]["macro_f1"] + 1e-12 for person in people)
    keys = ("macro_f1_mean", "false_feedback_rate_mean", "error_recall_mean", "specific_error_type_recall_mean")
    changes = {key: phase[key] - baseline[key] for key in keys}
    criteria = {
        "macro_f1_gain_at_least_0_02": changes["macro_f1_mean"] >= .02 - 1e-12,
        "false_feedback_increase_at_most_2pp": changes["false_feedback_rate_mean"] <= .02 + 1e-12,
        "error_recall_loss_at_most_2pp": changes["error_recall_mean"] >= -.02 - 1e-12,
        "macro_f1_improved_for_at_least_3_people": wins >= 3,
    }
    checks = {"phase42_minus_global35": changes, "people_with_higher_macro_f1": wins,
              "criteria": criteria, "practical_criterion_met": all(criteria.values()),
              "evidence": "exploratory, five previously inspected people; not a significance test"}
    historical = ROOT / "reports/rehab26_baseline/fold_predictions.csv"
    if historical.exists():
        with historical.open(encoding="utf-8", newline="") as stream:
            old = {(r["physical_id"], r["camera"]): int(r["predicted_class"])
                   for r in csv.DictReader(stream) if r["method"] == "inner_selected"}
        with (directory / "fold_predictions.csv").open(encoding="utf-8", newline="") as stream:
            current = {(r["physical_id"], r["camera"]): int(r["predicted_class"])
                       for r in csv.DictReader(stream) if r["method"] == "global35"}
        same = old.keys() == current.keys()
        checks["baseline_reproduction"] = {
            "historical_file_sha256": digest(historical), "same_observation_keys": same,
            "observations": len(current),
            "changed_classes": sum(current[k] != old[k] for k in current) if same else None,
        }
    write_json(directory / "hypothesis_checks.json", checks)
    names = {"global35": "Исходные 35 признаков", "phase_relative42": "35 + 7 относительных фазовых признаков"}
    lines = ["# R2: глобальные и относительные фазовые 3D-признаки", "",
             "Два условия на одних 388 физических повторах пяти людей / 1 164 наблюдениях. "
             "Одна камера при применении, полное покрытие, без новых видео и GPU. "
             "Гиперпараметры выбирались только во внутренних разбиениях по людям.", "",
             "[Протокол до обучения](protocol_before_run.md). "
             "Проверяется один заранее заданный набор дополнительных признаков; "
             "исходный pipeline и основная модель не заменяются автоматически.", "",
             "## Средние показатели по людям", "",
             "| Условие | Macro-F1 | Ложные замечания | Обнаружение ошибок | Верный тип ошибки |",
             "|---|---:|---:|---:|---:|"]
    for method, row in rows.items():
        lines.append(f"| {names[method]} | {row['macro_f1_mean']:.3f} | "
                     f"{100*row['false_feedback_rate_mean']:.1f}% | "
                     f"{100*row['error_recall_mean']:.1f}% | "
                     f"{100*row['specific_error_type_recall_mean']:.1f}% |")
    lines += ["", "## Проверка заранее записанной гипотезы", "",
              f"Практический критерий {'выполнен' if all(criteria.values()) else 'не выполнен'}. "
              "Это критерий пилота, не доказательство статистической значимости.", "",
              f"Изменения 42D минус 35D: Macro-F1 {changes['macro_f1_mean']:+.4f}; "
              f"ложные замечания {100*changes['false_feedback_rate_mean']:+.2f} п.п.; "
              f"обнаружение ошибок {100*changes['error_recall_mean']:+.2f} п.п. "
              f"Macro-F1 вырос у {wins} из {len(people)} людей.", ""]
    for key, passed in criteria.items():
        lines.append(f"- `{key}`: {'да' if passed else 'нет'}.")
    lines += ["", "## Каждый внешний человек", "",
              "| Человек | Условие | Macro-F1 | Ложные замечания | Обнаружение ошибок |",
              "|---|---|---:|---:|---:|"]
    for person in people:
        for method in rows:
            row = folds[(method, person)]
            lines.append(f"| {person} | {names[method]} | {row['macro_f1']:.3f} | "
                         f"{100*row['false_feedback_rate']:.1f}% | {100*row['error_recall']:.1f}% |")
    lines += ["", "## Границы вывода", "",
              "Пять людей и прежние результаты уже просмотрены при выборе гипотезы. "
              "Признаки используют наблюдаемый полный повтор: lower-phase proxy по углам "
              "и первые/последние кадры, не экспертную разметку фаз. "
              "Это не модель, обучающая весь временной порядок: обратное воспроизведение "
              "того же повтора сохраняет симметричные края и такие признаки. "
              "Новые телефоны, одежда, RGB-путь, неполные повторы и перенос на независимых "
              "людей здесь не проверены. Файлы старых моделей и закрытых запусков сохранены.", "",
              "- [Все метрики и матрицы ошибок](metrics.json).",
              "- [Предсказания каждого наблюдения](fold_predictions.csv).",
              "- [Внутренний выбор параметров](inner_search.json).",
              "- [По ракурсам и камерам](group_metrics.json).",
              "- [Критерии и проверка исходного baseline](hypothesis_checks.json).",
              "- [Время и проверка моделей](runtime.json).",
              "- [Настройки и хеши](run_config.json).", ""]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/rehab26")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--label", default="rehab26_phase_relative42_v1")
    args = parser.parse_args(argv)
    from evaluate_phase_features import METHODS, PROTOCOL_VERSION, run
    from rehab26_phase_features import FEATURE_METADATA, FEATURE_NAMES, FEATURE_VERSION
    journal = ROOT / "experiments/README.md"
    protocol = frozen_protocol(journal)
    experiment = Experiment(
        args.label, "REHAB26",
        hypothesis="R2: помогают ли семь относительных фазовых 3D-признаков против прежних 35 при проверке нового человека?",
        note="Один заранее заданный рецепт; nested LOPO; без новых клипов, GPU и перебора после внешнего теста.",
        parent_run="20261105-223447_rehab26_global_geometry_v1_b8d0914e",
        output=args.output, models_dir=args.models_dir,
    )
    with experiment.capture():
        print(f"Журнал запуска: {experiment.directory}", flush=True)
        (experiment.report_dir / "protocol_before_run.md").write_text(protocol, encoding="utf-8")
        shutil.copy2(journal, experiment.report_dir / "journal_before_run.md")
        records, x, y, _, baseline_meta, dataset_meta = prepare(args.data_dir, experiment.report_dir)
        phase_meta = {"version": FEATURE_VERSION, "names": list(FEATURE_NAMES), "details": FEATURE_METADATA}
        source_names = ("run_phase_features.py", "evaluate_phase_features.py", "rehab26_phase_features.py",
                        "rehab26_baseline.py", "rehab26_features.py", "rehab26_data.py", "experiment_log.py")
        write_json(experiment.report_dir / "run_config.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(), "methods": list(METHODS),
            "feature_metadata": {"version": "comparison_global35_phase42_v1",
                                 "conditions": {"global35": baseline_meta, "phase_relative42": phase_meta}},
            "dataset": dataset_meta,
            "scripts_sha256": {name: digest(ROOT / "scripts" / name) for name in source_names},
            "evidence": "exploratory; previous results on all five people inspected; no independent validation",
        })
        run(records, x, y, experiment.report_dir, experiment.models_dir, baseline_meta, dataset_meta)
        make_report(experiment.report_dir)
    print(f"Результаты и настройки сохранены: {experiment.directory}", flush=True)


if __name__ == "__main__":
    main()
