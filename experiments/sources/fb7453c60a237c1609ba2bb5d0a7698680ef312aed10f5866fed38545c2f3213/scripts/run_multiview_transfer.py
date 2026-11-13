"""Запустить зафиксированный исследовательский пилот в отдельном архиве."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from experiment_log import Experiment, digest
from rehab26_baseline import prepare, write_json


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "multiview-transfer-protocol.md"


def make_report(directory):
    """Описать все условия и заранее фиксированные критерии, не выбирать победителя."""
    payload = json.loads((directory / "metrics.json").read_text())
    summaries = {row["method"]: row for row in payload["summaries"]}
    baseline = summaries["supervised"]
    uniform = summaries["distill_uniform"]
    conditional = summaries["distill_class_view_gate"]
    people = sorted({row["test_person"] for row in payload["outer_folds"]})
    by_person = {(row["method"], row["test_person"]): row for row in payload["outer_folds"]}
    improved_people = sum(
        by_person[("distill_class_view_gate", person)]["false_feedback_rate"]
        < by_person[("distill_uniform", person)]["false_feedback_rate"] - 1e-12
        for person in people
    )
    fpr_change = conditional["false_feedback_rate_mean"] - uniform["false_feedback_rate_mean"]
    recall_change = conditional["error_recall_mean"] - uniform["error_recall_mean"]
    f1_change = conditional["macro_f1_mean"] - uniform["macro_f1_mean"]
    criteria = {
        "false_feedback_reduced_at_least_3pp": fpr_change <= -0.03 + 1e-12,
        "error_recall_loss_at_most_2pp": recall_change >= -0.02 - 1e-12,
        "macro_f1_not_reduced": f1_change >= -1e-12,
        "false_feedback_reduced_in_at_least_3_people": improved_people >= 3,
    }
    checks = {
        "criterion_frozen_in": "protocol_before_run.md",
        "conditional_minus_uniform": {
            "false_feedback_rate": fpr_change, "error_recall": recall_change,
            "macro_f1": f1_change, "people_with_lower_false_feedback": improved_people,
        },
        "criteria": criteria, "h1_practical_criterion_met": all(criteria.values()),
        "conditional_minus_controls": {
            method: {key: conditional[key] - summary[key] for key in (
                "macro_f1_mean", "false_feedback_rate_mean", "error_recall_mean")}
            for method, summary in summaries.items() if method != "distill_class_view_gate"
        },
        "evidence": "exploratory; no population significance claim; all five participants previously inspected",
    }
    historical = ROOT / "reports/rehab26_baseline/fold_predictions.csv"
    if historical.exists():
        with historical.open(newline="", encoding="utf-8") as stream:
            old = {(r["physical_id"], r["camera"]): int(r["predicted_class"])
                   for r in csv.DictReader(stream) if r["method"] == "inner_selected"}
        with (directory / "fold_predictions.csv").open(newline="", encoding="utf-8") as stream:
            current = {(r["physical_id"], r["camera"]): int(r["predicted_class"])
                       for r in csv.DictReader(stream) if r["method"] == "supervised"}
        same_keys = old.keys() == current.keys()
        checks["supervised_vs_historical_selected"] = {
            "historical_file_sha256": digest(historical), "same_observation_keys": same_keys,
            "observations": len(current),
            "changed_classes": sum(current[key] != old[key] for key in current) if same_keys else None,
        }
    write_json(directory / "hypothesis_checks.json", checks)
    names = {
        "supervised": "Только исходные метки",
        "label_smoothing": "Сглаживание меток",
        "distill_uniform": "Обычный перенос трёх камер",
        "distill_global_gate": "Перенос с общим ослаблением",
        "distill_class_view_gate": "Перенос с весами класса/ракурса",
    }
    lines = [
        "# Первый пилот: несколько камер при обучении, одна при применении", "",
        "Исследовательский опыт на 388 повторах пяти людей, 1 164 наблюдениях. "
        "На тесте модель получает только 35 геометрических признаков одной камеры; "
        "другие камеры и угол недоступны. Покрытие 100%. GPU не используется.", "",
        "Протокол, записанный до обучения: [protocol_before_run.md](protocol_before_run.md). "
        "Все пять условий имеют одинаковую внутреннюю сетку student-параметров. "
        "Teacher targets и веса заново cross-fit только внутри каждого train subset.", "",
        "## Результат: среднее по пяти людям", "",
        "| Условие | Macro-F1 | Ложные замечания | Обнаружение ошибок | Верный тип ошибки |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in summaries.items():
        lines.append(f"| {names[method]} | {row['macro_f1_mean']:.3f} | "
                     f"{100*row['false_feedback_rate_mean']:.1f}% | "
                     f"{100*row['error_recall_mean']:.1f}% | "
                     f"{100*row['specific_error_type_recall_mean']:.1f}% |")
    lines += ["", "## Заранее заданная проверка H1", "",
              f"Практический критерий {'выполнен' if all(criteria.values()) else 'не выполнен'}. "
              "Это решение данного пилота, не доказательство статистической значимости.", "",
              f"Conditional минус обычный перенос: Macro-F1 {f1_change:+.4f}; "
              f"ложные замечания {100*fpr_change:+.2f} п.п.; обнаружение ошибок {100*recall_change:+.2f} п.п. "
              f"FPR уменьшился у {improved_people} из {len(people)} людей.", ""]
    for key, passed in criteria.items():
        lines.append(f"- `{key}`: {'да' if passed else 'нет'}.")
    lines += ["", "Преимущество над плохим обычным переносом само по себе не означает "
              "улучшения проекта: нужны сравнения с supervised и обоими контролями. "
              "Все разности сохранены в [hypothesis_checks.json](hypothesis_checks.json).", "",
              "## Каждый внешний человек", "",
              "| Человек | Условие | Macro-F1 | Ложные замечания | Обнаружение ошибок |",
              "|---|---|---:|---:|---:|"]
    for person in people:
        for method in summaries:
            row = by_person[(method, person)]
            lines.append(f"| {person} | {names[method]} | {row['macro_f1']:.3f} | "
                         f"{100*row['false_feedback_rate']:.1f}% | {100*row['error_recall']:.1f}% |")
    lines += ["", "## Ограничения и артефакты", "",
              "Пять людей и прежние результаты уже просмотрены при выборе направления; "
              "опыт не является независимым подтверждением. Углы из разметки — дополнительная "
              "информация обучения. Веса класса/ракурса — proxy по ошибкам маленького teacher; "
              "они не измеряют физическую наблюдаемость. Поклассовая нормировка меняет массы "
              "классов; scalar gate не является точным matched-strength контролем. "
              "Пилот ограничен 35 глобальными признаками и фиксированным teacher. "
              "Разные камеры одного движения не являются независимыми людьми.", "",
              "- [Метрики и поклассовые результаты](metrics.json).",
              "- [Предсказания каждой камеры](fold_predictions.csv).",
              "- [Внутренний выбор до внешних ответов](inner_search.json).",
              "- [Аудит cross-fit и таблиц весов](training_audits.json).",
              "- [Группы ракурсов и камеры](group_metrics.json).",
              "- [Проверка загрузки моделей и время](runtime.json).",
              "- [Настройки и хеши данных/кода](run_config.json).", ""]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/rehab26")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--label", default="rehab26_multiview_transfer_v1")
    args = parser.parse_args(argv)
    if not PROTOCOL.is_file():
        raise FileNotFoundError(f"Нет зафиксированного протокола: {PROTOCOL}")
    import evaluate_multiview_transfer as transfer
    experiment = Experiment(
        args.label, "REHAB26",
        hypothesis="Проверить перенос трёх камер в одну и веса по классу/ракурсу при полностью вложенном cross-fit.",
        note="Exploratory pilot; пять уже просмотренных людей; 100% coverage; alpha=0.5; пять условий, без выбора победителя по внешнему тесту.",
        parent_run="20261105-223447_rehab26_global_geometry_v1_b8d0914e",
        output=args.output, models_dir=args.models_dir,
    )
    with experiment.capture():
        print(f"Журнал запуска: {experiment.directory}", flush=True)
        shutil.copy2(PROTOCOL, experiment.report_dir / "protocol_before_run.md")
        diary = ROOT / "docs/research-diary.md"
        if diary.exists():
            shutil.copy2(diary, experiment.report_dir / "diary_before_run.md")
        records, x, y, _, feature_metadata, dataset_metadata = prepare(args.data_dir, experiment.report_dir)
        sources = (
            "run_multiview_transfer.py", "evaluate_multiview_transfer.py",
            "rehab26_baseline.py", "rehab26_data.py", "rehab26_features.py",
            "experiment_log.py",
        )
        config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": transfer.PROTOCOL_VERSION,
            "protocol_sha256": digest(PROTOCOL),
            "methods": list(transfer.METHODS),
            "feature_metadata": feature_metadata,
            "dataset": dataset_metadata,
            "scripts_sha256": {name: digest(ROOT / "scripts" / name) for name in sources},
            "student_inference": "one observation, 35 geometry features; no angle, other cameras, labels or personal references",
            "evidence": "exploratory; previously inspected participants; not independent confirmation",
        }
        write_json(experiment.report_dir / "run_config.json", config)
        transfer.run(
            records, x, y, experiment.report_dir, experiment.models_dir,
            feature_metadata=feature_metadata, dataset_metadata=dataset_metadata,
        )
        make_report(experiment.report_dir)
    print(f"Результаты и настройки сохранены: {experiment.directory}", flush=True)


if __name__ == "__main__":
    main()
