"""Последние два опыта: временная CNN и фиксированная смесь с геометрией."""

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
PROTOCOL_HEADING = "## 2026-11-14 · запись 17 · протокол до обучения: последние две гипотезы о времени и дополнении геометрии"
EXECUTION_HEADING = "## 2026-11-14 · запись 18 · технический перезапуск H17/H18: параллельные внутренние обучения"
NAMES = {
    "global35": "Исходные 35 признаков",
    "temporal_ordered": "Временная CNN, исходный порядок",
    "temporal_shuffled": "Та же CNN, перемешанное время",
    "geometry_temporal_fusion": "Фиксированная смесь геометрии и CNN 50/50",
}


def frozen_protocol(journal, heading=PROTOCOL_HEADING):
    value = Path(journal).read_text(encoding="utf-8")
    if value.count(heading) != 1:
        raise ValueError("Нужна ровно одна предварительная запись протокола/исполнения")
    start = value.index(heading)
    stop = value.find("\n## ", start + len(heading))
    return value[start:stop if stop >= 0 else None].rstrip() + "\n"


def compare(rows, folds, candidate, reference):
    people = sorted({person for method, person in folds if method == candidate})
    changes = {key: rows[candidate][key] - rows[reference][key] for key in (
        "macro_f1_mean", "false_feedback_rate_mean", "error_recall_mean",
        "specific_error_type_recall_mean")}
    wins = sum(folds[(candidate, person)]["macro_f1"] >
               folds[(reference, person)]["macro_f1"] + 1e-12 for person in people)
    return {"reference": reference, "changes": changes, "people_with_higher_macro_f1": wins}


def practical_criteria(comparison):
    changes = comparison["changes"]
    return {
        "macro_f1_gain_at_least_0_02": changes["macro_f1_mean"] >= .02 - 1e-12,
        "false_feedback_increase_at_most_2pp": changes["false_feedback_rate_mean"] <= .02 + 1e-12,
        "error_recall_loss_at_most_2pp": changes["error_recall_mean"] >= -.02 - 1e-12,
        "macro_f1_improved_for_at_least_3_people": comparison["people_with_higher_macro_f1"] >= 3,
    }


def hypothesis_checks(payload, predictions_path, historical_path):
    rows = {row["method"]: row for row in payload["summaries"]}
    folds = {(row["method"], row["test_person"]): row for row in payload["outer_folds"]}
    checks = {"evidence": "exploratory; five previously inspected people; no significance claim",
              "further_model_search_stopped": True}
    pairs = (("H17", "temporal_ordered", "temporal_shuffled"),
             ("H18", "geometry_temporal_fusion", "temporal_ordered"))
    for hypothesis, candidate, secondary_reference in pairs:
        baseline = compare(rows, folds, candidate, "global35")
        secondary = compare(rows, folds, candidate, secondary_reference)
        criteria = practical_criteria(baseline)
        criteria.update({
            "secondary_macro_f1_gain_at_least_0_01": secondary["changes"]["macro_f1_mean"] >= .01 - 1e-12,
            "secondary_macro_f1_improved_for_at_least_3_people": secondary["people_with_higher_macro_f1"] >= 3,
        })
        checks[hypothesis] = {"candidate": candidate, "vs_baseline": baseline,
                              "vs_secondary": secondary, "criteria": criteria,
                              "practical_criterion_met": all(criteria.values())}
    if historical_path.exists():
        with historical_path.open(encoding="utf-8", newline="") as stream:
            historical = {(row["physical_id"], row["camera"]): int(row["predicted_class"])
                          for row in csv.DictReader(stream) if row["method"] == "inner_selected"}
        with predictions_path.open(encoding="utf-8", newline="") as stream:
            current = {(row["physical_id"], row["camera"]): int(row["predicted_class"])
                       for row in csv.DictReader(stream) if row["method"] == "global35"}
        same = historical.keys() == current.keys()
        checks["baseline_reproduction"] = {
            "historical_file_sha256": digest(historical_path), "same_observation_keys": same,
            "observations": len(current),
            "changed_classes": sum(current[key] != historical[key] for key in current) if same else None,
        }
        if not same or checks["baseline_reproduction"]["changed_classes"]:
            raise ValueError("Исходный контроль не воспроизведён; новые выводы требуют разбора")
    return checks


def make_report(directory):
    payload = json.loads((directory / "metrics.json").read_text())
    checks = hypothesis_checks(payload, directory / "fold_predictions.csv",
                               ROOT / "reports/rehab26_baseline/fold_predictions.csv")
    write_json(directory / "hypothesis_checks.json", checks)
    lines = ["# Две последние гипотезы: временная CNN и дополнение геометрии", "",
             "388 физических повторов пяти людей, три синхронные камеры, 1 164 наблюдения. "
             "Одна камера и полный повтор на входе. Полное покрытие; без новых видео. "
             "Все сравнения исследовательские на уже просмотренных пяти людях.", "",
             "[Протокол до обучения](protocol_before_run.md). "
             "Один общий balanced-флаг двух CNN выбран симметрично только на внутренних людях. "
             "Каждая CNN — среднее вероятностей трёх seed, не выбор лучшей инициализации. "
             "Смесь геометрии и ordered CNN зафиксирована 50/50, без настройки веса.", "",
             "## Средние показатели по людям", "",
             "| Условие | Macro-F1 | Ложные замечания | Обнаружение ошибок | Верный тип ошибки | Худший F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in payload["summaries"]:
        lines.append(f"| {NAMES[row['method']]} | {row['macro_f1_mean']:.3f} | "
                     f"{100*row['false_feedback_rate_mean']:.1f}% | "
                     f"{100*row['error_recall_mean']:.1f}% | "
                     f"{100*row['specific_error_type_recall_mean']:.1f}% | {row['macro_f1_worst']:.3f} |")
    lines += ["", "## Проверка гипотез", ""]
    for hypothesis in ("H17", "H18"):
        item = checks[hypothesis]
        lines += [f"### {hypothesis}", "",
                  f"Полный заранее заданный критерий {'выполнен' if item['practical_criterion_met'] else 'не выполнен'}.", ""]
        for comparison_name in ("vs_baseline", "vs_secondary"):
            comparison = item[comparison_name]
            delta = comparison["changes"]
            lines.append(f"- Против {NAMES[comparison['reference']]}: ΔMacro-F1 {delta['macro_f1_mean']:+.4f}; "
                         f"ΔFPR {100*delta['false_feedback_rate_mean']:+.2f} п.п.; "
                         f"Δобнаружения ошибок {100*delta['error_recall_mean']:+.2f} п.п.; "
                         f"F1 выше у {comparison['people_with_higher_macro_f1']}/5 людей.")
        lines += [""] + [f"- `{key}`: {'да' if value else 'нет'}." for key, value in item["criteria"].items()] + [""]
    lines += ["## Каждый внешний человек", "",
              "| Человек | Условие | Macro-F1 | Ложные замечания | Обнаружение ошибок |",
              "|---|---|---:|---:|---:|"]
    for row in payload["outer_folds"]:
        lines.append(f"| {row['test_person']} | {NAMES[row['method']]} | {row['macro_f1']:.3f} | "
                     f"{100*row['false_feedback_rate']:.1f}% | {100*row['error_recall']:.1f}% |")
    lines += ["", "## Решение и границы вывода", "",
              "Поиск архитектур/признаков на этом наборе остановлен по согласованному бюджету. "
              "Следующий этап — работа с собственными видео и статья. При невыполненных критериях "
              "прежний baseline остаётся основой видеопроверки. Ни один вариант здесь не заменяет "
              "автоматически deployment-модель: сохранены только модели внешних разбиений.", "",
              "Это небольшая 1D CNN над семью готовыми геометрическими сигналами. Она исследует "
              "локальные временные связи и грубую позицию фазы, не извлечение признаков из RGB "
              "и не воспроизведение GCN Zhao. Нормированное время не сохраняет абсолютную длительность. "
              "Перемешивание сохраняет значения, но разрушает соседство и фазовую позицию; "
              "его эффект не доказывает конкретную биомеханическую интерпретацию. "
              "Новые люди, телефонная камера, одежда, автоматические границы и полный RGB-путь "
              "не проверены. Неудача этого рецепта не доказывает бесполезность любых временных моделей.", "",
              "- [Метрики](metrics.json), [предсказания](fold_predictions.csv).",
              "- [Проверка критериев и воспроизведения](hypothesis_checks.json).",
              "- [Выбор только на внутренних людях](inner_search.json).",
              "- [Камеры и ракурсы](group_metrics.json).",
              "- [Диагностика обучения](training_diagnostics.json), [время](runtime.json).",
              "- [Настройки и исходники](run_config.json).", ""]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/rehab26")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--label", default="rehab26_temporal_final_two_v1")
    parser.add_argument("--inner-workers", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args(argv)
    from evaluate_temporal_comparison import METHODS, PROTOCOL_VERSION, run
    from rehab26_temporal import FEATURE_METADATA, FEATURE_VERSION
    import torch
    journal = ROOT / "experiments/README.md"
    protocol = frozen_protocol(journal)
    execution_note = frozen_protocol(journal, EXECUTION_HEADING)
    experiment = Experiment(
        args.label, "REHAB26",
        hypothesis="H17: полезен ли порядок движения CNN; H18: дополняет ли CNN геометрию в фиксированной смеси 50/50?",
        note="Ровно два последних сравнения; nested LOPO; shared balanced flag; три seed до argmax; затем видеопуть.",
        parent_run="20261114-010658_rehab26_phase_relative42_v1_caa50190",
        output=args.output, models_dir=args.models_dir,
    )
    with experiment.capture():
        print(f"Журнал запуска: {experiment.directory}", flush=True)
        (experiment.report_dir / "protocol_before_run.md").write_text(protocol, encoding="utf-8")
        (experiment.report_dir / "execution_note_before_run.md").write_text(execution_note, encoding="utf-8")
        shutil.copy2(journal, experiment.report_dir / "journal_before_run.md")
        records, x35, y, _, feature_metadata, dataset_metadata = prepare(args.data_dir, experiment.report_dir)
        source_names = ("run_temporal_comparison.py", "evaluate_temporal_comparison.py", "rehab26_temporal.py",
                        "evaluate_phase_features.py", "rehab26_baseline.py", "rehab26_features.py",
                        "rehab26_data.py", "experiment_log.py")
        write_json(experiment.report_dir / "run_config.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(), "methods": list(METHODS),
            "execution_note_sha256": hashlib.sha256(execution_note.encode()).hexdigest(),
            "execution": {"inner_workers": args.inner_workers, "start_method": "spawn", "torch_threads_per_worker": 1},
            "feature_metadata": {"version": "comparison_geometry_temporal_v1",
                                 "baseline": feature_metadata,
                                 "temporal": {"version": FEATURE_VERSION, "details": FEATURE_METADATA}},
            "dataset": dataset_metadata, "torch_version": torch.__version__,
            "scripts_sha256": {name: digest(ROOT / "scripts" / name) for name in source_names},
            "temporal_requirements_sha256": digest(ROOT / "requirements-temporal.txt"),
            "hypotheses": ["H17", "H18"], "fusion_weight": .5,
            "further_model_search_stopped_after_run": True,
            "evidence": "exploratory; five previously inspected people; no independent validation",
        })
        run(records, x35, y, experiment.report_dir, experiment.models_dir, feature_metadata, dataset_metadata,
            inner_workers=args.inner_workers)
        make_report(experiment.report_dir)
    print(f"Два опыта завершены; поиск моделей остановлен: {experiment.directory}", flush=True)


if __name__ == "__main__":
    main()
