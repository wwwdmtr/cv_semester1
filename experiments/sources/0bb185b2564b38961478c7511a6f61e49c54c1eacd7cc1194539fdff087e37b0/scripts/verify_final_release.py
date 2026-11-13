"""Сравнить контрольные запуски REHAB26 с сохранённым baseline.

Скрипт не обучает модели. Сначала выполните два запуска baseline в разные
пустые каталоги, затем передайте их каталоги отчётов через --control и
--repeat-control. Критическими считаются отсутствие утечки людей, совпадение
классов выбранной модели и совпадение двух запусков текущей версии.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


KEY_COLUMNS = ("method", "person", "physical_id", "camera")
SELECTED_METHOD = "inner_selected"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_predictions(report_dir, method=None):
    with (Path(report_dir) / "fold_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if method is not None:
        rows = [row for row in rows if row["method"] == method]
    return sorted(rows, key=lambda row: tuple(row[column] for column in KEY_COLUMNS))


def metric_summary(report_dir, method=SELECTED_METHOD):
    payload = json.loads((Path(report_dir) / "metrics.json").read_text(encoding="utf-8"))
    matches = [row for row in payload["summaries"] if row["method"] == method]
    if len(matches) != 1:
        raise ValueError(f"Ожидалась одна сводка {method}, получено {len(matches)}")
    return matches[0]


def check_splits(report_dir):
    folds = json.loads((Path(report_dir) / "splits.json").read_text(encoding="utf-8"))
    problems = []
    for fold in folds:
        test_person = fold["test_person"]
        if test_person in fold["train_people"]:
            problems.append(f"{test_person} одновременно в train_people и test")
        train_ids, test_ids = set(fold["train_physical_ids"]), set(fold["test_physical_ids"])
        overlap = train_ids & test_ids
        if overlap:
            problems.append(f"{test_person}: {len(overlap)} общих physical_id")
        if any(f"_{test_person}:" in physical_id for physical_id in train_ids):
            problems.append(f"{test_person}: его повтор найден в train")
        if any(f"_{test_person}:" not in physical_id for physical_id in test_ids):
            problems.append(f"{test_person}: в test найден повтор другого человека")
    return {"folds": len(folds), "leakage_detected": bool(problems), "problems": problems}


def compare_selected(reference, control):
    old, new = read_predictions(reference, SELECTED_METHOD), read_predictions(control, SELECTED_METHOD)
    if len(old) != len(new):
        raise ValueError(f"Разное число выбранных предсказаний: {len(old)} и {len(new)}")
    changed_classes, changed_truth = 0, 0
    max_confidence_delta = 0.0
    for left, right in zip(old, new):
        if tuple(left[column] for column in KEY_COLUMNS) != tuple(right[column] for column in KEY_COLUMNS):
            raise ValueError("Не совпал порядок/состав наблюдений выбранной модели")
        changed_truth += left["true_label"] != right["true_label"]
        changed_classes += left["predicted_class"] != right["predicted_class"]
        max_confidence_delta = max(
            max_confidence_delta,
            abs(float(left["confidence_score"]) - float(right["confidence_score"])),
        )
    return {
        "observations": len(old),
        "changed_true_labels": changed_truth,
        "changed_predicted_classes": changed_classes,
        "max_confidence_delta": max_confidence_delta,
        "metrics_equal": metric_summary(reference) == metric_summary(control),
    }


def compare_all_predictions(first, second):
    left, right = read_predictions(first), read_predictions(second)
    if len(left) != len(right):
        return {"rows": min(len(left), len(right)), "exact": False, "reason": "different_row_count"}
    exact = all(a == b for a, b in zip(left, right))
    return {
        "rows": len(left),
        "exact": exact,
        "sha256_first": sha256(Path(first) / "fold_predictions.csv"),
        "sha256_second": sha256(Path(second) / "fold_predictions.csv"),
    }


def compare_auxiliary_tree(reference, control):
    old, new = read_predictions(reference, "extra_trees_tuned"), read_predictions(control, "extra_trees_tuned")
    changed = sum(left["predicted_class"] != right["predicted_class"] for left, right in zip(old, new))
    return {"observations": len(old), "changed_predicted_classes": changed, "selected_for_final_result": False}


def build_report(reference, control, repeat_control):
    runtime_one = json.loads((Path(control) / "runtime.json").read_text(encoding="utf-8"))
    runtime_two = json.loads((Path(repeat_control) / "runtime.json").read_text(encoding="utf-8"))
    selected = compare_selected(reference, control)
    repeated = compare_all_predictions(control, repeat_control)
    splits = check_splits(control)
    critical_ok = (
        selected["observations"] == 1164
        and selected["changed_true_labels"] == 0
        and selected["changed_predicted_classes"] == 0
        and selected["metrics_equal"]
        and repeated["exact"]
        and not splits["leakage_detected"]
        and runtime_one["roundtrip_verified"]
        and runtime_two["roundtrip_verified"]
    )
    metrics = metric_summary(control)
    return {
        "verification_version": 1,
        "selected_method": SELECTED_METHOD,
        "critical_checks_passed": critical_ok,
        "selected_vs_historical": selected,
        "current_run_repeated": repeated,
        "split_integrity": splits,
        "model_roundtrip": {
            "first": runtime_one["roundtrip_verified"],
            "second": runtime_two["roundtrip_verified"],
            "saved_fold_bundles_each": runtime_one["saved_fold_bundles"],
            "saved_final_bundles_each": runtime_one["saved_final_bundles"],
        },
        "selected_metrics": {
            "macro_f1_mean": metrics["macro_f1_mean"],
            "false_feedback_rate_mean": metrics["false_feedback_rate_mean"],
            "error_recall_mean": metrics["error_recall_mean"],
            "specific_error_type_recall_mean": metrics["specific_error_type_recall_mean"],
            "mean_guardrails_met": metrics["mean_guardrails_met"],
            "all_people_guardrails_met": metrics["all_people_guardrails_met"],
        },
        "noncritical_historical_difference": compare_auxiliary_tree(reference, control),
        "immutable_inputs": {
            name: {
                "equal": sha256(Path(reference) / name) == sha256(Path(control) / name),
                "sha256": sha256(Path(control) / name),
            }
            for name in ("manifest.csv", "data_audit.json", "splits.json", "feature_metadata.json")
        },
    }


def markdown(report):
    selected = report["selected_vs_historical"]
    repeated = report["current_run_repeated"]
    metrics = report["selected_metrics"]
    auxiliary = report["noncritical_historical_difference"]
    return f"""# Финальная контрольная проверка REHAB26

**Критические проверки:** {'пройдены' if report['critical_checks_passed'] else 'НЕ ПРОЙДЕНЫ'}.

| Проверка | Результат |
|---|---:|
| Наблюдений выбранной модели | {selected['observations']} |
| Изменившихся классов против сохранённого baseline | {selected['changed_predicted_classes']} |
| Максимальное изменение confidence | {selected['max_confidence_delta']:.8f} |
| Строк предсказаний в двух новых запусках | {repeated['rows']} |
| Два новых запуска совпали полностью | {'да' if repeated['exact'] else 'нет'} |
| Утечка людей/повторов | {'обнаружена' if report['split_integrity']['leakage_detected'] else 'не обнаружена'} |
| Загрузка сохранённых моделей проверена | {'да' if report['model_roundtrip']['first'] and report['model_roundtrip']['second'] else 'нет'} |

## Повторённый основной результат

| Macro-F1 | Ложные замечания | Обнаружение ошибок | Верный тип ошибки |
|---:|---:|---:|---:|
| {metrics['macro_f1_mean']:.3f} | {metrics['false_feedback_rate_mean']:.1%} | {metrics['error_recall_mean']:.1%} | {metrics['specific_error_type_recall_mean']:.1%} |

Инженерские ориентиры не достигнуты ни в среднем, ни отдельно для каждого человека.

## Некритичное историческое отличие

У отклонённого ансамбля деревьев изменились {auxiliary['changed_predicted_classes']} из
{auxiliary['observations']} классов. Выбранная логистическая регрессия и все её метрики
совпали; два запуска текущей версии полностью совпадают. Отличие не используется как
улучшение и сохранено в отчёте.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("reports/rehab26_baseline"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--repeat-control", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("reports/final_verification_run.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/final_verification_run.md"))
    args = parser.parse_args()
    report = build_report(args.reference, args.control, args.repeat_control)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    print(f"Отчёты: {args.output_markdown}; {args.output_json}")
    if not report["critical_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
