"""Оценить простое объединение предсказаний трёх синхронных камер.

Скрипт не переобучает модель и использует только сохранённые внешние
предсказания. Одна физическая попытка после объединения учитывается один раз.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from rehab26_baseline import classification_metrics


CAMERAS = ("cam1", "cam2", "cam3")
METRIC_KEYS = (
    "macro_f1",
    "false_feedback_rate",
    "error_recall",
    "specific_error_type_recall",
    "specific_remark_precision",
)


def parse_scores(value):
    scores = np.asarray(json.loads(value), dtype=float)
    if scores.shape != (3,) or not np.isfinite(scores).all():
        raise ValueError("class_scores должны содержать три конечных числа")
    return scores


def read_predictions(path, method="inner_selected"):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if row["method"] == method]
    if not selected:
        raise ValueError(f"Нет предсказаний метода {method!r}")
    for row in selected:
        row["true_label"] = int(row["true_label"])
        row["predicted_class"] = int(row["predicted_class"])
        row["view_weight"] = float(row["view_weight"])
        row["parsed_scores"] = parse_scores(row["class_scores"])
    return selected


def aggregate_multiview(rows, cameras=CAMERAS):
    expected = tuple(cameras)
    groups = defaultdict(list)
    for row in rows:
        groups[row["physical_id"]].append(row)
    result = []
    for physical_id, group in sorted(groups.items()):
        by_camera = {row["camera"]: row for row in group}
        if len(by_camera) != len(group):
            raise ValueError(f"Дублируется камера для {physical_id}")
        if set(by_camera) != set(expected):
            raise ValueError(f"Для {physical_id} нужны камеры {expected}, получены {tuple(sorted(by_camera))}")
        if len({row["true_label"] for row in group}) != 1:
            raise ValueError(f"Между камерами расходится истинная метка {physical_id}")
        if len({row["person"] for row in group}) != 1:
            raise ValueError(f"Между камерами расходится участник {physical_id}")
        if len({row["candidate_id"] for row in group}) != 1:
            raise ValueError(f"Между камерами расходится модель {physical_id}")

        ordered = [by_camera[camera] for camera in expected]
        mean_scores = np.stack([row["parsed_scores"] for row in ordered]).mean(axis=0)
        predictions = [row["predicted_class"] for row in ordered]
        counts = Counter(predictions)
        most_votes = max(counts.values())
        tied = sorted(label for label, count in counts.items() if count == most_votes)
        majority = max(tied, key=lambda label: (mean_scores[label], -label))
        result.append({
            "person": group[0]["person"],
            "physical_id": physical_id,
            "true_label": group[0]["true_label"],
            "candidate_id": group[0]["candidate_id"],
            "majority_vote": int(majority),
            "mean_scores": int(np.argmax(mean_scores)),
            "camera_predictions": {camera: by_camera[camera]["predicted_class"] for camera in expected},
            "averaged_class_scores": mean_scores.tolist(),
        })
    return result


def summarize(rows, prediction_key, weight_key=None):
    people = sorted({row["person"] for row in rows})

    def calculate(subset):
        weights = None if weight_key is None else [row[weight_key] for row in subset]
        return classification_metrics(
            [row["true_label"] for row in subset],
            [row[prediction_key] for row in subset],
            weights,
        )

    per_person = {person: calculate([row for row in rows if row["person"] == person]) for person in people}
    means = {key: float(np.mean([per_person[person][key] for person in people])) for key in METRIC_KEYS}
    standard_deviations = {key: float(np.std([per_person[person][key] for person in people])) for key in METRIC_KEYS}
    return {
        "pooled": calculate(rows),
        "person_mean": means,
        "person_std": standard_deviations,
        "per_person": per_person,
    }


def analyze(rows, method="inner_selected", cameras=CAMERAS):
    aggregated = aggregate_multiview(rows, cameras)
    baseline = summarize(rows, "predicted_class", "view_weight")
    camera_results = {
        camera: summarize([row for row in rows if row["camera"] == camera], "predicted_class")
        for camera in cameras
    }
    majority = summarize(aggregated, "majority_vote")
    mean_scores = summarize(aggregated, "mean_scores")
    delta = {
        key: mean_scores["person_mean"][key] - baseline["person_mean"][key]
        for key in METRIC_KEYS
    }
    return {
        "method": method,
        "cameras": list(cameras),
        "observations": len(rows),
        "physical_repetitions": len(aggregated),
        "aggregation": {
            "majority_vote": "Большинство классов; при 1–1–1 ничья разрешается средним class_scores.",
            "mean_scores": "Среднее трёх class_scores, затем argmax.",
        },
        "observation_baseline": baseline,
        "single_camera": camera_results,
        "majority_vote": majority,
        "mean_scores": mean_scores,
        "mean_scores_minus_observation_baseline_person_mean": delta,
        "limitations": [
            "Анализ описательный и выполнен после просмотра baseline; это не независимое подтверждение нового метода.",
            "Требуются три синхронных ракурса одной попытки; результат не переносится автоматически на одну камеру.",
            "class_scores baseline не калиброваны; усредняются только оценки одного fold-моделя для одного повтора.",
            "Ни один вариант не подтвердил целевые ограничения отдельно на каждом из пяти участников.",
        ],
    }


def percent(value):
    return f"{100 * value:.1f}%"


def number(value):
    return f"{value:.3f}"


def markdown_report(result):
    baseline = result["observation_baseline"]["person_mean"]
    vote = result["majority_vote"]["person_mean"]
    mean_scores = result["mean_scores"]["person_mean"]
    pooled_baseline = result["observation_baseline"]["pooled"]
    pooled_mean = result["mean_scores"]["pooled"]
    rows = [
        ("Исходные наблюдения камер", baseline),
        ("Большинство трёх камер", vote),
        ("Среднее class_scores трёх камер", mean_scores),
    ]
    table = [
        "| Вариант | Macro-F1 | Ложные замечания | Обнаружение ошибок | Верный тип ошибки | Точность конкретного замечания |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in rows:
        table.append(
            f"| {label} | {number(metrics['macro_f1'])} | {percent(metrics['false_feedback_rate'])} | "
            f"{percent(metrics['error_recall'])} | {percent(metrics['specific_error_type_recall'])} | "
            f"{percent(metrics['specific_remark_precision'])} |"
        )
    per_person = result["mean_scores"]["per_person"]
    people_table = [
        "| Участник | Macro-F1 | Ложные замечания | Обнаружение ошибок |",
        "|---|---:|---:|---:|",
    ]
    for person, metrics in per_person.items():
        people_table.append(
            f"| {person} | {number(metrics['macro_f1'])} | {percent(metrics['false_feedback_rate'])} | {percent(metrics['error_recall'])} |"
        )
    return "\n".join([
        "# Исследование объединения синхронных ракурсов REHAB26",
        "",
        f"Проанализированы сохранённые внешние предсказания `{result['method']}`: "
        f"{result['observations']} наблюдения трёх камер для {result['physical_repetitions']} физических повторов. "
        "Модель не переобучалась, одна физическая попытка после объединения учитывается один раз.",
        "",
        "## Основной результат",
        "",
        "Средние показатели по пяти отложенным участникам:",
        "",
        *table,
        "",
        "Усреднение оценок трёх камер повышает средний Macro-F1 с "
        f"{number(baseline['macro_f1'])} до {number(mean_scores['macro_f1'])}, снижает ложные замечания с "
        f"{percent(baseline['false_feedback_rate'])} до {percent(mean_scores['false_feedback_rate'])} и сохраняет "
        f"обнаружение ошибок на близком уровне ({percent(baseline['error_recall'])} → {percent(mean_scores['error_recall'])}).",
        "",
        "Для строго pooled-сравнения с одинаковым способом подсчёта Macro-F1 меняется "
        f"с {number(pooled_baseline['macro_f1'])} до {number(pooled_mean['macro_f1'])}. Это уточнение важно: "
        "среднее по людям и pooled Macro-F1 не являются одной и той же величиной.",
        "",
        "## Результаты по людям",
        "",
        *people_table,
        "",
        "Разброс остаётся большим: у Honza сохраняется высокая доля ложных замечаний, а у Vasil — низкое обнаружение ошибок. "
        "Поэтому агрегатор пока является исследовательским результатом, а не готовым режимом автоматических рекомендаций.",
        "",
        "## Как считается",
        "",
        "1. Для каждого физического повтора проверяются ровно `cam1`, `cam2`, `cam3`, единая истинная метка и единая fold-модель.",
        "2. Основной вариант усредняет три вектора `class_scores` и берёт класс с максимальным средним значением.",
        "3. Контрольный вариант использует большинство классов; ничья 1–1–1 разрешается средним `class_scores`.",
        "4. Сравнение использует только уже сохранённые внешние предсказания, без повторного обучения.",
        "",
        "Воспроизведение:",
        "",
        "```bash",
        "python scripts/analyze_multiview_fusion.py",
        "```",
        "",
        "## Ограничения",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
        "Следующая проверка: заранее зафиксировать правило объединения и подтвердить его на новых людях или независимом наборе данных.",
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("reports/rehab26_baseline/fold_predictions.csv"))
    parser.add_argument("--method", default="inner_selected")
    parser.add_argument("--output-json", type=Path, default=Path("reports/rehab26_multiview_analysis.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/rehab26_multiview_analysis.md"))
    args = parser.parse_args(argv)
    result = analyze(read_predictions(args.predictions, args.method), args.method)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_report(result), encoding="utf-8")
    print(f"Отчёты: {args.output_markdown}; {args.output_json}")


if __name__ == "__main__":
    main()
