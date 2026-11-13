"""Сравнение качества и согласованности ответов между ракурсами REHAB26.

Скрипт использует уже сохранённые внешние предсказания. Он не переобучает
модель и не выбирает параметры по тестовым данным. Основная единица парного
анализа — одно физическое приседание, одновременно снятое несколькими камерами.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


LABEL_NAMES = {
    0: "корректно по протоколу",
    1: "колени внутрь",
    2: "недостаточная глубина",
}
LABELS = tuple(LABEL_NAMES)
DEFAULT_METHOD = "inner_selected"


def load_predictions(path: Path, method: str = DEFAULT_METHOD) -> list[dict]:
    """Загрузить строки одного заранее выбранного метода и проверить контракт."""
    with Path(path).open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["method"] == method]
    if not rows:
        raise ValueError(f"В {path} нет строк метода {method!r}")

    seen = set()
    for row in rows:
        key = (row["person"], row["physical_id"], row["camera"])
        if key in seen:
            raise ValueError(f"Повторное предсказание для человека/повтора/камеры: {key}")
        seen.add(key)
        row["true_label"] = int(row["true_label"])
        row["predicted_class"] = int(row["predicted_class"])
        row["view_weight"] = float(row["view_weight"])
        if row["true_label"] not in LABEL_NAMES or row["predicted_class"] not in LABEL_NAMES:
            raise ValueError(f"Неизвестная метка в строке {key}")
        if not np.isfinite(row["view_weight"]) or row["view_weight"] <= 0:
            raise ValueError(f"Некорректный вес наблюдения в строке {key}")
    return rows


def classification_metrics(rows: list[dict]) -> dict:
    """Посчитать показатели одиночного наблюдения с весами камер."""
    if not rows:
        raise ValueError("Нельзя вычислить метрики для пустой группы")
    truth = np.asarray([row["true_label"] for row in rows], dtype=int)
    prediction = np.asarray([row["predicted_class"] for row in rows], dtype=int)
    weights = np.asarray([row["view_weight"] for row in rows], dtype=float)

    correct_mask = truth == 0
    error_mask = truth != 0

    def weighted_rate(mask, event):
        if not np.any(mask):
            return None
        return float(np.average(event[mask], weights=weights[mask]))

    per_class = f1_score(
        truth,
        prediction,
        labels=LABELS,
        average=None,
        sample_weight=weights,
        zero_division=0,
    )
    return {
        "observations": int(len(rows)),
        "physical_repetitions": int(len({(row["person"], row["physical_id"]) for row in rows})),
        "macro_f1": float(np.mean(per_class)),
        "false_feedback_rate": weighted_rate(correct_mask, prediction != 0),
        "error_recall": weighted_rate(error_mask, prediction != 0),
        "specific_error_type_recall": weighted_rate(error_mask, prediction == truth),
        "per_class_f1": {
            LABEL_NAMES[label]: float(value) for label, value in zip(LABELS, per_class)
        },
    }


def grouped_metrics(rows: list[dict], field: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return [
        {"group": group, **classification_metrics(group_rows)}
        for group, group_rows in sorted(groups.items())
    ]


def paired_view_metrics(rows: list[dict]) -> dict:
    """Оценить расхождения камер, считая физический повтор один раз."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["person"], row["physical_id"])].append(row)

    counters = Counter()
    view_counts = Counter()
    by_label = {label: Counter() for label in LABELS}
    for key, group in groups.items():
        labels = {row["true_label"] for row in group}
        if len(labels) != 1:
            raise ValueError(f"Камеры одного физического повтора имеют разные истинные метки: {key}")
        label = labels.pop()
        predictions = [row["predicted_class"] for row in group]
        correctness = [prediction == label for prediction in predictions]
        view_counts[len(group)] += 1

        disagrees = len(set(predictions)) > 1
        all_correct = all(correctness)
        none_correct = not any(correctness)
        for counter in (counters, by_label[label]):
            counter["total"] += 1
            counter["view_disagreement"] += int(disagrees)
            counter["all_views_correct"] += int(all_correct)
            counter["no_view_correct"] += int(none_correct)

    def finish(counter):
        total = counter["total"]
        return {
            "physical_repetitions": total,
            "view_disagreement_count": counter["view_disagreement"],
            "view_disagreement_rate": counter["view_disagreement"] / total if total else None,
            "all_views_correct_count": counter["all_views_correct"],
            "all_views_correct_rate": counter["all_views_correct"] / total if total else None,
            "no_view_correct_count": counter["no_view_correct"],
            "no_view_correct_rate": counter["no_view_correct"] / total if total else None,
        }

    return {
        **finish(counters),
        "view_count_distribution": {str(key): value for key, value in sorted(view_counts.items())},
        "by_true_label": {
            LABEL_NAMES[label]: finish(by_label[label]) for label in LABELS
        },
    }


def summarize(rows: list[dict], method: str = DEFAULT_METHOD) -> dict:
    return {
        "analysis": "rehab26_view_consistency_v1",
        "method": method,
        "overall": classification_metrics(rows),
        "by_camera": grouped_metrics(rows, "camera"),
        "by_view_bin": grouped_metrics(rows, "view_bin"),
        "paired_views": paired_view_metrics(rows),
        "notes": [
            "Метрики получены из сохранённых внешних предсказаний; обучение не повторялось.",
            "Камеры одного физического повтора сравниваются попарно, но датасет содержит только пять людей.",
            "Camera ID и геометрический диапазон ракурса — разные группировки; выводы по ним нельзя смешивать.",
        ],
    }


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def metric_table(groups: list[dict], title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Группа | Наблюдения | Macro-F1 | Ложные замечания | Обнаруженные ошибки | Верный тип ошибки |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            f"| {group['group']} | {group['observations']} | {group['macro_f1']:.3f} | "
            f"{percent(group['false_feedback_rate'])} | {percent(group['error_recall'])} | "
            f"{percent(group['specific_error_type_recall'])} |"
        )
    return lines


def markdown_report(summary: dict, source: Path | str) -> str:
    paired = summary["paired_views"]
    lines = [
        "# REHAB26: согласованность предсказаний между ракурсами",
        "",
        f"Источник: `{source}`; анализируемый метод: `{summary['method']}`.",
        "Скрипт использует готовые внешние предсказания и не переобучает модель.",
        "",
        "## Главный результат",
        "",
        f"Для **{paired['view_disagreement_count']} из {paired['physical_repetitions']}** физических "
        f"приседаний (**{percent(paired['view_disagreement_rate'])}**) синхронные камеры дали "
        "разные категории. Все ракурсы одновременно распознаны правильно для "
        f"**{paired['all_views_correct_count']} повторов ({percent(paired['all_views_correct_rate'])})**; "
        f"ни один ракурс не дал правильную категорию для **{paired['no_view_correct_count']} "
        f"повторов ({percent(paired['no_view_correct_rate'])})**.",
        "",
        "Это показывает чувствительность текущего baseline к наблюдаемому ракурсу, но не "
        "доказывает причину расхождений: в наборе только пять людей, а camera ID и фактический "
        "угол съёмки не всегда означают одно и то же.",
        "",
    ]
    lines.extend(metric_table(summary["by_camera"], "По идентификатору камеры"))
    lines.append("")
    lines.extend(metric_table(summary["by_view_bin"], "По диапазону ракурса"))
    lines.extend([
        "",
        "## Расхождения по истинной категории",
        "",
        "| Истинная категория | Повторы | Камеры расходятся | Все камеры верны | Ни одна камера не верна |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, values in paired["by_true_label"].items():
        lines.append(
            f"| {label} | {values['physical_repetitions']} | {percent(values['view_disagreement_rate'])} | "
            f"{percent(values['all_views_correct_rate'])} | {percent(values['no_view_correct_rate'])} |"
        )
    lines.extend([
        "",
        "## Интерпретация и следующий шаг",
        "",
        "- Ответ модели заметно меняется для одного и того же движения при смене камеры.",
        "- Сравнение по camera ID нужно отделять от сравнения по фактическому диапазону угла.",
        "- Следующий эксперимент должен проверить, объясняют ли visibility нужных суставов и "
        "нижняя фаза эти расхождения лучше, чем одна уверенность классификатора.",
        "- До проверки на новых людях и независимых данных результат остаётся исследовательским.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "python scripts/analyze_view_consistency.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "reports" / "rehab26_baseline" / "fold_predictions.csv",
    )
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=root / "reports" / "rehab26_view_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=root / "reports" / "rehab26_view_analysis.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = load_predictions(args.predictions, args.method)
    result = summarize(rows, args.method)
    for output in (args.json_output, args.markdown_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        source_display = args.predictions.resolve().relative_to(root).as_posix()
    except ValueError:
        source_display = args.predictions.as_posix()
    args.markdown_output.write_text(markdown_report(result, source_display), encoding="utf-8")
    print(f"Saved: {args.json_output}")
    print(f"Saved: {args.markdown_output}")


if __name__ == "__main__":
    main()
