"""Диагностика потенциала переноса между камерами без обучения новых моделей.

Используются прежние внешние предсказания. Выбор камеры по истинной метке —
только недостижимый на практике ориентир, а не результат нового метода.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path

from analyze_multiview_fusion import aggregate_multiview, read_predictions, summarize
from experiment_log import Experiment, digest, write_json


CAMERAS = ("cam1", "cam2", "cam3")
LABELS = {0: "correct", 1: "knees_inward", 2: "shallow_squat"}


def diagnose(rows):
    groups = aggregate_multiview(rows)
    if len(groups) != 388 or len({row["person"] for row in rows}) != 5:
        raise ValueError("Диагностика зафиксирована для 388 повторов пяти людей")
    configurations = {
        "single_view_observation_baseline": summarize(rows, "predicted_class", "view_weight"),
    }
    for camera in CAMERAS:
        configurations[camera] = summarize(
            [row for row in rows if row["camera"] == camera], "predicted_class"
        )
    for cameras in combinations(CAMERAS, 2):
        subset = [row for row in rows if row["camera"] in cameras]
        fused = aggregate_multiview(subset, cameras)
        configurations["mean_" + "_".join(cameras)] = summarize(fused, "mean_scores")
    configurations["mean_all_three"] = summarize(groups, "mean_scores")

    categories = {}
    for label, name in LABELS.items():
        subset = [row for row in groups if row["true_label"] == label]
        categories[name] = {
            "physical_repetitions": len(subset),
            "at_least_one_camera_correct": sum(
                label in row["camera_predictions"].values() for row in subset
            ),
            "all_cameras_correct": sum(
                all(value == label for value in row["camera_predictions"].values())
                for row in subset
            ),
            "no_camera_correct": sum(
                label not in row["camera_predictions"].values() for row in subset
            ),
            "mean_all_three_correct": sum(row["mean_scores"] == label for row in subset),
            "unanimous_but_wrong": sum(
                len(set(row["camera_predictions"].values())) == 1
                and row["mean_scores"] != label for row in subset
            ),
        }
    return {
        "analysis_kind": "descriptive_reanalysis_of_existing_outer_predictions",
        "new_model_trained": False,
        "new_independent_test": False,
        "counts": {"people": 5, "physical_repetitions": len(groups), "camera_observations": len(rows)},
        "configurations": configurations,
        "by_true_category": categories,
        "unanimous_but_wrong_total": sum(row["unanimous_but_wrong"] for row in categories.values()),
        "interpretation_limits": [
            "Знание истинной метки для выбора камеры недоступно при применении.",
            "Наличие верного ответа другой камеры не доказывает переносимость информации в один ракурс.",
            "Согласие камер не означает правильность: камеры могут ошибаться одинаково.",
            "Пары камер просмотрены после baseline; это диагностика, а не подтверждение метода.",
            "Mean по людям и pooled метрики не взаимозаменяемы.",
        ],
    }


def markdown(result):
    lines = [
        "# Диагностика следующего исследовательского направления — 13.11.2026",
        "",
        "Новые модели не обучались. Повторно разобраны прежние внешние ответы на 388 приседаниях пяти людей.",
        "",
        "| Конфигурация | Macro-F1 по людям | Ложные замечания | Обнаруженные ошибки |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in result["configurations"].items():
        mean = summary["person_mean"]
        lines.append(
            f"| {name} | {mean['macro_f1']:.3f} | {mean['false_feedback_rate']:.1%} | {mean['error_recall']:.1%} |"
        )
    lines += [
        "",
        "| Метка | Всего повторов | Хотя бы одна камера верна | Все камеры верны | Ни одна не верна | Среднее трёх верно |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in result["by_true_category"].items():
        fields = ("physical_repetitions", "at_least_one_camera_correct", "all_cameras_correct", "no_camera_correct", "mean_all_three_correct")
        lines.append("| " + label + " | " + " | ".join(str(row[key]) for key in fields) + " |")
    lines += [
        "",
        f"Все камеры одновременно согласны и ошибаются в {result['unanimous_but_wrong_total']} повторах.",
        "",
        "Основной резерв — колени внутрь: другая камера иногда содержит полезный ответ, который теряется при усреднении. Это мотивирует проверку переноса между ракурсами, но не доказывает, что перенос улучшит однокамерную модель.",
        "",
        "Ограничения:",
        "",
        *["- " + value for value in result["interpretation_limits"]],
        "",
        "Воспроизведение: `python scripts/diagnose_rehab26_transfer.py` (новый отдельный запуск в журнале).",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("reports/rehab26_baseline/fold_predictions.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = Experiment(
        "rehab26_transfer_diagnostics", "REHAB26",
        hypothesis="Описательная проверка наличия различающей информации в других камерах, особенно для коленей внутрь.",
        note="Не обучение и не независимое подтверждение. Нельзя считать выбор камеры по истинной метке рабочим методом.",
        parent_run="20261105-223447_rehab26_global_geometry_v1_b8d0914e",
        output=args.output,
    )
    with run.capture():
        result = diagnose(read_predictions(args.predictions))
        result["input_predictions"] = {"path": run.relative(args.predictions), "sha256": digest(args.predictions)}
        write_json(run.report_dir / "diagnostics.json", result)
        (run.report_dir / "README.md").write_text(markdown(result), encoding="utf-8")
        for label, values in result["by_true_category"].items():
            print(label, json.dumps(values, ensure_ascii=False))
        print("Сохранён отчёт:", run.relative(run.report_dir))


if __name__ == "__main__":
    main()
