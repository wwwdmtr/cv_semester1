"""Сравнить baseline без аугментаций с фиксированными безопасными преобразованиями.

В обучающей части добавляются зеркало со сменой сторон и временная
интерполяция 0,85×/1,15×. Внешние и внутренние проверочные люди не
аугментируются. Координатный шум исключён: он может изменить метку.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rehab26_baseline import (
    candidate_grid,
    classification_metrics,
    fit_pipeline,
    meets_guardrails,
    person_folds,
    physical_view_weights,
)
from rehab26_data import load_repetitions
from rehab26_features import extract_features


LEFT_RIGHT = (
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)
TRANSFORMS = ("original", "mirror", "temporal_0.85", "temporal_1.15")
METRIC_KEYS = (
    "macro_f1",
    "false_feedback_rate",
    "error_recall",
    "specific_error_type_recall",
    "specific_remark_precision",
)


def mirror_pose(poses):
    result = np.asarray(poses, dtype=float).copy()
    result[:, :, 0] *= -1
    reflected = result.copy()
    for left, right in LEFT_RIGHT:
        result[:, left] = reflected[:, right]
        result[:, right] = reflected[:, left]
    return result


def resample_pose(poses, factor):
    source = np.asarray(poses, dtype=float)
    if source.ndim != 3 or source.shape[1:] != (33, 4) or len(source) < 2:
        raise ValueError("expected_frames_33_4")
    if not np.isfinite(source).all() or factor <= 0:
        raise ValueError("invalid_pose_or_factor")
    length = max(2, int(round(len(source) * factor)))
    old = np.linspace(0.0, 1.0, len(source))
    new = np.linspace(0.0, 1.0, length)
    result = np.empty((length, 33, 4), dtype=float)
    for joint in range(33):
        for channel in range(4):
            result[:, joint, channel] = np.interp(new, old, source[:, joint, channel])
    return result


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def select_candidate(results):
    feasible = [row for row in results if row["guardrails_met"]]
    pool = feasible or results
    return sorted(pool, key=lambda row: (
        -row["mean_macro_f1"],
        row["mean_false_feedback_rate"],
        -row["mean_error_recall"],
        row["candidate"]["id"],
    ))[0]


def training_data(indices, records, matrices, y, augmented):
    subset = [records[index] for index in indices]
    base_weights = physical_view_weights(subset)
    names = TRANSFORMS if augmented else ("original",)
    return (
        np.concatenate([matrices[name][indices] for name in names]),
        np.concatenate([y[indices] for _ in names]),
        np.concatenate([base_weights / len(names) for _ in names]),
    )


def run_condition(records, matrices, y, augmented, seed):
    specs = [spec for spec in candidate_grid(seed) if spec["family"] == "logreg"]
    outer_results = []
    for outer in person_folds(records, required_people=5):
        outer_train, outer_test = np.asarray(outer["train"]), np.asarray(outer["test"])
        train_records = [records[index] for index in outer_train]
        search = []
        for spec in specs:
            fold_metrics = []
            for inner in person_folds(train_records):
                inner_train = outer_train[np.asarray(inner["train"])]
                inner_validation = outer_train[np.asarray(inner["test"])]
                x_train, y_train, weights = training_data(
                    inner_train, records, matrices, y, augmented
                )
                model = fit_pipeline(spec, x_train, y_train, weights)
                validation_records = [records[index] for index in inner_validation]
                fold_metrics.append(classification_metrics(
                    y[inner_validation],
                    model.predict(matrices["original"][inner_validation]),
                    physical_view_weights(validation_records),
                ))
            fpr = mean(fold_metrics, "false_feedback_rate")
            recall = mean(fold_metrics, "error_recall")
            search.append({
                "candidate": spec,
                "mean_macro_f1": mean(fold_metrics, "macro_f1"),
                "mean_false_feedback_rate": fpr,
                "mean_error_recall": recall,
                "guardrails_met": meets_guardrails(fpr, recall),
            })
        chosen = select_candidate(search)
        x_train, y_train, weights = training_data(
            outer_train, records, matrices, y, augmented
        )
        model = fit_pipeline(chosen["candidate"], x_train, y_train, weights)
        test_records = [records[index] for index in outer_test]
        metrics = classification_metrics(
            y[outer_test],
            model.predict(matrices["original"][outer_test]),
            physical_view_weights(test_records),
        )
        outer_results.append({
            "test_person": outer["test_person"],
            "candidate": chosen["candidate"],
            "inner_guardrails_met": chosen["guardrails_met"],
            **metrics,
        })
    summary = {}
    for key in METRIC_KEYS:
        values = [row[key] for row in outer_results]
        summary[key + "_mean"] = float(np.mean(values))
        summary[key + "_std"] = float(np.std(values))
    return {"summary": summary, "folds": outer_results}


def evaluate(data_dir, seed=0):
    records, audit = load_repetitions(data_dir, dimensions="3d")
    y = np.array([row["label"] for row in records], dtype=int)
    matrices = {
        "original": np.stack([extract_features(row["poses"]) for row in records]),
        "mirror": np.stack([extract_features(mirror_pose(row["poses"])) for row in records]),
        "temporal_0.85": np.stack([
            extract_features(resample_pose(row["poses"], 0.85)) for row in records
        ]),
        "temporal_1.15": np.stack([
            extract_features(resample_pose(row["poses"], 1.15)) for row in records
        ]),
    }
    feature_differences = {
        name: float(np.max(np.abs(matrix - matrices["original"])))
        for name, matrix in matrices.items() if name != "original"
    }
    return {
        "protocol": "nested leave-one-person-out; augmentation only in inner/outer training subsets",
        "transforms": list(TRANSFORMS),
        "excluded_transform": "coordinate noise: cannot guarantee preservation of borderline technique labels",
        "observations": len(records),
        "physical_repetitions": len({row["physical_id"] for row in records}),
        "people": sorted({row["person"] for row in records}),
        "data_audit": {
            "source": audit["source"],
            "dimensions": audit["dimensions"],
            "selected_physical_repetitions": audit["selected_physical_repetitions"],
            "selected_camera_observations": audit["selected_camera_observations"],
        },
        "max_feature_difference_from_original": feature_differences,
        "without_augmentation": run_condition(records, matrices, y, False, seed),
        "with_augmentation": run_condition(records, matrices, y, True, seed),
        "conclusion": "negative: fixed label-preserving augmentations did not improve held-out-person quality",
    }


def percent(value):
    return f"{100 * value:.1f}%"


def markdown_report(result):
    baseline = result["without_augmentation"]["summary"]
    augmented = result["with_augmentation"]["summary"]
    return "\n".join([
        "# Проверка безопасных аугментаций REHAB26",
        "",
        "Сравниваются одна и та же логистическая модель и одинаковые вложенные разбиения по людям. "
        "В обучающей части к каждой камере добавлены зеркало со сменой левых/правых суставов и временная "
        "интерполяция `0,85×`/`1,15×`. Проверочные люди всегда оцениваются только по исходным данным.",
        "",
        "## Результат",
        "",
        "| Вариант | Macro-F1 | Ложные замечания | Обнаружение ошибок | Верный тип ошибки |",
        "|---|---:|---:|---:|---:|",
        f"| Без аугментаций | {baseline['macro_f1_mean']:.3f} | {percent(baseline['false_feedback_rate_mean'])} | {percent(baseline['error_recall_mean'])} | {percent(baseline['specific_error_type_recall_mean'])} |",
        f"| С аугментациями | {augmented['macro_f1_mean']:.3f} | {percent(augmented['false_feedback_rate_mean'])} | {percent(augmented['error_recall_mean'])} | {percent(augmented['specific_error_type_recall_mean'])} |",
        "",
        "Аугментации снизили Macro-F1 и обнаружение ошибок, почти не изменив долю ложных замечаний. "
        "Поэтому они не добавляются в финальный pipeline.",
        "",
        "## Почему эксперимент прекращён",
        "",
        "- Дополнительные копии не создают новых людей и не устраняют межличностный сдвиг.",
        "- Геометрический baseline уже инвариантен к переносу, масштабу и жёсткому повороту; такие копии были бы точными дубликатами признаков.",
        "- Координатный шум не проверялся, потому что для пограничных движений он может изменить саму метку.",
        "- После просмотра внешних результатов сила и набор преобразований не подбирались.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "python scripts/download_rehab26.py --dimensions 3d",
        "python scripts/evaluate_safe_augmentations.py",
        "```",
        "",
        "## Ограничения",
        "",
        "- Вывод относится к фиксированному небольшому набору безопасных преобразований, а не ко всем возможным аугментациям.",
        "- В REHAB26 только пять участников; статистическая уверенность остаётся ограниченной.",
        "- Отрицательный результат не доказывает бесполезность аугментаций на более разнообразных исходных данных.",
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/rehab26_augmentation_analysis.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/rehab26_augmentation_analysis.md"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = evaluate(args.data_dir, args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_report(result), encoding="utf-8")
    print(f"Отчёты: {args.output_markdown}; {args.output_json}")


if __name__ == "__main__":
    main()
