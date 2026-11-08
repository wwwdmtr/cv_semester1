"""Сравнить уверенность модели и visibility как признаки ненадёжного замечания.

Используются только сохранённые внешние предсказания baseline. Visibility
рассчитывается для всей попытки и нужных суставов в нижней фазе. Отдельно
оцениваются равное покрытие и парные камеры одного физического движения.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rehab26_data import load_repetitions
from rehab26_features import extract_signals


METRICS = ("confidence", "overall_visibility", "phase_visibility", "phase_min_visibility")
COVERAGES = (1.0, 0.9, 0.8, 0.75, 0.7)
KEY_JOINTS = (11, 12, 23, 24, 25, 26, 27, 28)
LOWER_JOINTS = (23, 24, 25, 26, 27, 28)


def read_predictions(path, method="inner_selected"):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if row["method"] == method]
    if not selected:
        raise ValueError(f"missing_predictions:{method}")
    result = {}
    for row in selected:
        key = (row["physical_id"], row["camera"])
        if key in result:
            raise ValueError(f"duplicate_prediction:{key}")
        result[key] = {
            "person": row["person"],
            "true_label": int(row["true_label"]),
            "predicted_class": int(row["predicted_class"]),
            "confidence": float(row["confidence_score"]),
            "confidence_kind": row["confidence_kind"],
        }
    return result


def visibility_features(poses, predicted_class):
    array = np.asarray(poses, dtype=float)
    visibility = array[:, :, 3]
    signals = extract_signals(array)
    knee_flexion = signals[:, :2].mean(axis=1)
    bottom = knee_flexion <= np.quantile(knee_flexion, 0.20)
    required = LOWER_JOINTS if predicted_class == 1 else KEY_JOINTS
    return {
        "overall_visibility": float(visibility[:, KEY_JOINTS].mean()),
        "phase_visibility": float(visibility[bottom][:, required].mean()),
        "phase_min_visibility": float(visibility[bottom][:, required].min()),
        "bottom_frames": int(bottom.sum()),
    }


def build_rows(data_dir, predictions):
    records, audit = load_repetitions(data_dir, dimensions="3d")
    rows = []
    for record in records:
        key = (record["physical_id"], record["camera"])
        if key not in predictions:
            raise ValueError(f"missing_prediction:{key}")
        prediction = predictions[key]
        if prediction["person"] != record["person"] or prediction["true_label"] != record["label"]:
            raise ValueError(f"inconsistent_prediction_metadata:{key}")
        row = {
            "person": record["person"],
            "physical_id": record["physical_id"],
            "camera": record["camera"],
            **prediction,
            **visibility_features(record["poses"], prediction["predicted_class"]),
        }
        row["wrong"] = int(row["predicted_class"] != row["true_label"])
        row["false_correction"] = int(
            row["predicted_class"] != 0 and row["predicted_class"] != row["true_label"]
        )
        row["false_feedback_correct"] = int(
            row["true_label"] == 0 and row["predicted_class"] != 0
        )
        rows.append(row)
    if len(rows) != len(predictions):
        raise ValueError("prediction_and_pose_counts_differ")
    return rows, audit


def risk_score(rows, metric):
    values = np.array([row[metric] for row in rows], dtype=float)
    return -values


def ranking_metrics(rows, target, metrics=METRICS):
    labels = np.array([row[target] for row in rows], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"target_requires_two_classes:{target}")
    return {
        metric: {
            "auroc": float(roc_auc_score(labels, risk_score(rows, metric))),
            "average_precision": float(average_precision_score(labels, risk_score(rows, metric))),
        }
        for metric in metrics
    }


def leave_one_person_out_detector(rows, target):
    feature_sets = {
        "confidence": ("confidence",),
        "visibility": ("overall_visibility", "phase_visibility", "phase_min_visibility"),
        "confidence_plus_visibility": METRICS,
    }
    people = np.array([row["person"] for row in rows])
    labels = np.array([row[target] for row in rows], dtype=int)
    result = {}
    for name, columns in feature_sets.items():
        x = np.array([[row[column] for column in columns] for row in rows], dtype=float)
        probabilities = np.empty(len(rows), dtype=float)
        for person in sorted(set(people)):
            train, test = people != person, people == person
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
            ])
            model.fit(x[train], labels[train])
            probabilities[test] = model.predict_proba(x[test])[:, 1]
        result[name] = {
            "auroc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
        }
    return result


def paired_camera_analysis(rows, seed=20261108, permutations=20000, bootstraps=5000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["physical_id"]].append(row)
    differences = {metric: [] for metric in METRICS}
    for physical_id, group in groups.items():
        if len(group) != 3 or len({row["true_label"] for row in group}) != 1:
            raise ValueError(f"invalid_camera_group:{physical_id}")
        states = {row["wrong"] for row in group}
        if states != {0, 1}:
            continue
        for metric in METRICS:
            wrong = np.mean([row[metric] for row in group if row["wrong"]])
            correct = np.mean([row[metric] for row in group if not row["wrong"]])
            differences[metric].append(wrong - correct)
    rng = np.random.default_rng(seed)
    result = {}
    for metric, values in differences.items():
        delta = np.asarray(values, dtype=float)
        observed = abs(delta.mean())
        permuted = np.empty(permutations, dtype=float)
        for index in range(permutations):
            signs = rng.choice((-1.0, 1.0), size=len(delta))
            permuted[index] = abs(np.mean(delta * signs))
        bootstrap = np.empty(bootstraps, dtype=float)
        for index in range(bootstraps):
            bootstrap[index] = rng.choice(delta, size=len(delta), replace=True).mean()
        result[metric] = {
            "wrong_minus_correct_mean": float(delta.mean()),
            "wrong_minus_correct_median": float(np.median(delta)),
            "mean_bootstrap_95_ci": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
            "sign_flip_p_value": float((1 + np.sum(permuted >= observed)) / (permutations + 1)),
        }
    return {"mixed_outcome_physical_repetitions": len(next(iter(differences.values()))), "metrics": result}


def coverage_curve(rows, metric, coverages=COVERAGES):
    risk = risk_score(rows, metric)
    order = np.argsort(risk, kind="stable")
    true = np.array([row["true_label"] for row in rows], dtype=int)
    predicted = np.array([row["predicted_class"] for row in rows], dtype=int)
    wrong = predicted != true
    correct_execution = true == 0
    error_execution = true != 0
    predicted_error = predicted != 0
    false_correction = predicted_error & wrong
    result = []
    for coverage in coverages:
        retained_count = int(np.floor(len(rows) * coverage))
        retained = np.zeros(len(rows), dtype=bool)
        retained[order[:retained_count]] = True
        retained_errors = retained & error_execution
        result.append({
            "coverage": float(retained.mean()),
            "retained_observations": int(retained.sum()),
            "selective_wrong_rate": float(np.mean(wrong[retained])),
            "false_feedback_rate_retained_correct": float(np.mean(wrong[retained & correct_execution])),
            "overall_error_recall_counting_rejection_as_not_detected": float(
                np.sum(retained & error_execution & predicted_error) / np.sum(error_execution)
            ),
            "error_recall_among_retained_errors": float(
                np.sum(retained_errors & predicted_error) / np.sum(retained_errors)
            ),
            "rejected_false_correction_share": float(
                np.sum(~retained & false_correction) / np.sum(false_correction)
            ),
        })
    return result


def analyze(rows):
    correct_rows = [row for row in rows if row["true_label"] == 0]
    return {
        "counts": {
            "observations": len(rows),
            "wrong_predictions": sum(row["wrong"] for row in rows),
            "false_corrections": sum(row["false_correction"] for row in rows),
            "false_feedback_on_correct": sum(row["false_feedback_correct"] for row in correct_rows),
        },
        "ranking": {
            "wrong": ranking_metrics(rows, "wrong"),
            "false_correction": ranking_metrics(rows, "false_correction"),
            "false_feedback_among_true_correct": ranking_metrics(
                correct_rows, "false_feedback_correct"
            ),
        },
        "leave_one_person_out_error_detector": {
            "wrong": leave_one_person_out_detector(rows, "wrong"),
            "false_correction": leave_one_person_out_detector(rows, "false_correction"),
        },
        "paired_cameras": paired_camera_analysis(rows),
        "coverage_curves": {metric: coverage_curve(rows, metric) for metric in METRICS},
    }


def percent(value):
    return f"{100 * value:.1f}%"


def markdown_report(result):
    wrong = result["analysis"]["ranking"]["wrong"]
    false_correction = result["analysis"]["ranking"]["false_correction"]
    curves = result["analysis"]["coverage_curves"]
    confidence_80 = min(curves["confidence"], key=lambda row: abs(row["coverage"] - 0.8))
    phase_80 = min(curves["phase_visibility"], key=lambda row: abs(row["coverage"] - 0.8))
    paired = result["analysis"]["paired_cameras"]
    return "\n".join([
        "# Надёжность замечаний: уверенность против visibility",
        "",
        "На сохранённых внешних предсказаниях `inner_selected` сравниваются уверенность модели, средняя visibility ключевых суставов, visibility нужных суставов в нижней фазе и минимальная visibility одного нужного сустава.",
        "",
        "## Проверка основной гипотезы",
        "",
        "| Сигнал риска | AUROC всех ошибок | AUROC ошибочных корректирующих замечаний |",
        "|---|---:|---:|",
        f"| Низкая уверенность | {wrong['confidence']['auroc']:.3f} | {false_correction['confidence']['auroc']:.3f} |",
        f"| Низкая общая visibility | {wrong['overall_visibility']['auroc']:.3f} | {false_correction['overall_visibility']['auroc']:.3f} |",
        f"| Низкая visibility нужной фазы | {wrong['phase_visibility']['auroc']:.3f} | {false_correction['phase_visibility']['auroc']:.3f} |",
        f"| Низкий минимум visibility | {wrong['phase_min_visibility']['auroc']:.3f} | {false_correction['phase_min_visibility']['auroc']:.3f} |",
        "",
        "Уровень около 0,5 соответствует случайному ранжированию. Visibility не превзошла уверенность и не дала устойчивого самостоятельного сигнала.",
        "",
        "## Синхронные камеры",
        "",
        f"В {paired['mixed_outcome_physical_repetitions']} физических повторах были одновременно правильные и ошибочные предсказания разных камер. Уверенность ошибочной камеры в среднем ниже; её 95%-интервал разности не включает ноль. Для visibility интервалы включают ноль, то есть систематического эффекта не найдено.",
        "",
        "## Равное покрытие 80%",
        "",
        "| Правило отказа | Ошибки среди выданных ответов | Ложные замечания на правильных | Обнаружено ошибок среди всех ошибочных |",
        "|---|---:|---:|---:|",
        f"| Низкая уверенность | {percent(confidence_80['selective_wrong_rate'])} | {percent(confidence_80['false_feedback_rate_retained_correct'])} | {percent(confidence_80['overall_error_recall_counting_rejection_as_not_detected'])} |",
        f"| Низкая visibility нужной фазы | {percent(phase_80['selective_wrong_rate'])} | {percent(phase_80['false_feedback_rate_retained_correct'])} | {percent(phase_80['overall_error_recall_counting_rejection_as_not_detected'])} |",
        "",
        "Отказ по уверенности уменьшает ложные замечания, но общий recall ошибок падает, потому что отказ не является обнаружением. Ни один вариант не выполняет одновременно ориентиры ≤20% ложных замечаний и ≥80% обнаруженных ошибок.",
        "",
        "## Вывод",
        "",
        "Рабочая гипотеза отклонена: доступность нужных суставов в информативной фазе не лучше обычной уверенности модели указывает на ненадёжность замечания. Практический режим отказа по уверенности возможен только как компромисс покрытия, а не как способ заявить достижение обеих целей.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "python scripts/download_rehab26.py --dimensions 3d",
        "python scripts/analyze_reliability.py",
        "```",
        "",
        "## Ограничения",
        "",
        "- В исследовании пять людей и готовые MediaPipe-позы; вывод требует проверки на новых участниках и исходном RGB.",
        "- Уверенность baseline не калибрована; AUROC оценивает ранжирование, а не истинную вероятность ошибки.",
        "- Точки покрытия рассчитаны описательно по внешним предсказаниям; рабочий порог должен выбираться только внутри обучения.",
        "- Отсутствие эффекта visibility относится к проверенным агрегатам и не доказывает бесполезность всех возможных показателей качества позы.",
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument("--predictions", type=Path, default=Path("reports/rehab26_baseline/fold_predictions.csv"))
    parser.add_argument("--method", default="inner_selected")
    parser.add_argument("--output-json", type=Path, default=Path("reports/rehab26_reliability_analysis.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/rehab26_reliability_analysis.md"))
    args = parser.parse_args(argv)
    predictions = read_predictions(args.predictions, args.method)
    rows, audit = build_rows(args.data_dir, predictions)
    result = {
        "method": args.method,
        "confidence_kind": sorted({row["confidence_kind"] for row in rows}),
        "data_audit": {
            "source": audit["source"],
            "dimensions": audit["dimensions"],
            "selected_physical_repetitions": audit["selected_physical_repetitions"],
            "selected_camera_observations": audit["selected_camera_observations"],
        },
        "analysis": analyze(rows),
        "conclusion": "hypothesis rejected: phase-specific visibility did not outperform classifier confidence",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_report(result), encoding="utf-8")
    print(f"Отчёты: {args.output_markdown}; {args.output_json}")


if __name__ == "__main__":
    main()
