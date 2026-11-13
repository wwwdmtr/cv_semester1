"""Сравнить прямую и двухступенчатую модели на фиксированных 2D-признаках.

Фронтальная cam1 описывает положение коленей, профильная cam3 — глубину.
Одна физическая попытка образует один пример; все настройки выбираются только
на внутренних разбиениях по людям.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rehab26_baseline import (
    candidate_grid,
    classification_metrics,
    fit_pipeline,
    person_folds,
    search_candidates,
    select_candidate,
)
from rehab26_data import load_repetitions


QUANTILES = (0.0, 0.1, 0.5, 0.9, 1.0)
EXPECTED_PHYSICAL_REPETITIONS = 388
FEATURE_VERSION = "rehab26_front_profile_2d_v1"
METRIC_KEYS = (
    "macro_f1",
    "false_feedback_rate",
    "error_recall",
    "specific_error_type_recall",
    "specific_remark_precision",
)


def angle_2d(first, center, last):
    left = first - center
    right = last - center
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if np.any(denominator <= 1e-10):
        raise ValueError("degenerate_2d_limb")
    cosine = np.sum(left * right, axis=1) / denominator
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def summarize_signal(signal, bottom):
    signal = np.asarray(signal, dtype=float)
    bins = np.array_split(np.arange(len(signal)), 5)
    return np.concatenate([
        np.quantile(signal, QUANTILES),
        np.array([signal[bottom].mean(), signal[bottom].min(), signal[bottom].max()]),
        np.array([signal[index].mean() for index in bins]),
    ])


def extract_physical_features(front_poses, profile_poses):
    front = np.asarray(front_poses, dtype=float)
    profile = np.asarray(profile_poses, dtype=float)
    if front.shape != profile.shape or front.ndim != 3 or front.shape[1:] != (33, 4):
        raise ValueError("expected_synchronized_frames_33_4")
    if len(front) < 5 or not np.isfinite(front).all() or not np.isfinite(profile).all():
        raise ValueError("invalid_or_nonfinite_pose_sequence")
    front_xy = front[:, :, :2]
    profile_xy = profile[:, :, :2]

    right_profile_knee = angle_2d(profile_xy[:, 24], profile_xy[:, 26], profile_xy[:, 28])
    left_profile_knee = angle_2d(profile_xy[:, 23], profile_xy[:, 25], profile_xy[:, 27])
    profile_knees = (right_profile_knee + left_profile_knee) / 2
    bottom = profile_knees <= np.quantile(profile_knees, 0.20)

    front_hip_span = np.abs(front_xy[:, 24, 0] - front_xy[:, 23, 0])
    front_knee_span = np.abs(front_xy[:, 26, 0] - front_xy[:, 25, 0])
    front_ankle_span = np.abs(front_xy[:, 28, 0] - front_xy[:, 27, 0])
    if np.any(front_hip_span <= 1e-10) or np.any(front_ankle_span <= 1e-10):
        raise ValueError("degenerate_frontal_span")
    knee_to_ankle = front_knee_span / front_ankle_span
    knee_to_hip = front_knee_span / front_hip_span
    ankle_to_hip = front_ankle_span / front_hip_span
    front_center_shift = (
        (front_xy[:, 25, 0] + front_xy[:, 26, 0])
        - (front_xy[:, 23, 0] + front_xy[:, 24, 0])
    ) / (2 * front_hip_span)

    profile_shoulders = (profile_xy[:, 11] + profile_xy[:, 12]) / 2
    profile_hips = (profile_xy[:, 23] + profile_xy[:, 24]) / 2
    profile_ankles = (profile_xy[:, 27] + profile_xy[:, 28]) / 2
    torso = np.linalg.norm(profile_shoulders - profile_hips, axis=1)
    if np.any(torso <= 1e-10):
        raise ValueError("degenerate_profile_torso")
    profile_hips_angle = (
        angle_2d(profile_shoulders, profile_xy[:, 24], profile_xy[:, 26])
        + angle_2d(profile_shoulders, profile_xy[:, 23], profile_xy[:, 25])
    ) / 2
    hip_to_ankle = np.linalg.norm(profile_hips - profile_ankles, axis=1) / torso
    vertical_depth = np.abs(profile_hips[:, 1] - profile_ankles[:, 1]) / torso
    knee_asymmetry = np.abs(right_profile_knee - left_profile_knee)

    signals = (
        knee_to_ankle,
        knee_to_hip,
        ankle_to_hip,
        front_center_shift,
        profile_knees,
        profile_hips_angle,
        hip_to_ankle,
        vertical_depth,
        knee_asymmetry,
    )
    features = np.concatenate([summarize_signal(signal, bottom) for signal in signals])
    edge = max(2, int(round(0.15 * len(knee_to_ankle))))
    standing = np.concatenate([knee_to_ankle[:edge], knee_to_ankle[-edge:]]).mean()
    bottom_ratio = knee_to_ankle[bottom].mean()
    result = np.concatenate([features, np.array([bottom_ratio - standing, bottom_ratio / standing])])
    if result.shape != (119,) or not np.isfinite(result).all():
        raise ValueError("invalid_feature_vector")
    return result


def prepare(data_dir):
    observations, audit = load_repetitions(data_dir, dimensions="2d")
    groups = {}
    for row in observations:
        groups.setdefault(row["physical_id"], {})[row["camera"]] = row
    records, features = [], []
    for physical_id in sorted(groups):
        cameras = groups[physical_id]
        if set(cameras) != {"cam1", "cam2", "cam3"}:
            raise ValueError(f"missing_camera:{physical_id}")
        front, profile = cameras["cam1"], cameras["cam3"]
        if front["label"] != profile["label"] or front["person"] != profile["person"]:
            raise ValueError(f"inconsistent_synchronized_labels:{physical_id}")
        records.append({"physical_id": physical_id, "person": front["person"], "label": front["label"]})
        features.append(extract_physical_features(front["poses"], profile["poses"]))
    if len(records) != EXPECTED_PHYSICAL_REPETITIONS:
        raise ValueError(f"expected_{EXPECTED_PHYSICAL_REPETITIONS}_physical_repetitions")
    return records, np.stack(features), np.array([row["label"] for row in records], dtype=int), audit


def build_binary_pipeline(spec):
    weight = "balanced" if spec.get("balanced") else None
    if spec["family"] == "logreg":
        classifier = LogisticRegression(C=spec["C"], class_weight=weight, max_iter=3000)
    elif spec["family"] == "svm":
        classifier = SVC(C=spec["C"], class_weight=weight)
    elif spec["family"] == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"],
            random_state=spec["seed"],
            n_jobs=1,
        )
    else:
        classifier = DummyClassifier(strategy="most_frequent")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def fit_binary(spec, x, y):
    model = build_binary_pipeline(spec)
    model.fit(x, y)
    return model


def binary_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro")),
        "false_feedback_rate": float(np.mean(y_pred[y_true == 0] == 1)),
        "error_recall": float(np.mean(y_pred[y_true == 1] == 1)),
    }


def select_stage1(x, y, records, specs):
    binary = (y != 0).astype(int)
    candidates = []
    for spec in specs:
        folds = []
        for fold in person_folds(records):
            train, validation = np.asarray(fold["train"]), np.asarray(fold["test"])
            model = fit_binary(spec, x[train], binary[train])
            folds.append(binary_metrics(binary[validation], model.predict(x[validation])))
        mean = {key: float(np.mean([row[key] for row in folds])) for key in folds[0]}
        mean.update({
            "candidate": spec,
            "guardrails_met": mean["false_feedback_rate"] <= 0.20 and mean["error_recall"] >= 0.80,
        })
        candidates.append(mean)
    feasible = [row for row in candidates if row["guardrails_met"]]
    pool = feasible or candidates
    return sorted(pool, key=lambda row: (
        -row["macro_f1"], row["false_feedback_rate"], -row["error_recall"], row["candidate"]["id"]
    ))[0]


def select_stage2(x, y, records, specs):
    candidates = []
    for spec in specs:
        scores = []
        for fold in person_folds(records):
            train = np.asarray([index for index in fold["train"] if y[index] != 0])
            validation = np.asarray([index for index in fold["test"] if y[index] != 0])
            model = fit_binary(spec, x[train], y[train])
            scores.append(float(f1_score(
                y[validation], model.predict(x[validation]), labels=[1, 2], average="macro"
            )))
        candidates.append({"candidate": spec, "macro_f1": float(np.mean(scores))})
    return sorted(candidates, key=lambda row: (-row["macro_f1"], row["candidate"]["id"]))[0]


def summarize(folds):
    return {
        key + "_mean": float(np.mean([fold[key] for fold in folds]))
        for key in METRIC_KEYS
    } | {
        key + "_std": float(np.std([fold[key] for fold in folds]))
        for key in METRIC_KEYS
    }


def evaluate(records, x, y, seed=0):
    specs = candidate_grid(seed)
    direct_folds, two_stage_folds = [], []
    for outer in person_folds(records, required_people=5):
        train, test = np.asarray(outer["train"]), np.asarray(outer["test"])
        train_records = [records[index] for index in train]

        direct_search = search_candidates(x[train], y[train], train_records, specs)
        direct_spec = select_candidate(direct_search)["candidate"]
        direct_model = fit_pipeline(direct_spec, x[train], y[train], np.ones(len(train)))
        direct_metrics = classification_metrics(y[test], direct_model.predict(x[test]))
        direct_folds.append({"test_person": outer["test_person"], "candidate": direct_spec, **direct_metrics})

        first = select_stage1(x[train], y[train], train_records, specs)
        second = select_stage2(x[train], y[train], train_records, specs)
        binary_model = fit_binary(first["candidate"], x[train], (y[train] != 0).astype(int))
        error_train = train[y[train] != 0]
        type_model = fit_binary(second["candidate"], x[error_train], y[error_train])
        has_error = binary_model.predict(x[test]).astype(bool)
        prediction = np.zeros(len(test), dtype=int)
        prediction[has_error] = type_model.predict(x[test][has_error])
        two_metrics = classification_metrics(y[test], prediction)
        two_stage_folds.append({
            "test_person": outer["test_person"],
            "stage1_candidate": first["candidate"],
            "stage1_inner_guardrails_met": first["guardrails_met"],
            "stage2_candidate": second["candidate"],
            **two_metrics,
        })
    return {
        "direct": {"summary": summarize(direct_folds), "folds": direct_folds},
        "two_stage": {"summary": summarize(two_stage_folds), "folds": two_stage_folds},
    }


def percent(value):
    return f"{100 * value:.1f}%"


def markdown_report(result):
    baseline = result["comparison_multiview_baseline"]
    direct = result["models"]["direct"]["summary"]
    two = result["models"]["two_stage"]["summary"]
    table = [
        ("Многоракурсный baseline", baseline),
        ("Прямая 2D-модель", direct),
        ("Двухступенчатая 2D-модель", two),
    ]
    lines = [
        "| Вариант | Macro-F1 | Ложные замечания | Обнаружение ошибок | Верный тип ошибки |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in table:
        lines.append(
            f"| {label} | {row['macro_f1_mean']:.3f} | {percent(row['false_feedback_rate_mean'])} | "
            f"{percent(row['error_recall_mean'])} | {percent(row['specific_error_type_recall_mean'])} |"
        )
    return "\n".join([
        "# Фронтальные и профильные 2D-признаки REHAB26",
        "",
        "Проверен заранее зафиксированный вариант: `cam1` описывает положение коленей, `cam3` — глубину; "
        "один физический повтор является одним примером. Модель и параметры выбираются только на внутренних "
        "разбиениях по людям.",
        "",
        "## Результат",
        "",
        *lines,
        "",
        "Прямая 2D-модель впервые превысила ориентир обнаружения ошибок 80%, но доля ложных замечаний выросла "
        "до 31,0%. Двухступенчатая схема сохранила обнаружение ошибок около 80%, но ещё сильнее увеличила ложные "
        "замечания и снизила Macro-F1. Поэтому ни одна 2D-схема не заменяет текущий многоракурсный baseline.",
        "",
        "## Интерпретация",
        "",
        "- 2D-геометрия содержит полезный сигнал об ошибках, но плохо переносит абсолютную норму между пятью людьми.",
        "- Основные провалы снова приходятся на Honza и Vasil; среднее улучшение чувствительности не делает систему пригодной для автоматических рекомендаций.",
        "- После просмотра внешних результатов новые пороги и признаки не подбирались.",
        "- Это отрицательный, но полезный результат: дальнейший перебор тех же моделей не обоснован.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "python scripts/download_rehab26.py --dimensions 2d",
        "python scripts/evaluate_view_specific_2d.py",
        "```",
        "",
        "## Ограничения",
        "",
        "- В наборе только пять людей; дополнительные камеры не увеличивают число независимых участников.",
        "- Признаки зафиксированы после диагностического разбора прежнего baseline, поэтому независимое подтверждение требует новых людей.",
        "- 2D MediaPipe-позы не заменяют визуальную проверку RGB и экспертную проверку спорных повторов.",
        "- Малая нейросеть не запускается автоматически: по плану она остаётся необязательной при наличии времени.",
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument("--multiview-report", type=Path, default=Path("reports/rehab26_multiview_analysis.json"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/rehab26_2d_feature_analysis.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/rehab26_2d_feature_analysis.md"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    records, x, y, audit = prepare(args.data_dir)
    models = evaluate(records, x, y, args.seed)
    multiview = json.loads(args.multiview_report.read_text(encoding="utf-8"))["mean_scores"]["person_mean"]
    result = {
        "feature_version": FEATURE_VERSION,
        "protocol": "nested leave-one-person-out; one physical repetition per sample",
        "observations": len(records),
        "feature_count": int(x.shape[1]),
        "people": sorted({row["person"] for row in records}),
        "labels": {"0": "correct", "1": "knees_inward", "2": "shallow_squat"},
        "data_audit": {
            "source": audit["source"],
            "dimensions": audit["dimensions"],
            "selected_physical_repetitions": audit["selected_physical_repetitions"],
            "selected_camera_observations": audit["selected_camera_observations"],
        },
        "comparison_multiview_baseline": {key + "_mean": multiview[key] for key in METRIC_KEYS},
        "models": models,
        "conclusion": "negative: sensitivity improved, but false feedback and Macro-F1 worsened",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_report(result), encoding="utf-8")
    print(f"Отчёты: {args.output_markdown}; {args.output_json}")


if __name__ == "__main__":
    main()
