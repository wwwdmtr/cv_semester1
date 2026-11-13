"""Вложенное CPU-сравнение исходной 35D и фиксированной 42D геометрии.

Вычисление признаков не обучается и не использует классы. Классификаторы,
масштабирование, class weights и выбор параметров видят только train-людей.
Обёртка отвечает за предварительный снимок протокола и журнал запуска.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from rehab26_baseline import (
    artifact_bundle, classification_metrics, fit_pipeline, meets_guardrails,
    person_folds, physical_view_weights, save_verified_bundle,
    search_candidates, select_candidate, view_bin, write_json, write_predictions,
)
from rehab26_phase_features import (
    FEATURE_METADATA, FEATURE_NAMES, FEATURE_VERSION, extract_features,
)
from rehab26_features import FEATURE_VERSION as BASE_FEATURE_VERSION


PROTOCOL_VERSION = "rehab26_phase_relative_geometry_v1"
METHODS = ("global35", "phase_relative42")
METRIC_KEYS = ("macro_f1", "false_feedback_rate", "error_recall", "correct_recall",
               "specific_error_type_recall", "specific_remark_precision")


def candidates():
    return [
        {"id": f"logreg_C{c:g}_{'balanced' if balanced else 'unweighted'}",
         "family": "logreg", "C": c, "balanced": balanced}
        for c in (0.1, 1.0, 10.0) for balanced in (False, True)
    ]


def validate_inputs(records, x35, y):
    x35, y = np.asarray(x35), np.asarray(y)
    if x35.shape != (len(records), 35) or not np.isfinite(x35).all():
        raise ValueError("Нужны конечные исходные 35D для каждого наблюдения")
    if y.shape != (len(records),) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Нужны все три целевые категории")
    groups = defaultdict(list)
    for index, row in enumerate(records):
        if row["label"] != y[index] or not np.isfinite(float(row["view_angle"])):
            raise ValueError("Несогласованная метка или неконечный угол")
        groups[row["physical_id"]].append(row)
    for physical_id, rows in groups.items():
        if len(rows) != 3 or {row["camera"] for row in rows} != {"cam1", "cam2", "cam3"}:
            raise ValueError(f"Нужны ровно три разных камеры: {physical_id}")
        if len({row["person"] for row in rows}) != 1 or len({row["label"] for row in rows}) != 1:
            raise ValueError(f"Разные люди или метки одного повтора: {physical_id}")


def prepare_phase_matrix(records, x35, output):
    """Не фильтровать сбои: сохранить все причины и прервать опыт."""
    rows, failures = [], []
    for index, record in enumerate(records):
        try:
            vector = extract_features(record["poses"])
            if not np.array_equal(vector[:35], x35[index]):
                raise ValueError("baseline_prefix_not_exactly_equal_to_prepared_35D")
            rows.append(vector)
        except (ValueError, FloatingPointError, KeyError) as exc:
            failures.append({
                "observation_index": index, "person": record["person"],
                "physical_id": record["physical_id"], "camera": record["camera"], "reason": str(exc),
            })
    write_json(output / "phase_feature_failures.json", failures)
    if failures:
        raise ValueError(f"42D не рассчитаны для {len(failures)} наблюдений; исключения запрещены")
    x42 = np.stack(rows)
    if x42.shape != (len(records), 42) or not np.isfinite(x42).all():
        raise ValueError("Неконечная или неверная матрица 42D")
    metadata = {"version": FEATURE_VERSION, "names": list(FEATURE_NAMES), "details": FEATURE_METADATA}
    write_json(output / "phase_feature_metadata.json", metadata)
    write_json(output / "phase_feature_matrix_metadata.json", {
        "shape": list(x42.shape), "dtype": str(x42.dtype),
        "feature_matrix_sha256": hashlib.sha256(x42.tobytes(order="C")).hexdigest(),
        "baseline_prefix_exact": True, "feature_version": FEATURE_VERSION,
        "observations_excluded": 0,
    })
    return x42, metadata


def detailed_metrics(y, prediction, weights):
    result = classification_metrics(y, prediction, weights)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=[0, 1, 2], sample_weight=weights, zero_division=0)
    result["per_class"] = {
        str(label): {"precision": float(precision[label]), "recall": float(recall[label]),
                     "f1": float(f1[label]), "physical_weighted_support": float(support[label])}
        for label in (0, 1, 2)
    }
    return result


def summarize(outer_metrics, predictions):
    summaries = []
    for method in METHODS:
        folds = [row for row in outer_metrics if row["method"] == method]
        summary = {"method": method, "n_test_people": len(folds),
                   "feature_version": BASE_FEATURE_VERSION if method == "global35" else FEATURE_VERSION}
        for metric in METRIC_KEYS:
            values = [row[metric] for row in folds if row[metric] is not None]
            summary[metric + "_mean"] = float(np.mean(values)) if values else None
            summary[metric + "_std"] = float(np.std(values)) if values else None
            summary[metric + "_worst"] = (
                float(max(values) if metric == "false_feedback_rate" else min(values)) if values else None)
        rows = [row for row in predictions if row["method"] == method]
        summary["pooled_repeat_weighted"] = detailed_metrics(
            [row["true_label"] for row in rows], [row["predicted_class"] for row in rows],
            [row["view_weight"] for row in rows])
        summary["mean_guardrails_met"] = meets_guardrails(
            summary["false_feedback_rate_mean"], summary["error_recall_mean"])
        summary["all_people_guardrails_met"] = all(
            meets_guardrails(row["false_feedback_rate"], row["error_recall"]) for row in folds)
        summaries.append(summary)
    return summaries


def run(records, x35, y, output, models_dir, feature_metadata, dataset_metadata):
    """Сравнить два extractor на результате baseline.prepare; вернуть metrics.

    Только десять outer-fold bundles, без финального refit и замены deployment.
    output допускает уже записанные prepare/config/protocol артефакты обёртки.
    """
    output, models_dir = Path(output), Path(models_dir)
    artifacts = ("phase_feature_failures.json", "phase_feature_metadata.json",
                 "phase_feature_matrix_metadata.json", "inner_search.json",
                 "outer_fold_metrics.json", "fold_predictions.csv", "group_metrics.json",
                 "metrics.json", "runtime.json")
    if any((output / name).exists() for name in artifacts):
        raise ValueError("Каталог уже содержит результаты phase experiment")
    if models_dir.exists() and any(models_dir.iterdir()):
        raise ValueError("Каталог моделей должен быть пустым")
    output.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validate_inputs(records, x35, y)
    x35, y = np.asarray(x35), np.asarray(y, dtype=int)
    folds = person_folds(records, required_people=5)
    x42, phase_metadata = prepare_phase_matrix(records, x35, output)
    matrices = {"global35": x35, "phase_relative42": x42}
    metadata = {"global35": feature_metadata, "phase_relative42": phase_metadata}
    searches, outer_metrics, predictions, group_metrics = [], [], [], []
    specs = candidates()
    for fold in folds:
        train, test = np.asarray(fold["train"]), np.asarray(fold["test"])
        train_rows, test_rows = [records[i] for i in train], [records[i] for i in test]
        train_weights, test_weights = physical_view_weights(train_rows), physical_view_weights(test_rows)
        for method in METHODS:
            print(f"Внешний {fold['test_person']}; {method}: подбор только на {', '.join(fold['train_people'])}", flush=True)
            x = matrices[method]
            entry = {"outer_test_person": fold["test_person"], "method": method,
                     "train_people": fold["train_people"], "search": []}
            searches.append(entry)
            # Используем исходную процедуру; сохраняем каждый завершённый
            # кандидат до перехода к следующему и до любых outer predictions.
            search = []
            for spec in specs:
                search.extend(search_candidates(x[train], y[train], train_rows, [spec]))
                entry["search"] = list(search)
                write_json(output / "inner_search.json", searches)
            chosen = select_candidate(search)
            spec = chosen["candidate"]
            entry["chosen"] = spec
            entry["selection_complete_before_outer_prediction"] = True
            write_json(output / "inner_search.json", searches)
            fitting_started = time.perf_counter()
            pipeline = fit_pipeline(spec, x[train], y[train], train_weights)
            fit_seconds = time.perf_counter() - fitting_started
            predicted, probabilities = pipeline.predict(x[test]), pipeline.predict_proba(x[test])
            if (list(pipeline.classes_) != [0, 1, 2] or probabilities.shape != (len(test), 3)
                    or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                    or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-10)):
                raise ValueError("Некорректные вероятности модели")
            outer_metrics.append({
                "method": method, "test_person": fold["test_person"], "candidate": spec,
                "fit_seconds": fit_seconds, **detailed_metrics(y[test], predicted, test_weights),
            })
            bundle_dataset = dict(dataset_metadata)
            bundle_dataset["feature_matrix_sha256"] = hashlib.sha256(x.tobytes(order="C")).hexdigest()
            bundle = artifact_bundle(pipeline, spec, metadata[method], bundle_dataset,
                                     train_rows, "outer_evaluation_fold", fold["test_person"])
            bundle.update({"protocol_version": PROTOCOL_VERSION, "method": method,
                           "input_feature_dimension": int(x.shape[1]),
                           "feature_extractor_module": "rehab26_features" if method == "global35" else "rehab26_phase_features",
                           "deployment_replacement": False})
            save_verified_bundle(models_dir / "folds" / f"{method}_heldout_{fold['test_person']}.joblib",
                                 bundle, x[test])
            for index, record in enumerate(test_rows):
                predictions.append({
                    "method": method, "candidate_id": spec["id"], "person": record["person"],
                    "physical_id": record["physical_id"], "camera": record["camera"],
                    "orientation": record.get("orientation", ""), "view_angle": record["view_angle"],
                    "view_bin": view_bin(record["view_angle"]), "true_label": int(y[test[index]]),
                    "predicted_class": int(predicted[index]), "view_weight": float(test_weights[index]),
                    "confidence_score": float(probabilities[index].max()),
                    "confidence_kind": "max_predict_proba_uncalibrated",
                    "class_scores": json.dumps(probabilities[index].tolist()),
                })
            for dimension in ("camera", "view_bin", "view_angle"):
                values = [view_bin(row["view_angle"]) if dimension == "view_bin" else row[dimension]
                          for row in test_rows]
                for value in sorted(set(values)):
                    indices = np.array([i for i, item in enumerate(values) if item == value])
                    subset = [test_rows[i] for i in indices]
                    group_metrics.append({
                        "method": method, "test_person": fold["test_person"],
                        "grouping": dimension, "group": value,
                        **detailed_metrics(y[test][indices], predicted[indices], physical_view_weights(subset)),
                    })
            write_json(output / "outer_fold_metrics.json", outer_metrics)
            write_predictions(output / "fold_predictions.csv", predictions)
            write_json(output / "group_metrics.json", group_metrics)
    metrics = {
        "protocol_version": PROTOCOL_VERSION, "summaries": summarize(outer_metrics, predictions),
        "outer_folds": outer_metrics,
        "note": "Исследовательское сравнение одного фиксированного рецепта 42D с исходными 35D; без новой архитектуры, visibility, отказов и финального refit.",
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "runtime.json", {
        "seconds": time.perf_counter() - started, "saved_fold_bundles": len(outer_metrics),
        "saved_final_bundles": 0, "roundtrip_verified": True, "gpu_used": False,
        "feature_dimensions": {"global35": 35, "phase_relative42": 42},
    })
    return metrics
