"""Два ограниченных опыта: порядок движения и фиксированное объединение.

Внешние люди не участвуют в стандартизации, обучении и выборе кандидатов.
Перемешанная последовательность сохраняет те же наблюдения/значения, но
разрушает их порядок; синхронные камеры получают одну перестановку повтора.
Объединение вероятностей двух выбранных компонентов фиксировано 50/50.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from evaluate_phase_features import detailed_metrics, validate_inputs
from rehab26_baseline import (
    artifact_bundle, fit_pipeline, meets_guardrails, person_folds,
    physical_view_weights, save_verified_bundle, select_candidate, view_bin,
    write_json, write_predictions,
)
from rehab26_features import FEATURE_VERSION as BASE_FEATURE_VERSION
import rehab26_temporal as temporal


PROTOCOL_VERSION = "rehab26_temporal_order_and_fixed_fusion_v1"
METHODS = ("global35", "temporal_ordered", "temporal_shuffled", "geometry_temporal_fusion")
FUSION_ALPHA = 0.5
SHUFFLE_SEED = 1729
METRIC_KEYS = ("macro_f1", "false_feedback_rate", "error_recall", "correct_recall",
               "specific_error_type_recall", "specific_remark_precision")


def geometry_candidates():
    return [{"id": f"logreg_C{c:g}_{'balanced' if balanced else 'unweighted'}",
             "family": "logreg", "C": c, "balanced": balanced}
            for c in (0.1, 1.0, 10.0) for balanced in (False, True)]


def temporal_candidates():
    return [{"id": f"temporal_cnn_{'balanced' if balanced else 'unweighted'}",
             "family": "temporal_cnn", "balanced": balanced}
            for balanced in (False, True)]


def feature_versions():
    base = temporal.FEATURE_VERSION
    return {"global35": BASE_FEATURE_VERSION, "temporal_ordered": base,
            "temporal_shuffled": base + "_shuffled_control",
            "geometry_temporal_fusion": BASE_FEATURE_VERSION + "+" + base + "_mean50"}


def checked_probabilities(model, x, count):
    probabilities = np.asarray(model.predict_proba(x), dtype=float)
    if (list(model.classes_) != [0, 1, 2] or probabilities.shape != (count, 3)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-10, rtol=0)):
        raise ValueError("Некорректные вероятности или порядок классов")
    return probabilities


class FusionClassifier:
    """Повторно использовать два fitted-компонента, без дополнительного fit."""

    def __init__(self, geometry_model, temporal_model, alpha=FUSION_ALPHA):
        if alpha != FUSION_ALPHA:
            raise ValueError("В протоколе разрешено только фиксированное объединение 50/50")
        self.geometry_model = geometry_model
        self.temporal_model = temporal_model
        self.alpha = alpha
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, x):
        geometry, sequence = x
        if len(geometry) != len(sequence):
            raise ValueError("Несогласованные входы объединения")
        first = checked_probabilities(self.geometry_model, geometry, len(geometry))
        second = checked_probabilities(self.temporal_model, sequence, len(sequence))
        return (1 - self.alpha) * first + self.alpha * second

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)


def prepare_sequence_matrix(records, output):
    rows, failures = [], []
    for index, record in enumerate(records):
        try:
            rows.append(temporal.resample_signals(record["poses"]))
        except (ValueError, FloatingPointError, KeyError) as exc:
            failures.append({"observation_index": index, "person": record["person"],
                             "physical_id": record["physical_id"], "camera": record["camera"],
                             "reason": str(exc)})
    write_json(output / "temporal_feature_failures.json", failures)
    if failures:
        raise ValueError(f"Последовательности не получены: {len(failures)}; исключать наблюдения запрещено")
    ordered = np.stack(rows)
    if ordered.shape != (len(records), 64, 7) or not np.isfinite(ordered).all():
        raise ValueError("Ожидается конечная матрица (N,64,7)")
    ids = [record["physical_id"] for record in records]
    shuffled = temporal.transform_order(ordered, ids, shuffled=True, seed=SHUFFLE_SEED)
    if not np.array_equal(np.sort(ordered, axis=1), np.sort(shuffled, axis=1)):
        raise ValueError("Перестановка изменила значения сигналов")
    metadata = {"version": temporal.FEATURE_VERSION,
                "names": list(getattr(temporal, "SIGNAL_NAMES", ())),
                "details": getattr(temporal, "FEATURE_METADATA", {})}
    matrix_metadata = {
        "shape": list(ordered.shape), "dtype": str(ordered.dtype),
        "ordered_sha256": hashlib.sha256(ordered.tobytes(order="C")).hexdigest(),
        "shuffled_sha256": hashlib.sha256(shuffled.tobytes(order="C")).hexdigest(),
        "identical_sorted_values_per_channel": True, "observations_excluded": 0,
        "shuffle_seed": SHUFFLE_SEED, "shuffle_key": "physical_id",
        "same_permutation_for_synchronized_cameras": True,
        "same_permutation_across_channels": True, "feature_version": temporal.FEATURE_VERSION,
    }
    np.savez_compressed(output / "temporal_matrices.npz", ordered=ordered, shuffled=shuffled)
    write_json(output / "temporal_feature_metadata.json", metadata)
    write_json(output / "temporal_matrix_metadata.json", matrix_metadata)
    return ordered, shuffled, metadata, matrix_metadata


def _summary_search(candidate, fold_metrics, seconds):
    def mean(key):
        values = [row[key] for row in fold_metrics if row[key] is not None]
        return float(np.mean(values)) if values else None
    fpr, recall = mean("false_feedback_rate"), mean("error_recall")
    return {"candidate": candidate, "mean_macro_f1": mean("macro_f1"),
            "mean_false_feedback_rate": fpr, "mean_error_recall": recall,
            "guardrails_met": meets_guardrails(fpr, recall),
            "fit_seconds": seconds, "folds": fold_metrics}


def shared_temporal_selection(ordered_search, shuffled_search):
    """Один balanceflag для абляции: симметричная внутренняя оценка условий."""
    first = {row["candidate"]["balanced"]: row for row in ordered_search}
    second = {row["candidate"]["balanced"]: row for row in shuffled_search}
    if len(first) != len(ordered_search) or len(second) != len(shuffled_search) or first.keys() != second.keys():
        raise ValueError("Кандидаты двух временных условий не совпадают")
    combined = []
    for balance in sorted(first):
        ordered, shuffled = first[balance], second[balance]
        if ordered["candidate"] != shuffled["candidate"]:
            raise ValueError("Архитектура/настройки абляции должны быть одинаковы")
        candidate = dict(ordered["candidate"])
        means = {key: (ordered[key] + shuffled[key]) / 2 for key in (
            "mean_macro_f1", "mean_false_feedback_rate", "mean_error_recall")}
        combined.append({"candidate": candidate, **means,
                         "guardrails_met": meets_guardrails(means["mean_false_feedback_rate"], means["mean_error_recall"]),
                         "selection": "symmetric_mean_of_ordered_and_shuffled_inner_metrics",
                         "ordered_candidate_id": ordered["candidate"]["id"],
                         "shuffled_candidate_id": shuffled["candidate"]["id"]})
    return combined, select_candidate(combined)["candidate"]


def inner_search(x, y, records, specs, method, outer_person, output,
                 all_predictions, diagnostics):
    """OOF только внутри четырёх обучающих людей; параметры fit train-only."""
    results = []
    for spec in specs:
        metrics, fit_seconds = [], 0.
        for fold in person_folds(records):
            train, val = np.asarray(fold["train"]), np.asarray(fold["test"])
            training = [records[i] for i in train]
            validation = [records[i] for i in val]
            weights = physical_view_weights(training)
            started = time.perf_counter()
            model = (fit_pipeline(spec, x[train], y[train], weights) if method == "global35"
                     else temporal.fit_temporal(x[train], y[train], weights, spec))
            seconds = time.perf_counter() - started
            print(f"  {outer_person}; {method}; {spec['id']}; validation {fold['test_person']}: {seconds:.2f} с", flush=True)
            fit_seconds += seconds
            probabilities = checked_probabilities(model, x[val], len(val))
            predicted = np.argmax(probabilities, axis=1)
            metrics.append({"validation_person": fold["test_person"],
                            "train_people": fold["train_people"],
                            **detailed_metrics(y[val], predicted, physical_view_weights(validation))})
            if method != "global35":
                diagnostics.append({"scope": "inner_training", "outer_test_person": outer_person,
                                    "method": method, "candidate_id": spec["id"],
                                    "validation_person": fold["test_person"],
                                    "train_people": fold["train_people"], "seconds": seconds,
                                    "training_log": getattr(model, "training_log_", None)})
            val_weights = physical_view_weights(validation)
            for j, record in enumerate(validation):
                all_predictions.append({
                    "outer_test_person": outer_person, "method": method,
                    "candidate_id": spec["id"], "validation_person": fold["test_person"],
                    "train_people": json.dumps(fold["train_people"]), "person": record["person"],
                    "physical_id": record["physical_id"], "camera": record["camera"],
                    "true_label": int(y[val[j]]), "predicted_class": int(predicted[j]),
                    "view_weight": float(val_weights[j]),
                    "class_scores": json.dumps(probabilities[j].tolist()),
                })
        results.append(_summary_search(spec, metrics, fit_seconds))
        write_predictions(output / "inner_predictions.csv", all_predictions)
        write_json(output / "training_diagnostics.json", diagnostics)
    return results


def summarize(outer_metrics, predictions):
    summaries = []
    for method in METHODS:
        folds = [row for row in outer_metrics if row["method"] == method]
        summary = {"method": method, "n_test_people": len(folds),
                   "feature_version": feature_versions()[method]}
        for metric in METRIC_KEYS:
            values = [row[metric] for row in folds if row[metric] is not None]
            summary[metric + "_mean"] = float(np.mean(values)) if values else None
            summary[metric + "_std"] = float(np.std(values)) if values else None
            summary[metric + "_worst"] = (float(max(values) if metric == "false_feedback_rate" else min(values))
                                          if values else None)
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
    """Полный вложенный опыт, четыре условия; без финального refit/deployment."""
    output, models_dir = Path(output), Path(models_dir)
    artifacts = ("temporal_feature_failures.json", "temporal_feature_metadata.json",
                 "temporal_matrix_metadata.json", "temporal_matrices.npz", "inner_search.json", "inner_predictions.csv",
                 "training_diagnostics.json", "outer_fold_metrics.json", "fold_predictions.csv",
                 "group_metrics.json", "metrics.json", "runtime.json")
    if any((output / name).exists() for name in artifacts):
        raise ValueError("Каталог уже содержит результаты temporal-сравнения")
    if models_dir.exists() and any(models_dir.iterdir()):
        raise ValueError("Каталог моделей должен быть пустым")
    output.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validate_inputs(records, x35, y)
    x35, y = np.asarray(x35), np.asarray(y, dtype=int)
    ordered, shuffled, sequence_meta, matrix_meta = prepare_sequence_matrix(records, output)
    matrices = {"global35": x35, "temporal_ordered": ordered, "temporal_shuffled": shuffled}
    metadata = {"global35": feature_metadata,
                "temporal_ordered": sequence_meta,
                "temporal_shuffled": {**sequence_meta, "version": feature_versions()["temporal_shuffled"],
                                      "control": "fixed_physical_id_permutation_after_resampling"},
                "geometry_temporal_fusion": {
                    "version": feature_versions()["geometry_temporal_fusion"],
                    "names": ["geometry35", "ordered_sequence64x7"],
                    "details": {"fusion_alpha": FUSION_ALPHA, "rule": "mean_of_component_probabilities"}}}
    specs = {"global35": geometry_candidates(), "temporal_ordered": temporal_candidates(),
             "temporal_shuffled": temporal_candidates()}
    searches, outer_metrics, predictions, inner_predictions, diagnostics, groups = [], [], [], [], [], []
    total_outer_fit_seconds = 0.
    for fold in person_folds(records, required_people=5):
        train, test = np.asarray(fold["train"]), np.asarray(fold["test"])
        training, testing = [records[i] for i in train], [records[i] for i in test]
        weights, test_weights = physical_view_weights(training), physical_view_weights(testing)
        entry = {"outer_test_person": fold["test_person"], "train_people": fold["train_people"],
                 "searches": {}, "chosen": {}, "fusion_alpha": FUSION_ALPHA,
                 "fusion_weight_tuned": False}
        searches.append(entry)
        # Complete all choices, and persist them, before fitting/scoring outer models.
        for method in matrices:
            print(f"Внешний {fold['test_person']}; {method}: подбор на {', '.join(fold['train_people'])}", flush=True)
            entry["searches"][method] = inner_search(
                matrices[method][train], y[train], training, specs[method], method,
                fold["test_person"], output, inner_predictions, diagnostics)
            if method == "global35":
                entry["chosen"][method] = select_candidate(entry["searches"][method])["candidate"]
            write_json(output / "inner_search.json", searches)
        shared_search, shared_candidate = shared_temporal_selection(
            entry["searches"]["temporal_ordered"], entry["searches"]["temporal_shuffled"])
        entry["shared_temporal_selection"] = shared_search
        entry["chosen_common_temporal_candidate"] = shared_candidate
        entry["chosen"]["temporal_ordered"] = dict(shared_candidate)
        entry["chosen"]["temporal_shuffled"] = dict(shared_candidate)
        entry["selection_complete_before_outer_prediction"] = True
        write_json(output / "inner_search.json", searches)
        fitted, fit_times = {}, {}
        for method in matrices:
            x, spec = matrices[method], entry["chosen"][method]
            fit_started = time.perf_counter()
            fitted[method] = (fit_pipeline(spec, x[train], y[train], weights) if method == "global35"
                              else temporal.fit_temporal(x[train], y[train], weights, spec))
            fit_times[method] = time.perf_counter() - fit_started
            total_outer_fit_seconds += fit_times[method]
            if method != "global35":
                diagnostics.append({"scope": "outer_training", "outer_test_person": fold["test_person"],
                                    "method": method, "candidate_id": spec["id"],
                                    "train_people": fold["train_people"], "seconds": fit_times[method],
                                    "training_log": getattr(fitted[method], "training_log_", None)})
        fitted["geometry_temporal_fusion"] = FusionClassifier(fitted["global35"], fitted["temporal_ordered"])
        fit_times["geometry_temporal_fusion"] = 0.
        method_specs = dict(entry["chosen"])
        method_specs["geometry_temporal_fusion"] = {
            "id": "fixed_mean_geometry_temporal_50_50", "family": "fixed_probability_fusion",
            "alpha": FUSION_ALPHA, "geometry": entry["chosen"]["global35"],
            "temporal": entry["chosen"]["temporal_ordered"]}
        for method in METHODS:
            probe = (x35[test], ordered[test]) if method == "geometry_temporal_fusion" else matrices[method][test]
            probabilities = checked_probabilities(fitted[method], probe, len(test))
            predicted = np.argmax(probabilities, axis=1)
            outer_metrics.append({"method": method, "test_person": fold["test_person"],
                                  "candidate": method_specs[method], "fit_seconds": fit_times[method],
                                  **detailed_metrics(y[test], predicted, test_weights)})
            bundle = artifact_bundle(fitted[method], method_specs[method], metadata[method],
                                     dataset_metadata, training, "outer_evaluation_fold", fold["test_person"])
            bundle.update({"protocol_version": PROTOCOL_VERSION, "method": method,
                           "deployment_replacement": False, "confidence_calibrated": False,
                           "matrix_metadata": matrix_meta,
                           "input_format": ("tuple: 35D geometry, ordered 64x7 geometry sequence" if method == "geometry_temporal_fusion"
                                            else "35D global geometry" if method == "global35"
                                            else "64x7 geometry sequence; fixed physical_id permutation required for shuffled control" if method == "temporal_shuffled"
                                            else "ordered 64x7 geometry sequence"),
                           "input_feature_dimension": 35 if method == "global35" else [64, 7] if method != "geometry_temporal_fusion" else [35, [64, 7]],
                           "training_log": getattr(fitted[method], "training_log_", None)})
            bundle["versions"]["torch"] = str(temporal.torch.__version__)
            save_verified_bundle(models_dir / "folds" / f"{method}_heldout_{fold['test_person']}.joblib", bundle, probe)
            for j, record in enumerate(testing):
                predictions.append({"method": method, "candidate_id": method_specs[method]["id"],
                                    "person": record["person"], "physical_id": record["physical_id"],
                                    "camera": record["camera"], "orientation": record.get("orientation", ""),
                                    "view_angle": record["view_angle"], "view_bin": view_bin(record["view_angle"]),
                                    "true_label": int(y[test[j]]), "predicted_class": int(predicted[j]),
                                    "view_weight": float(test_weights[j]),
                                    "alpha": FUSION_ALPHA if method == "geometry_temporal_fusion" else "",
                                    "confidence_score": float(probabilities[j].max()),
                                    "confidence_kind": "max_predict_proba_uncalibrated",
                                    "class_scores": json.dumps(probabilities[j].tolist())})
            for dimension in ("camera", "view_bin", "view_angle"):
                values = [view_bin(row["view_angle"]) if dimension == "view_bin" else row[dimension] for row in testing]
                for value in sorted(set(values)):
                    indices = np.array([i for i, item in enumerate(values) if item == value])
                    subset = [testing[i] for i in indices]
                    groups.append({"method": method, "test_person": fold["test_person"],
                                   "grouping": dimension, "group": value,
                                   **detailed_metrics(y[test][indices], predicted[indices], physical_view_weights(subset))})
        write_json(output / "training_diagnostics.json", diagnostics)
        write_json(output / "outer_fold_metrics.json", outer_metrics)
        write_predictions(output / "fold_predictions.csv", predictions)
        write_json(output / "group_metrics.json", groups)
    metrics = {"protocol_version": PROTOCOL_VERSION, "summaries": summarize(outer_metrics, predictions),
               "outer_folds": outer_metrics,
               "note": "Два заранее ограниченных опыта: порядок в temporal CNN с shuffled-контролем и фиксированное объединение 50/50. Без финального refit."}
    write_json(output / "metrics.json", metrics)
    write_json(output / "runtime.json", {
        "seconds": time.perf_counter() - started, "outer_fit_seconds": total_outer_fit_seconds,
        "inner_fit_seconds": sum(result["fit_seconds"] for entry in searches for search in entry["searches"].values() for result in search),
        "saved_fold_bundles": len(outer_metrics), "saved_final_bundles": 0,
        "roundtrip_verified": True, "gpu_used": False, "fusion_alpha": FUSION_ALPHA,
        "fusion_additional_fits": 0, "training_diagnostics_file": "training_diagnostics.json",
        "temporal_ensemble_fits": len(diagnostics),
        "temporal_member_fits": len(diagnostics) * len(temporal.ENSEMBLE_SEEDS),
        "feature_dimensions": {"global35": 35, "temporal": [64, 7]},
    })
    return metrics
