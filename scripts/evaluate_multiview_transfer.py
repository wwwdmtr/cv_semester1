"""CPU-пилот: многокамерная информация только при обучении однокамерной модели.

Параметры опыта фиксируются внешним протоколом до вызова run. Учителя для
мягких меток обучаются заново на людях только текущего обучающего поднабора;
сохранённые исторические внешние предсказания не используются.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from rehab26_baseline import (
    artifact_bundle, build_pipeline, classification_metrics, fit_pipeline,
    mean_metric, meets_guardrails, person_folds, physical_view_weights,
    save_verified_bundle, select_candidate, view_bin, write_json, write_predictions,
)


PROTOCOL_VERSION = "rehab26_multiview_transfer_cpu_v1"
METHODS = ("supervised", "label_smoothing", "distill_uniform",
           "distill_global_gate", "distill_class_view_gate")
ALPHA = 0.5
CAMERAS = ("cam1", "cam2", "cam3")
VIEW_BINS = ("frontal_abs_le_30", "oblique_30_to_60", "profile_abs_ge_60")
TEACHER = {"id": "logreg_C1_balanced", "family": "logreg", "C": 1.0, "balanced": True}
METRIC_KEYS = (
    "macro_f1", "false_feedback_rate", "error_recall", "correct_recall",
    "specific_error_type_recall", "specific_remark_precision",
)


def student_candidates():
    return [
        {"id": f"logreg_C{c:g}_{'balanced' if balanced else 'unweighted'}",
         "family": "logreg", "C": c, "balanced": balanced}
        for c in (0.1, 1.0, 10.0) for balanced in (False, True)
    ]


def validate_records(records, x, y):
    """Проверить полный парный дизайн до создания любого разбиения."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != (len(records), 35) or not np.isfinite(x).all():
        raise ValueError("Ожидается конечная матрица 35D для каждого наблюдения")
    if y.shape != (len(records),) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Нужны все три целевые категории")
    groups = defaultdict(list)
    for index, row in enumerate(records):
        if row["label"] != y[index] or not np.isfinite(float(row["view_angle"])):
            raise ValueError("Несогласованная метка или неконечный фактический угол")
        groups[row["physical_id"]].append(row)
    if not groups:
        raise ValueError("Пустой набор повторов")
    for physical_id, rows in groups.items():
        if len(rows) != 3 or {row["camera"] for row in rows} != set(CAMERAS):
            raise ValueError(f"Нужны ровно три разные камеры: {physical_id}")
        if len({row["person"] for row in rows}) != 1 or len({row["label"] for row in rows}) != 1:
            raise ValueError(f"Разные люди или метки одного повтора: {physical_id}")


def assert_disjoint(records, train, target):
    train, target = tuple(train), tuple(target)
    if not train or not target or set(train) & set(target):
        raise ValueError("Пустые или пересекающиеся индексы разбиения")
    for key in ("person", "physical_id"):
        if {records[i][key] for i in train} & {records[i][key] for i in target}:
            raise ValueError(f"Утечка {key} между обучением и проверкой")


def validate_probabilities(values, n=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3
            or (n is not None and len(values) != n)
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("Мягкие метки должны быть конечными распределениями трёх классов")
    return values


def mix_targets(y, teacher_probabilities, gates, alpha=ALPHA):
    """Поклассовая смесь с нормировкой; gate=0 — hard, gate=1 — uniform KD."""
    y = np.asarray(y, dtype=int)
    teacher = validate_probabilities(teacher_probabilities, len(y))
    gates = np.asarray(gates, dtype=float)
    if (gates.shape != teacher.shape or not np.isfinite(gates).all()
            or np.any(gates < 0) or np.any(gates > 1) or not 0 <= alpha <= 1
            or not set(np.unique(y)).issubset({0, 1, 2})):
        raise ValueError("Некорректные gates, alpha или классы")
    hard = np.eye(3)[y]
    mixed = (1.0 - alpha * gates) * hard + alpha * gates * teacher
    sums = mixed.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Нулевая масса мягкой метки")
    return validate_probabilities(mixed / sums, len(y))


def estimate_gate_table(records, y, single_view_probabilities, weights=None):
    """Beta(1,1)-сглаженные TPR/TNR OOF-учителя только на train subset."""
    y = np.asarray(y, dtype=int)
    predicted = validate_probabilities(single_view_probabilities, len(y)).argmax(axis=1)
    if len(records) != len(y):
        raise ValueError("Несогласованный размер данных для gates")
    weights = physical_view_weights(records) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Некорректные веса gates")
    bins = np.array([view_bin(row["view_angle"]) for row in records])
    table = {}
    for group in VIEW_BINS:
        selected = bins == group
        classes = {}
        for label in (1, 2):
            positive = selected & (y == label)
            negative = selected & (y != label)
            tp = float(weights[positive & (predicted == label)].sum())
            tn = float(weights[negative & (predicted != label)].sum())
            positives, negatives = float(weights[positive].sum()), float(weights[negative].sum())
            tpr, tnr = (tp + 1.0) / (positives + 2.0), (tn + 1.0) / (negatives + 2.0)
            balanced_accuracy = (tpr + tnr) / 2.0
            supported = positives > 0 and negatives > 0
            classes[str(label)] = {
                "true_positive_weight": tp, "positive_weight": positives,
                "true_negative_weight": tn, "negative_weight": negatives,
                "positive_observations": int(positive.sum()),
                "negative_observations": int(negative.sum()),
                "positive_physical_repetitions": len({records[i]["physical_id"] for i in np.flatnonzero(positive)}),
                "negative_physical_repetitions": len({records[i]["physical_id"] for i in np.flatnonzero(negative)}),
                "smoothed_tpr": tpr, "smoothed_tnr": tnr,
                "balanced_accuracy": balanced_accuracy,
                "supported": supported,
                "gate": max(0.0, 2.0 * balanced_accuracy - 1.0) if supported else 0.0,
            }
        table[group] = {
            "classes": classes,
            "gate_vector": [min(classes["1"]["gate"], classes["2"]["gate"]),
                            classes["1"]["gate"], classes["2"]["gate"]],
            "observation_count": int(selected.sum()),
            "physical_repetition_count": len({records[i]["physical_id"] for i in np.flatnonzero(selected)}),
            "physical_weight": float(weights[selected].sum()),
        }
    return table


def fit_soft_student(spec, x, y, weights, targets):
    """Weighted class expansion; scaler и class weights как у hard baseline.

    Scaler видит только оригинальные строки. До class_weight суммарный вес
    расширенных строк равен физическому весу исходных наблюдений.
    """
    x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights, dtype=float)
    targets = validate_probabilities(targets, len(y))
    if set(np.unique(y)) != {0, 1, 2} or x.shape != (len(y), 35):
        raise ValueError("Нужны исходные 35D и три hard-класса")
    if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Некорректные веса студента")
    pipeline = build_pipeline(spec, y, weights)
    scaler, classifier = pipeline.named_steps["scaler"], pipeline.named_steps["classifier"]
    scaler.fit(x, sample_weight=weights)
    transformed = scaler.transform(x)
    expanded_weights = (weights[:, None] * targets).ravel()
    expanded_y = np.tile(np.arange(3), len(y))
    expanded_x = np.repeat(transformed, 3, axis=0)
    keep = expanded_weights > 0
    np.testing.assert_allclose(expanded_weights.sum(), weights.sum(), rtol=0, atol=1e-10)
    classifier.fit(expanded_x[keep], expanded_y[keep], sample_weight=expanded_weights[keep])
    return pipeline


class TrainingTargetCache:
    """Кэш принадлежит одному запуску и привязан к точным train indices."""

    def __init__(self, records, x, y):
        validate_records(records, x, y)
        self.records, self.x, self.y = records, np.asarray(x), np.asarray(y)
        self.teacher_models = {}
        self.targets = {}
        self.audits = []

    def get(self, indices):
        key = tuple(sorted(int(i) for i in indices))
        if len(key) != len(set(key)) or not key or key[0] < 0 or key[-1] >= len(self.records):
            raise ValueError("Неверные индексы обучающего поднабора")
        if key in self.targets:
            return self.targets[key]
        rows = [self.records[i] for i in key]
        people = sorted({row["person"] for row in rows})
        if len(people) < 2:
            raise ValueError("Crossfit-учителю нужны хотя бы два человека")
        # Поднабор должен сохранять все камеры физических повторов.
        validate_records(rows, self.x[list(key)], self.y[list(key)])
        local = np.array(key)
        single = np.full((len(key), 3), np.nan)
        assignments = np.zeros(len(key), dtype=int)
        split_audits = []
        for fold in person_folds(rows):
            local_train, local_target = np.array(fold["train"]), np.array(fold["test"])
            train, target = local[local_train], local[local_target]
            assert_disjoint(self.records, train, target)
            train_key = tuple(sorted(train.tolist()))
            train_rows = [self.records[i] for i in train]
            if train_key not in self.teacher_models:
                self.teacher_models[train_key] = fit_pipeline(
                    TEACHER, self.x[train], self.y[train], physical_view_weights(train_rows))
            teacher = self.teacher_models[train_key]
            if list(teacher.classes_) != [0, 1, 2]:
                raise ValueError("Неверный порядок классов учителя")
            single[local_target] = teacher.predict_proba(self.x[target])
            assignments[local_target] += 1
            split_audits.append({
                "teacher_train_people": sorted({self.records[i]["person"] for i in train}),
                "target_people": sorted({self.records[i]["person"] for i in target}),
                "teacher_train_indices": train.tolist(), "target_indices": target.tolist(),
                "teacher_train_physical_ids": sorted({self.records[i]["physical_id"] for i in train}),
                "target_physical_ids": sorted({self.records[i]["physical_id"] for i in target}),
                "people_disjoint": True, "physical_repetitions_disjoint": True,
            })
        if not np.all(assignments == 1):
            raise ValueError("Каждая train-строка должна иметь ровно один OOF-ответ учителя")
        validate_probabilities(single, len(key))
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["physical_id"]].append(index)
        averaged = np.empty_like(single)
        for group in groups.values():
            averaged[group] = single[group].mean(axis=0)
        table = estimate_gate_table(rows, self.y[local], single)
        gates = np.array([table[view_bin(row["view_angle"])]["gate_vector"] for row in rows])
        global_gate = float(np.average(gates.mean(axis=1), weights=physical_view_weights(rows)))
        weights = physical_view_weights(rows)
        teacher_diagnostics = {}
        for name, probabilities in (("single_view", single), ("mean_three_views", averaged)):
            prediction = probabilities.argmax(axis=1)
            per_person = {}
            for person in people:
                mask = np.array([row["person"] == person for row in rows])
                per_person[person] = classification_metrics(self.y[local][mask], prediction[mask], weights[mask])
            teacher_diagnostics[name] = {
                "pooled_repeat_weighted": classification_metrics(self.y[local], prediction, weights),
                "per_target_person": per_person,
            }
        audit = {
            "subset_id": hashlib.sha256(np.asarray(key, dtype=np.int64).tobytes()).hexdigest(),
            "subset_indices": list(key), "subset_people": people,
            "teacher_spec": TEACHER, "teacher_crossfit_splits": split_audits,
            "gate_table": table, "gate_source": "crossfit single-view teacher predictions on this exact train subset",
            "gate_counting": "physical view weights (1/3 per observation), Beta(1,1) per TPR/TNR",
            "global_gate": global_gate,
            "teacher_diagnostics": teacher_diagnostics,
            "teacher_target": "mean of three synchronous crossfit predict_proba vectors per physical repetition",
            "target_probability_min": float(averaged.min()),
            "target_probability_row_sum_max_error": float(np.max(np.abs(averaged.sum(axis=1) - 1.0))),
        }
        self.audits.append(audit)
        result = {"indices": local, "teacher_probabilities": validate_probabilities(averaged),
                  "single_view_probabilities": single, "gates": gates,
                  "global_gate": global_gate, "audit": audit}
        self.targets[key] = result
        return result

    def for_condition(self, indices, method):
        indices = np.asarray(indices, dtype=int)
        if method == "supervised":
            return np.eye(3)[self.y[indices]]
        if method == "label_smoothing":
            return (1.0 - ALPHA) * np.eye(3)[self.y[indices]] + ALPHA / 3.0
        if method not in METHODS:
            raise ValueError(f"Неизвестное условие {method}")
        cached = self.get(indices)
        positions = {value: index for index, value in enumerate(cached["indices"])}
        order = [positions[index] for index in indices]
        if method == "distill_class_view_gate":
            gates = cached["gates"][order]
        elif method == "distill_global_gate":
            gates = np.full((len(indices), 3), cached["global_gate"])
        else:
            gates = np.ones((len(indices), 3))
        return mix_targets(self.y[indices], cached["teacher_probabilities"][order], gates)


def search_students(cache, indices, method, on_progress=None):
    indices = np.asarray(indices, dtype=int)
    records = [cache.records[i] for i in indices]
    results = []
    for spec in student_candidates():
        fold_results = []
        started = time.perf_counter()
        for fold in person_folds(records):
            train, validation = indices[fold["train"]], indices[fold["test"]]
            assert_disjoint(cache.records, train, validation)
            train_rows = [cache.records[i] for i in train]
            validation_rows = [cache.records[i] for i in validation]
            targets = cache.for_condition(train, method)
            model = fit_soft_student(spec, cache.x[train], cache.y[train], physical_view_weights(train_rows), targets)
            measured = classification_metrics(cache.y[validation], model.predict(cache.x[validation]),
                                              physical_view_weights(validation_rows))
            fold_results.append({"validation_person": fold["test_person"], **measured})
        fpr, recall = mean_metric(fold_results, "false_feedback_rate"), mean_metric(fold_results, "error_recall")
        results.append({
            "candidate": spec, "folds": fold_results,
            "mean_macro_f1": mean_metric(fold_results, "macro_f1"),
            "mean_false_feedback_rate": fpr, "mean_error_recall": recall,
            "guardrails_met": meets_guardrails(fpr, recall), "fit_seconds": time.perf_counter() - started,
        })
        if on_progress is not None:
            on_progress(results)
    return results


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


def summarize_methods(outer_metrics, predictions):
    summaries = []
    for method in METHODS:
        rows = [row for row in outer_metrics if row["method"] == method]
        summary = {"method": method, "n_test_people": len(rows)}
        for metric in METRIC_KEYS:
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[metric + "_mean"] = float(np.mean(values)) if values else None
            summary[metric + "_std"] = float(np.std(values)) if values else None
            summary[metric + "_worst"] = (
                float(max(values) if metric == "false_feedback_rate" else min(values)) if values else None)
        selected = [row for row in predictions if row["method"] == method]
        summary["pooled_repeat_weighted"] = detailed_metrics(
            [row["true_label"] for row in selected], [row["predicted_class"] for row in selected],
            [row["view_weight"] for row in selected])
        summary["mean_guardrails_met"] = meets_guardrails(
            summary["false_feedback_rate_mean"], summary["error_recall_mean"])
        summary["all_people_guardrails_met"] = all(
            meets_guardrails(row["false_feedback_rate"], row["error_recall"]) for row in rows)
        summaries.append(summary)
    return summaries


def write_training_targets(path, cache):
    """Сохранить уже рассчитанные OOF targets/gates, без новых fit."""
    exported = []
    for key, cached in sorted(cache.targets.items()):
        weights = physical_view_weights([cache.records[i] for i in key])
        for position, index in enumerate(key):
            row = cache.records[index]
            exported.append({
                "subset_id": cached["audit"]["subset_id"],
                "subset_people": json.dumps(cached["audit"]["subset_people"]),
                "observation_index": index, "person": row["person"],
                "physical_id": row["physical_id"], "camera": row["camera"],
                "view_angle": row["view_angle"], "view_bin": view_bin(row["view_angle"]),
                "true_label": int(cache.y[index]),
                "physical_view_weight": float(weights[position]),
                "teacher_single_view_probabilities": json.dumps(cached["single_view_probabilities"][position].tolist()),
                "teacher_mean_three_view_probabilities": json.dumps(cached["teacher_probabilities"][position].tolist()),
                "class_view_gate": json.dumps(cached["gates"][position].tolist()),
                "global_gate": cached["global_gate"],
            })
    if exported:
        write_predictions(path, exported)


def run(records, x, y, output, models_dir, feature_metadata=None, dataset_metadata=None):
    """Выполнить зафиксированный nested-пилот на результате baseline.prepare.

    output может уже содержать manifest/config, созданные обёрткой. Файлы
    результатов этого опыта и непустой models_dir перезаписывать запрещено.
    Сохраняются только 25 внешних моделей; отдельного финального refit нет.
    """
    output, models_dir = Path(output), Path(models_dir)
    artifacts = ("inner_search.json", "outer_fold_metrics.json", "fold_predictions.csv",
                 "training_audits.json", "training_targets.csv", "metrics.json", "group_metrics.json", "runtime.json")
    if any((output / name).exists() for name in artifacts):
        raise ValueError("Каталог уже содержит результаты multiview transfer")
    if models_dir.exists() and any(models_dir.iterdir()):
        raise ValueError("Каталог моделей должен быть пустым")
    output.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validate_records(records, x, y)
    x, y = np.asarray(x), np.asarray(y, dtype=int)
    folds = person_folds(records, required_people=5)
    if feature_metadata is None:
        from rehab26_features import FEATURE_METADATA, FEATURE_NAMES, FEATURE_VERSION
        feature_metadata = {"version": FEATURE_VERSION, "names": list(FEATURE_NAMES), "details": FEATURE_METADATA}
    dataset_metadata = {} if dataset_metadata is None else dataset_metadata
    cache = TrainingTargetCache(records, x, y)
    searches, outer_metrics, predictions, group_metrics = [], [], [], []
    for outer in folds:
        train, test = np.asarray(outer["train"]), np.asarray(outer["test"])
        assert_disjoint(records, train, test)
        train_rows, test_rows = [records[i] for i in train], [records[i] for i in test]
        test_weights = physical_view_weights(test_rows)
        for method in METHODS:
            print(f"Внешний {outer['test_person']}; {method}: подбор на {', '.join(outer['train_people'])}", flush=True)
            search_entry = {"outer_test_person": outer["test_person"], "method": method,
                            "train_people": outer["train_people"], "search": []}
            searches.append(search_entry)

            def checkpoint(partial):
                search_entry["search"] = list(partial)
                write_json(output / "inner_search.json", searches)
                write_json(output / "training_audits.json", cache.audits)

            search = search_students(cache, train, method, checkpoint)
            chosen = select_candidate(search)
            spec = chosen["candidate"]
            search_entry["chosen"] = spec
            search_entry["selection_complete_before_outer_prediction"] = True
            write_json(output / "inner_search.json", searches)
            targets = cache.for_condition(train, method)
            write_json(output / "training_audits.json", cache.audits)
            fit_started = time.perf_counter()
            pipeline = fit_soft_student(spec, x[train], y[train], physical_view_weights(train_rows), targets)
            fit_seconds = time.perf_counter() - fit_started
            proba = validate_probabilities(pipeline.predict_proba(x[test]), len(test))
            predicted = pipeline.predict(x[test])
            measured = detailed_metrics(y[test], predicted, test_weights)
            outer_metrics.append({"method": method, "test_person": outer["test_person"],
                                  "candidate": spec, "fit_seconds": fit_seconds, **measured})
            bundle = artifact_bundle(pipeline, spec, feature_metadata, dataset_metadata,
                                     train_rows, "outer_evaluation_fold", outer["test_person"])
            bundle.update({"protocol_version": PROTOCOL_VERSION, "method": method,
                           "transfer_contract": {"alpha": ALPHA, "teacher": TEACHER,
                               "teacher_used": method.startswith("distill_"),
                               "teacher_excludes_target_person": method.startswith("distill_"),
                               "inference": "one complete 35D repetition from one camera; no angle or other cameras",
                               "angle_used_only_for_training_gate": method in ("distill_class_view_gate", "distill_global_gate")}})
            save_verified_bundle(models_dir / "folds" / f"{method}_heldout_{outer['test_person']}.joblib",
                                 bundle, x[test])
            for index, record in enumerate(test_rows):
                predictions.append({
                    "method": method, "candidate_id": spec["id"], "person": record["person"],
                    "physical_id": record["physical_id"], "camera": record["camera"],
                    "orientation": record.get("orientation", ""), "view_angle": record["view_angle"],
                    "view_bin": view_bin(record["view_angle"]), "true_label": int(y[test[index]]),
                    "predicted_class": int(predicted[index]), "view_weight": float(test_weights[index]),
                    "confidence_score": float(proba[index].max()),
                    "confidence_kind": "max_predict_proba_uncalibrated",
                    "class_scores": json.dumps(proba[index].tolist()),
                })
            for dimension in ("camera", "view_bin", "view_angle"):
                values = [view_bin(row["view_angle"]) if dimension == "view_bin" else row[dimension]
                          for row in test_rows]
                for value in sorted(set(values)):
                    indices = np.array([i for i, item in enumerate(values) if item == value])
                    subset = [test_rows[i] for i in indices]
                    group_metrics.append({
                        "method": method, "test_person": outer["test_person"],
                        "grouping": dimension, "group": value,
                        **detailed_metrics(y[test][indices], predicted[indices], physical_view_weights(subset)),
                    })
            write_json(output / "outer_fold_metrics.json", outer_metrics)
            write_json(output / "group_metrics.json", group_metrics)
            write_predictions(output / "fold_predictions.csv", predictions)
    metrics = {
        "protocol_version": PROTOCOL_VERSION, "alpha": ALPHA,
        "summaries": summarize_methods(outer_metrics, predictions), "outer_folds": outer_metrics,
        "note": "Исследовательский пилот; внешний тест не выбирает метод или alpha. Одна камера при inference. Отказы не оценивались.",
    }
    write_json(output / "metrics.json", metrics)
    write_training_targets(output / "training_targets.csv", cache)
    write_json(output / "runtime.json", {
        "seconds": time.perf_counter() - started, "saved_fold_bundles": len(outer_metrics),
        "saved_final_bundles": 0, "roundtrip_verified": True, "gpu_used": False,
        "crossfit_target_subsets": len(cache.targets), "unique_teacher_fit_subsets": len(cache.teacher_models),
    })
    return metrics
