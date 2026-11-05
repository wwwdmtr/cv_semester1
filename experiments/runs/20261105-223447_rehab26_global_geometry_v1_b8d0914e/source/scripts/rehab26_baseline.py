"""Сравнить CPU-классификаторы REHAB26 во вложенной проверке по людям.

Признаки, кандидаты и правила выбора фиксируются до внешнего теста.
Ни кадры, ни синхронные камеры не являются независимыми участниками.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSES = [0, 1, 2]
SOURCE = "https://zenodo.org/records/23018128"
PROTOCOL_VERSION = "rehab26_nested_people_geometry_v1"
METHODS = ("dummy", "logreg_fixed", "logreg_tuned", "svm_tuned", "extra_trees_tuned", "inner_selected")
GUARDRAILS = {"max_false_feedback_rate": .2, "min_error_recall": .8}


def meets_guardrails(false_feedback_rate, error_recall):
    return (false_feedback_rate is not None and error_recall is not None
            and false_feedback_rate <= GUARDRAILS["max_false_feedback_rate"] + 1e-12
            and error_recall >= GUARDRAILS["min_error_recall"] - 1e-12)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def physical_view_weights(records):
    counts = Counter(r["physical_id"] for r in records)
    return np.array([1. / counts[r["physical_id"]] for r in records], dtype=float)


def person_folds(records, required_people=None):
    people = sorted({r["person"] for r in records})
    if required_people is not None and len(people) != required_people:
        raise ValueError(f"Ожидалось {required_people} людей, получены {people}")
    folds = []
    for person in people:
        train = [i for i, row in enumerate(records) if row["person"] != person]
        test = [i for i, row in enumerate(records) if row["person"] == person]
        train_people = {records[i]["person"] for i in train}
        test_people = {records[i]["person"] for i in test}
        train_physical = {records[i]["physical_id"] for i in train}
        test_physical = {records[i]["physical_id"] for i in test}
        if train_people & test_people or train_physical & test_physical:
            raise ValueError("Один человек или физический повтор попал в обе части")
        folds.append({"test_person": person, "train_people": sorted(train_people), "train": train, "test": test})
    return folds


def classification_metrics(y_true, y_pred, weights=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("Пустой поднабор не имеет метрик распознавания")
    if not set(np.unique(y_true)).issubset(CLASSES) or not set(np.unique(y_pred)).issubset(CLASSES):
        raise ValueError("Неизвестный класс")
    weights = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != y_true.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Некорректные веса наблюдений")
    normal = y_true == 0
    errors = ~normal
    false_feedback = normal & (y_pred != 0)
    found_errors = errors & (y_pred != 0)
    correct_error_type = errors & (y_pred == y_true)
    remarks = y_pred != 0

    def ratio(num, den):
        return float(weights[num].sum() / weights[den].sum()) if weights[den].sum() else None

    def counts(weight):
        return {
            "all": float(weight.sum()), "correct_executions": float(weight[normal].sum()),
            "false_feedback": float(weight[false_feedback].sum()),
            "error_executions": float(weight[errors].sum()),
            "detected_errors": float(weight[found_errors].sum()),
            "missed_errors": float(weight[errors & (y_pred == 0)].sum()),
            "correct_error_type": float(weight[correct_error_type].sum()),
            "remarks": float(weight[remarks].sum()),
        }

    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", sample_weight=weights, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=weights)),
        "false_feedback_rate": ratio(false_feedback, normal),
        "error_recall": ratio(found_errors, errors),
        "correct_recall": ratio(normal & (y_pred == 0), normal),
        "specific_error_type_recall": ratio(correct_error_type, errors),
        "specific_remark_precision": ratio(correct_error_type, remarks),
        "per_class_f1": f1_score(y_true, y_pred, labels=CLASSES, average=None, sample_weight=weights, zero_division=0).tolist(),
        "per_class_support": [float(weights[y_true == label].sum()) for label in CLASSES],
        "confusion_matrix_repeat_weighted": confusion_matrix(y_true, y_pred, labels=CLASSES, sample_weight=weights).tolist(),
        "confusion_matrix_observations": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "counts_repeat_weighted": counts(weights),
        "counts_observations": {k: int(v) for k, v in counts(np.ones(len(y_true))).items()},
    }


def candidate_grid(seed=0):
    specs = []
    for family in ("logreg", "svm"):
        for c in (.1, 1., 10.):
            for balance in (False, True):
                specs.append({
                    "id": f"{family}_C{c:g}_{'balanced' if balance else 'unweighted'}",
                    "family": family, "C": c, "balanced": balance,
                })
    for depth in (3, 6, None):
        for leaf in (2, 5):
            specs.append({
                "id": f"extra_trees_depth{depth}_leaf{leaf}_seed{seed}",
                "family": "extra_trees", "max_depth": depth,
                "min_samples_leaf": leaf, "seed": seed,
            })
    return specs


def build_pipeline(spec, y_train, weights):
    class_weight = None
    if spec.get("balanced"):
        total = weights.sum()
        class_weight = {label: float(total / (len(CLASSES) * weights[y_train == label].sum())) for label in CLASSES}
    if spec["family"] == "dummy":
        estimator = DummyClassifier(strategy="most_frequent")
    elif spec["family"] == "logreg":
        estimator = LogisticRegression(C=spec["C"], class_weight=class_weight, max_iter=3000)
    elif spec["family"] == "svm":
        estimator = SVC(C=spec["C"], kernel="rbf", gamma="scale", class_weight=class_weight, probability=False)
    elif spec["family"] == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=200, max_depth=spec["max_depth"], min_samples_leaf=spec["min_samples_leaf"],
            class_weight=None, random_state=spec["seed"], n_jobs=1,
        )
    else:
        raise ValueError(spec)
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def fit_pipeline(spec, x, y, weights):
    if set(np.unique(y)) != set(CLASSES):
        raise ValueError("В обучающей части отсутствует целевой класс")
    pipeline = build_pipeline(spec, y, weights)
    pipeline.fit(x, y, scaler__sample_weight=weights, classifier__sample_weight=weights)
    return pipeline


def prediction_scores(pipeline, x):
    if hasattr(pipeline, "predict_proba"):
        values = pipeline.predict_proba(x)
        return values.max(axis=1), values, "max_predict_proba_uncalibrated"
    values = pipeline.decision_function(x)
    ordered = np.sort(values, axis=1)
    return ordered[:, -1] - ordered[:, -2], values, "ovr_decision_margin_uncalibrated"


def mean_metric(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def search_candidates(x, y, records, specs):
    folds = person_folds(records)
    results = []
    for spec in specs:
        fold_results = []
        fit_seconds = 0.
        for fold in folds:
            train, val = np.array(fold["train"]), np.array(fold["test"])
            train_rows = [records[i] for i in train]
            val_rows = [records[i] for i in val]
            started = time.perf_counter()
            model = fit_pipeline(spec, x[train], y[train], physical_view_weights(train_rows))
            fit_seconds += time.perf_counter() - started
            metrics = classification_metrics(y[val], model.predict(x[val]), physical_view_weights(val_rows))
            fold_results.append({"validation_person": fold["test_person"], **metrics})
        mean_fpr = mean_metric(fold_results, "false_feedback_rate")
        mean_recall = mean_metric(fold_results, "error_recall")
        results.append({
            "candidate": spec, "mean_macro_f1": mean_metric(fold_results, "macro_f1"),
            "mean_false_feedback_rate": mean_fpr, "mean_error_recall": mean_recall,
            "guardrails_met": meets_guardrails(mean_fpr, mean_recall),
            "fit_seconds": fit_seconds, "folds": fold_results,
        })
    return results


def select_candidate(results, family=None):
    pool = [r for r in results if family is None or r["candidate"]["family"] == family]
    if not pool:
        raise ValueError("Пустой набор кандидатов")
    feasible = [r for r in pool if r["guardrails_met"]]
    ranked = sorted(feasible or pool, key=lambda r: (
        -r["mean_macro_f1"],
        r["mean_false_feedback_rate"] if r["mean_false_feedback_rate"] is not None else float("inf"),
        -r["mean_error_recall"] if r["mean_error_recall"] is not None else float("inf"),
        r["candidate"]["id"],
    ))
    return ranked[0]


def view_bin(angle):
    a = abs(float(angle))
    return "frontal_abs_le_30" if a <= 30 else ("profile_abs_ge_60" if a >= 60 else "oblique_30_to_60")


def artifact_bundle(pipeline, spec, feature_metadata, dataset_metadata, train_records, scope, test_person=None):
    return {
        "format_version": 1, "protocol_version": PROTOCOL_VERSION, "pipeline": pipeline,
        "candidate": spec, "feature_version": feature_metadata["version"],
        "feature_names": feature_metadata["names"], "feature_metadata": feature_metadata,
        "input_format": "MediaPipe world landmarks; shape (frames, 33, 4): x,y,z,visibility; complete single repetition",
        "classes": CLASSES, "label_names": {0: "correct", 1: "knees_inward", 2: "shallow_squat"},
        "training_people": sorted({r["person"] for r in train_records}),
        "training_physical_ids": sorted({r["physical_id"] for r in train_records}),
        "scope": scope, "held_out_person": test_person,
        "confidence_threshold": None, "confidence_calibrated": False,
        "dataset": dataset_metadata,
        "versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
    }


def save_verified_bundle(path, bundle, x_probe):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    restored = joblib.load(path)
    before, after = bundle["pipeline"], restored["pipeline"]
    np.testing.assert_array_equal(before.predict(x_probe), after.predict(x_probe))
    np.testing.assert_allclose(prediction_scores(before, x_probe)[1], prediction_scores(after, x_probe)[1], atol=0, rtol=0)


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(data_dir, output):
    # Ленивый импорт позволяет отдельно проверять метрики/разбиения до загрузки данных.
    from rehab26_data import load_repetitions
    from rehab26_features import extract_features, FEATURE_NAMES, FEATURE_VERSION
    try:
        from rehab26_features import FEATURE_METADATA
    except ImportError:
        FEATURE_METADATA = {}
    records, audit = load_repetitions(data_dir, dimensions="3d")
    if not records or {r["label"] for r in records} != set(CLASSES):
        raise ValueError("Не получены все три выбранные категории")
    seen = set()
    for row in records:
        key = (row["physical_id"], row["camera"])
        if key in seen:
            raise ValueError(f"Дублируется камера физического повтора: {key}")
        seen.add(key)
    output.mkdir(parents=True, exist_ok=True)
    feature_rows, feature_failures = [], []
    for record in records:
        try:
            feature_rows.append(extract_features(record["poses"]))
        except (ValueError, FloatingPointError) as exc:
            feature_failures.append({"physical_id": record["physical_id"], "camera": record["camera"], "reason": str(exc)})
    write_json(output / "feature_failures.json", feature_failures)
    if feature_failures:
        raise ValueError(f"Признаки не рассчитаны для {len(feature_failures)} наблюдений; исключения не применены, причины в feature_failures.json")
    x = np.stack(feature_rows)
    if not np.isfinite(x).all() or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Признаки не конечны или не соответствуют схеме")
    y = np.array([r["label"] for r in records], dtype=int)
    folds = person_folds(records, required_people=5)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": FEATURE_VERSION, "names": list(FEATURE_NAMES),
        "details": FEATURE_METADATA,
    }
    annotations = data_dir / "Segmentation.csv"
    dataset_metadata = {
        "source_record": SOURCE,
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "feature_matrix_sha256": hashlib.sha256(x.tobytes(order="C")).hexdigest(),
        "n_observations": len(records), "n_physical_repetitions": len({r["physical_id"] for r in records}),
        "people": sorted({r["person"] for r in records}),
        "audit": audit,
    }
    manifest_path = data_dir / "download_manifest_3d.json"
    if manifest_path.exists():
        dataset_metadata["source_download_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(output / "data_audit.json", audit)
    write_json(output / "feature_metadata.json", metadata)
    write_json(output / "splits.json", [{
        "test_person": fold["test_person"], "train_people": fold["train_people"],
        "train_physical_ids": sorted({records[i]["physical_id"] for i in fold["train"]}),
        "test_physical_ids": sorted({records[i]["physical_id"] for i in fold["test"]}),
        "inner_validation_people": fold["train_people"],
    } for fold in folds])
    manifest = [{k: r[k] for k in ("physical_id", "person", "camera", "orientation", "view_angle", "first_frame", "last_frame", "label", "label_name", "n_frames")} for r in records]
    write_predictions(output / "manifest.csv", manifest)
    return records, x, y, folds, metadata, dataset_metadata


def run(data_dir, output, models_dir, seed=0, prepare_only=False):
    started = time.perf_counter()
    records, x, y, folds, feature_metadata, dataset_metadata = prepare(data_dir, output)
    specs = candidate_grid(seed)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_version": PROTOCOL_VERSION,
        "stage": "global_geometry_without_augmentation_or_abstention",
        "guardrails": GUARDRAILS, "selection": "inner feasibility first, then mean person Macro-F1; deterministic tie",
        "candidates": specs, "seed": seed, "dataset": dataset_metadata,
        "feature_metadata": feature_metadata,
        "scripts_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("rehab26_baseline.py", "rehab26_features.py", "rehab26_data.py")},
        "protocol_document_sha256": hashlib.sha256(Path("docs/experiment-protocol.md").read_bytes()).hexdigest(),
        "prepare_only": prepare_only,
    }
    write_json(output / "run_config.json", config)
    if prepare_only:
        print(f"Подготовлено {len(records)} наблюдений / {dataset_metadata['n_physical_repetitions']} физических повторов; обучение не запускалось.")
        return
    predictions, outer_metrics, inner_results, group_metrics = [], [], [], []
    fixed = {"id": "logreg_C1_balanced", "family": "logreg", "C": 1., "balanced": True}
    dummy = {"id": "dummy_most_frequent", "family": "dummy"}
    for fold in folds:
        train, test = np.array(fold["train"]), np.array(fold["test"])
        train_rows, test_rows = [records[i] for i in train], [records[i] for i in test]
        print(f"Внешний участник {fold['test_person']}: подбор только на {', '.join(fold['train_people'])}.", flush=True)
        search = search_candidates(x[train], y[train], train_rows, specs)
        choices = {
            "dummy": dummy, "logreg_fixed": fixed,
            "logreg_tuned": select_candidate(search, "logreg")["candidate"],
            "svm_tuned": select_candidate(search, "svm")["candidate"],
            "extra_trees_tuned": select_candidate(search, "extra_trees")["candidate"],
            "inner_selected": select_candidate(search)["candidate"],
        }
        inner_results.append({"outer_test_person": fold["test_person"], "search": search, "choices": choices})
        # Сохраняем выбор ДО вычисления внешних предсказаний.
        write_json(output / "inner_search.json", inner_results)
        fit_cache = {}
        for method, spec in choices.items():
            if spec["id"] not in fit_cache:
                fitting_started = time.perf_counter()
                pipeline = fit_pipeline(spec, x[train], y[train], physical_view_weights(train_rows))
                fit_cache[spec["id"]] = (pipeline, time.perf_counter() - fitting_started)
            pipeline, fit_seconds = fit_cache[spec["id"]]
            pred = pipeline.predict(x[test])
            confidence, class_scores, confidence_kind = prediction_scores(pipeline, x[test])
            measured = classification_metrics(y[test], pred, physical_view_weights(test_rows))
            row = {"method": method, "test_person": fold["test_person"], "candidate": spec, "fit_seconds": fit_seconds, **measured}
            outer_metrics.append(row)
            path = models_dir / "folds" / f"{method}_heldout_{fold['test_person']}.joblib"
            bundle = artifact_bundle(pipeline, spec, feature_metadata, dataset_metadata, train_rows, "outer_evaluation_fold", fold["test_person"])
            save_verified_bundle(path, bundle, x[test])
            for j, record in enumerate(test_rows):
                predictions.append({
                    "method": method, "candidate_id": spec["id"], "person": record["person"],
                    "physical_id": record["physical_id"], "camera": record["camera"],
                    "orientation": record["orientation"], "view_angle": record["view_angle"],
                    "view_bin": view_bin(record["view_angle"]), "true_label": int(y[test[j]]),
                    "predicted_class": int(pred[j]), "view_weight": float(physical_view_weights(test_rows)[j]),
                    "confidence_score": float(confidence[j]), "confidence_kind": confidence_kind,
                    "class_scores": json.dumps(class_scores[j].tolist()),
                })
            for dimension in ("camera", "view_bin"):
                values = [r["camera"] if dimension == "camera" else view_bin(r["view_angle"]) for r in test_rows]
                for value in sorted(set(values)):
                    idx = [j for j, v in enumerate(values) if v == value]
                    subset = [test_rows[j] for j in idx]
                    group_metrics.append({"method": method, "test_person": fold["test_person"], "grouping": dimension, "group": value,
                                          **classification_metrics(y[test][idx], pred[idx], physical_view_weights(subset))})
        # Промежуточные артефакты позволяют видеть прогресс, но не используются для выбора.
        write_json(output / "outer_fold_metrics.json", outer_metrics)
        write_predictions(output / "fold_predictions.csv", predictions)
    summaries = []
    for method in METHODS:
        rows = [r for r in outer_metrics if r["method"] == method]
        summary = {"method": method, "n_test_people": len(rows)}
        for key in ("macro_f1", "false_feedback_rate", "error_recall", "correct_recall", "specific_error_type_recall", "specific_remark_precision"):
            values = [r[key] for r in rows if r[key] is not None]
            summary[key + "_mean"] = float(np.mean(values)) if values else None
            summary[key + "_std"] = float(np.std(values)) if values else None
            summary[key + "_worst"] = float(max(values) if key == "false_feedback_rate" else min(values)) if values else None
        method_predictions = [r for r in predictions if r["method"] == method]
        summary["pooled_repeat_weighted"] = classification_metrics(
            [r["true_label"] for r in method_predictions], [r["predicted_class"] for r in method_predictions], [r["view_weight"] for r in method_predictions])
        summary["mean_guardrails_met"] = meets_guardrails(summary["false_feedback_rate_mean"], summary["error_recall_mean"])
        summary["all_people_guardrails_met"] = all(meets_guardrails(r["false_feedback_rate"], r["error_recall"]) for r in rows)
        summaries.append(summary)
    write_json(output / "metrics.json", {
        "protocol_version": PROTOCOL_VERSION, "summaries": summaries, "outer_folds": outer_metrics,
        "note": "Главный автоматический результат: inner_selected. Сравнение отдельных семейств описательное; выбирать финальный вариант по внешнему тесту нельзя. Отказы не оценивались.",
    })
    write_json(output / "group_metrics.json", group_metrics)
    print("Финальный выбор параметров: отдельная проверка по людям на всех доступных обучающих данных.", flush=True)
    final_search = search_candidates(x, y, records, specs)
    final_choice = select_candidate(final_search)
    write_json(output / "final_training_selection.json", {"search": final_search, "chosen": final_choice["candidate"], "guardrails_met_in_selection": final_choice["guardrails_met"]})
    final_model = fit_pipeline(final_choice["candidate"], x, y, physical_view_weights(records))
    bundle = artifact_bundle(final_model, final_choice["candidate"], feature_metadata, dataset_metadata, records, "refit_all_people_no_independent_score")
    save_verified_bundle(models_dir / "squat_geometry.joblib", bundle, x)
    write_json(output / "runtime.json", {"seconds": time.perf_counter() - started, "saved_fold_bundles": len(folds) * len(METHODS), "saved_final_bundles": 1, "roundtrip_verified": True, "gpu_used": False})
    for row in summaries:
        print(f"{row['method']:18} Macro-F1={row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}; FPR={row['false_feedback_rate_mean']:.3f}; recall_error={row['error_recall_mean']:.3f}")
    print(f"Отчёты: {output}; модели: {models_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument("--output", type=Path, default=Path("reports/rehab26_baseline"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/rehab26_baseline"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output, args.models_dir, args.seed, args.prepare_only)


if __name__ == "__main__":
    main()
