"""Аудит EC3D и исходный эксперимент без пересечения участников."""

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from download_ec3d import COMMIT, SHA256, URL, verify

# BODY_25: OpenPose. Метку 10 не включаем: она отсутствует в таблице 1 статьи.
JOINTS = list(range(15)) + [19, 21, 22, 24]
INSTRUCTIONS = (1, 2, 3, 4, 5)


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy", "dtype"): np.dtype,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        }
        if (module, name) in allowed:
            return allowed[module, name]
        raise pickle.UnpicklingError(f"Неподдерживаемый объект: {module}.{name}")


def load_sequences(path):
    verify(path)
    data = ArrayUnpickler(io.BytesIO(path.read_bytes())).load()
    poses = np.asarray(data["poses"], dtype=float).transpose(0, 2, 1)
    labels = np.asarray(data["labels"])
    if poses.shape != (len(labels), 25, 3) or labels.shape[1] != 5:
        raise ValueError("Неожиданный формат EC3D")
    if not np.isfinite(poses).all():
        raise ValueError("В координатах есть NaN или бесконечности")
    groups = defaultdict(list)
    for i, (action, subject, instruction, repetition, frame) in enumerate(labels):
        groups[(str(action), str(subject), int(instruction), int(repetition))].append(
            (int(frame), i)
        )
    sequences = []
    for key, frames in sorted(groups.items()):
        frame_ids, indices = zip(*sorted(frames))
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError(f"Дублирующиеся кадры: {key}")
        action, subject, instruction, repetition = key
        sequences.append({
            "id": f"{action}_{subject}_{instruction}_{repetition}",
            "action": action, "subject": subject, "instruction": instruction,
            "repetition": repetition, "n_frames": len(indices),
            "poses": poses[list(indices)],
        })
    return sequences, len(labels)


def select_squats(sequences):
    return [s for s in sequences if s["action"] == "SQUAT" and s["instruction"] in INSTRUCTIONS]


def normalize(poses):
    centered = poses - poses[:, 8:9]
    scale = np.median(np.linalg.norm(poses[:, 1] - poses[:, 8], axis=-1))
    if scale < 1e-8:
        raise ValueError("Нулевая длина корпуса")
    return centered / scale


def resampled_features(poses, n_frames=32):
    p = normalize(poses)[:, JOINTS].reshape(len(poses), -1)
    source = np.linspace(0, 1, len(p))
    target = np.linspace(0, 1, n_frames)
    return np.stack([np.interp(target, source, col) for col in p.T], axis=1).ravel()


def angle(a, b, c):
    left, right = a - b, c - b
    denom = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    if np.any(denom < 1e-10):
        raise ValueError("Вырожденные сегменты при расчёте угла")
    return np.arccos(np.clip(np.sum(left * right, axis=-1) / denom, -1, 1))


def geometric_features(poses):
    p = normalize(poses)
    signals = [
        angle(p[:, 9], p[:, 10], p[:, 11]),  # правое колено
        angle(p[:, 12], p[:, 13], p[:, 14]),  # левое колено
        angle(p[:, 1], p[:, 9], p[:, 10]),  # правое бедро
        angle(p[:, 1], p[:, 12], p[:, 13]),  # левое бедро
        np.linalg.norm(p[:, 10] - p[:, 13], axis=-1),
        np.linalg.norm(p[:, 11] - p[:, 14], axis=-1),
        np.linalg.norm(p[:, 8] - (p[:, 11] + p[:, 14]) / 2, axis=-1),
    ]
    # Сначала используем всё повторение. Выделение фаз — отдельный будущий опыт.
    return np.quantile(np.stack(signals, axis=1), [0, .1, .5, .9, 1], axis=0).ravel()


def subject_folds(sequences):
    subjects = sorted({s["subject"] for s in sequences})
    if len(subjects) != 4:
        raise ValueError(f"Ожидались четыре участника, получены {subjects}")
    for i, test_subject in enumerate(subjects):
        val_subject = subjects[(i + 1) % len(subjects)]
        train_subjects = set(subjects) - {test_subject, val_subject}
        assert not train_subjects.intersection({test_subject, val_subject})
        yield {
            "test_subject": test_subject, "val_subject": val_subject,
            "train_subjects": sorted(train_subjects),
            "train": [j for j, s in enumerate(sequences) if s["subject"] in train_subjects],
            "val": [j for j, s in enumerate(sequences) if s["subject"] == val_subject],
            "test": [j for j, s in enumerate(sequences) if s["subject"] == test_subject],
        }


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(data_path, out):
    out.mkdir(parents=True, exist_ok=True)
    all_sequences, n_frames = load_sequences(data_path)
    sequences = select_squats(all_sequences)
    excluded = [s for s in all_sequences if s["action"] == "SQUAT" and s["instruction"] not in INSTRUCTIONS]
    audit = {
        "source_url": URL, "source_commit": COMMIT, "sha256": SHA256,
        "total_frames": n_frames, "total_sequences": len(all_sequences),
        "by_action": dict(Counter(s["action"] for s in all_sequences)),
        "selected_squat_sequences": len(sequences),
        "selected_by_instruction": dict(sorted(Counter(str(s["instruction"]) for s in sequences).items())),
        "selected_by_subject": dict(Counter(s["subject"] for s in sequences)),
        "excluded_squat_ids": [s["id"] for s in excluded],
        "all_coordinates_finite": True,
        "note": "Метка 10 не описана в таблице 1 статьи и исключена до выяснения её смысла.",
    }
    write_json(out / "audit.json", audit)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "action", "subject", "instruction", "repetition", "n_frames"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sequences)
    features = {
        "geometry_logreg": np.stack([geometric_features(s["poses"]) for s in sequences]),
        "resampled_logreg": np.stack([resampled_features(s["poses"]) for s in sequences]),
    }
    folds = list(subject_folds(sequences))
    write_json(out / "splits.json", folds)
    records, summaries, predictions = [], [], []
    for task in ("binary", "instruction"):
        y = np.array([int(s["instruction"] != 1) if task == "binary" else s["instruction"] for s in sequences])
        classes = [0, 1] if task == "binary" else list(INSTRUCTIONS)
        for name in ("dummy", *features):
            x = features.get(name, features["geometry_logreg"])
            all_true, all_pred = [], []
            for fold in folds:
                train, val, test = (np.array(fold[k]) for k in ("train", "val", "test"))
                if set(y[train]) != set(classes):
                    raise ValueError("В обучении отсутствует один из целевых классов")
                model = DummyClassifier(strategy="most_frequent") if name == "dummy" else make_pipeline(
                    StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=3000)
                )
                start = time.perf_counter()
                model.fit(x[train], y[train])
                elapsed = time.perf_counter() - start
                pred = model.predict(x[test])
                confidence = model.predict_proba(x[test]).max(axis=1)
                # Порог выбран только по validation, фактическое покрытие test может отличаться.
                threshold = float(np.quantile(model.predict_proba(x[val]).max(axis=1), .2))
                accepted = confidence >= threshold
                records.append({
                    "task": task, "model": name, "test_subject": fold["test_subject"],
                    "n_test": len(test), "macro_f1": float(f1_score(y[test], pred, labels=classes, average="macro", zero_division=0)),
                    "accuracy": float(accuracy_score(y[test], pred)), "fit_seconds": elapsed,
                    "val_target_coverage": .8, "confidence_threshold": threshold,
                    "test_coverage": float(accepted.mean()),
                    "accepted_error_rate": float(np.mean(pred[accepted] != y[test][accepted])) if accepted.any() else None,
                    "false_feedback_rate": false_feedback_rate(y[test], pred, task),
                })
                for j, idx in enumerate(test):
                    predictions.append({
                        "task": task, "model": name, "id": sequences[idx]["id"],
                        "subject": sequences[idx]["subject"], "true": int(y[idx]),
                        "pred": int(pred[j]), "confidence": float(confidence[j]),
                        "accepted": bool(accepted[j]),
                    })
                all_true.extend(y[test].tolist())
                all_pred.extend(pred.tolist())
            rows = [r for r in records if r["task"] == task and r["model"] == name]
            summaries.append({
                "task": task, "model": name,
                "fold_macro_f1_mean": float(np.mean([r["macro_f1"] for r in rows])),
                "fold_macro_f1_std": float(np.std([r["macro_f1"] for r in rows])),
                "pooled_macro_f1": float(f1_score(all_true, all_pred, labels=classes, average="macro", zero_division=0)),
                "pooled_accuracy": float(accuracy_score(all_true, all_pred)),
                "per_class_f1": f1_score(all_true, all_pred, labels=classes, average=None, zero_division=0).tolist(),
                "false_feedback_rate": false_feedback_rate(np.array(all_true), np.array(all_pred), task),
                "labels": classes, "confusion_matrix": confusion_matrix(all_true, all_pred, labels=classes).tolist(),
            })
    write_json(out / "metrics.json", {"protocol": "4 outer folds; 2 train / 1 validation / 1 test subjects; C fixed at 1", "folds": records, "summary": summaries})
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    plot_confusions(summaries, out)
    for row in summaries:
        print(f"{row['task']:12} {row['model']:20} macro-F1={row['fold_macro_f1_mean']:.3f} ± {row['fold_macro_f1_std']:.3f}")
    print(f"Проверено {len(sequences)} приседаний. Отчёты: {out}")


def false_feedback_rate(y_true, y_pred, task):
    correct_class = 0 if task == "binary" else 1
    correctly_performed = y_true == correct_class
    return float(np.mean(y_pred[correctly_performed] != correct_class)) if correctly_performed.any() else None


def plot_confusions(summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in summaries if r["task"] == "instruction" and r["model"] != "dummy"]
    fig, axes = plt.subplots(1, len(selected), figsize=(9, 4), constrained_layout=True)
    for ax, row in zip(np.atleast_1d(axes), selected):
        matrix = np.array(row["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=INSTRUCTIONS, yticklabels=INSTRUCTIONS,
               xlabel="Predicted instruction ID", ylabel="True instruction ID", title=row["model"])
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, str(count), ha="center", va="center", color="white" if count > matrix.max() / 2 else "black")
    fig.savefig(out / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ec3d/data_3D.pickle"))
    parser.add_argument("--output", type=Path, default=Path("reports/ec3d_baseline"))
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
