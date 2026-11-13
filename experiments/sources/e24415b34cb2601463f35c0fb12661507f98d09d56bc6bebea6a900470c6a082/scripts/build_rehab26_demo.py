"""Собрать фиксированную демонстрацию результатов REHAB26 по трём камерам.

Демонстрация использует только сохранённые внешние предсказания и готовые
MediaPipe world-позы. Она не переобучает модель и не является новой оценкой
качества или визуальной проверкой исходных RGB-видео.
"""

import argparse
import ast
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rehab26_data import load_repetitions
from rehab26_features import extract_signals


LABELS_RU = {0: "корректно", 1: "колени внутрь", 2: "малая глубина"}
CAMERAS = ("cam1", "cam2", "cam3")
EDGES = (
    (11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (29, 31), (31, 27), (28, 30),
    (30, 32), (32, 28),
)
JOINTS = sorted({joint for edge in EDGES for joint in edge})

# Примеры зафиксированы до построения рисунка. Они покрывают три класса,
# расхождение камер и два типа ошибки системы; это не случайная выборка.
EXAMPLES = (
    {
        "kind": "верный ответ: корректно",
        "physical_id": "ex6_Vasil:10252-10294",
        "expected_true": 0,
        "expected_fused": 0,
    },
    {
        "kind": "верный ответ: колени внутрь",
        "physical_id": "ex6_Mima:11418-11481",
        "expected_true": 1,
        "expected_fused": 1,
    },
    {
        "kind": "верный ответ: малая глубина",
        "physical_id": "ex6_Petra:10822-10898",
        "expected_true": 2,
        "expected_fused": 2,
    },
    {
        "kind": "камеры расходятся, объединение верно",
        "physical_id": "ex6_Terka:10107-10183",
        "expected_true": 1,
        "expected_fused": 1,
    },
    {
        "kind": "ложное замечание",
        "physical_id": "ex6_Honza:11895-12000",
        "expected_true": 0,
        "expected_fused": 1,
    },
    {
        "kind": "пропущенная ошибка",
        "physical_id": "ex6_Vasil:10074-10108",
        "expected_true": 2,
        "expected_fused": 0,
    },
)


def parse_scores(value):
    """Прочитать сохранённый в CSV вектор оценок классов."""
    scores = np.asarray(ast.literal_eval(value) if isinstance(value, str) else value, dtype=float)
    if scores.shape != (3,) or not np.all(np.isfinite(scores)):
        raise ValueError(f"Неверный class_scores: {value!r}")
    return scores


def prepare_examples(predictions):
    """Проверить фиксированные ID и вернуть строки для отчёта."""
    required = {
        "method", "person", "physical_id", "camera", "true_label",
        "predicted_class", "confidence_score", "class_scores",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"В таблице предсказаний нет столбцов: {sorted(missing)}")
    selected = predictions[predictions["method"].eq("inner_selected")].copy()
    output = []
    for spec in EXAMPLES:
        group = selected[selected["physical_id"].eq(spec["physical_id"])].sort_values("camera")
        if tuple(group["camera"]) != CAMERAS:
            raise ValueError(f"Для {spec['physical_id']} нужны ровно {CAMERAS}")
        true_labels = set(group["true_label"].astype(int))
        if true_labels != {spec["expected_true"]}:
            raise ValueError(f"Изменилась метка {spec['physical_id']}: {true_labels}")
        scores = np.mean(np.stack([parse_scores(value) for value in group["class_scores"]]), axis=0)
        fused = int(np.argmax(scores))
        if fused != spec["expected_fused"]:
            raise ValueError(f"Изменился объединённый ответ {spec['physical_id']}: {fused}")
        output.append({
            **spec,
            "person": group["person"].iloc[0],
            "fused_confidence": float(scores[fused]),
            "cameras": {
                row.camera: {
                    "prediction": int(row.predicted_class),
                    "confidence": float(row.confidence_score),
                }
                for row in group.itertuples()
            },
        })
    return output


def render_pose(record, ax, prediction, confidence):
    poses = np.asarray(record["poses"])
    xyz = poses[:, :, :3]
    shoulders = (xyz[:, 11] + xyz[:, 12]) / 2
    hips = (xyz[:, 23] + xyz[:, 24]) / 2
    scale = float(np.median(np.linalg.norm(shoulders - hips, axis=-1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Нельзя нормировать {record['observation_id']}")
    centered = (xyz - hips[:, None]) / scale
    signals = extract_signals(poses)
    bottom = int(np.argmin(signals[:, :2].mean(axis=1)))
    for index, offset, color in zip(
        (0, bottom, len(poses) - 1),
        (-2.15, 0.0, 2.15),
        ("#8d99ae", "#0077b6", "#e09f3e"),
    ):
        xy = centered[index, :, :2].copy()
        xy[:, 0] += offset
        for first, last in EDGES:
            ax.plot(xy[[first, last], 0], xy[[first, last], 1], color=color, linewidth=1.0)
        ax.scatter(xy[JOINTS, 0], xy[JOINTS, 1], c=color, s=7)
        low_visibility = [joint for joint in JOINTS if poses[index, joint, 3] < 0.5]
        if low_visibility:
            ax.scatter(
                xy[low_visibility, 0], xy[low_visibility, 1], facecolors="none",
                edgecolors="#c1121f", s=28, linewidths=0.7,
            )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        f"{record['camera']} · прогноз: {LABELS_RU[prediction]}\n"
        f"score={confidence:.3f} · угол {record['view_angle']:g}°",
        fontsize=8,
    )


def write_manifest(examples, output_path):
    fields = [
        "kind", "person", "physical_id", "true_label", "cam1_prediction",
        "cam2_prediction", "cam3_prediction", "fused_prediction", "fused_confidence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in examples:
            writer.writerow({
                "kind": item["kind"],
                "person": item["person"],
                "physical_id": item["physical_id"],
                "true_label": LABELS_RU[item["expected_true"]],
                "cam1_prediction": LABELS_RU[item["cameras"]["cam1"]["prediction"]],
                "cam2_prediction": LABELS_RU[item["cameras"]["cam2"]["prediction"]],
                "cam3_prediction": LABELS_RU[item["cameras"]["cam3"]["prediction"]],
                "fused_prediction": LABELS_RU[item["expected_fused"]],
                "fused_confidence": f"{item['fused_confidence']:.6f}",
            })


def write_readme(examples, output_path):
    rows = []
    for item in examples:
        camera_predictions = " / ".join(
            LABELS_RU[item["cameras"][camera]["prediction"]] for camera in CAMERAS
        )
        rows.append(
            f"| {item['kind']} | {item['person']} | `{item['physical_id']}` | "
            f"{LABELS_RU[item['expected_true']]} | {camera_predictions} | "
            f"{LABELS_RU[item['expected_fused']]} ({item['fused_confidence']:.3f}) |"
        )
    text = """# Демонстрация REHAB26 на фиксированных примерах

Это иллюстрация уже полученных внешних предсказаний, а не новая оценка качества.
Шесть примеров заранее зафиксированы так, чтобы показать три верных класса,
расхождение камер, ложное замечание и пропущенную ошибку. Поэтому их нельзя
использовать для расчёта доли успешных случаев.

На рисунке показаны готовые MediaPipe world-позы в начале, оценённой нижней фазе
и конце повторения. Это не RGB-видео и не повторная экспертная разметка. Красные
кольца обозначают visibility ниже 0,5. `score` не является откалиброванной
вероятностью правильности.

![Фиксированные примеры трёх камер](examples.png)

| Случай | Человек | ID | Метка набора | cam1 / cam2 / cam3 | Среднее трёх камер |
|---|---|---|---|---|---|
""" + "\n".join(rows) + """

## Воспроизведение

Из корня репозитория после загрузки 3D-поз:

```bash
python scripts/download_rehab26.py --dimensions 3d
python scripts/build_rehab26_demo.py
```

Скрипт проверяет исходные метки, наличие трёх камер и сохранённые объединённые
ответы. Табличная версия примеров сохраняется в `examples.csv`.

## Как использовать в показе

1. Сначала показать три верных случая и объяснить три поддерживаемые категории.
2. Затем показать расхождение камер: одно движение получает разные ответы.
3. Закончить двумя ошибками системы. Это объясняет, почему результат пока нельзя
   использовать для автоматических рекомендаций и зачем нужна внешняя проверка.
"""
    output_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("reports/rehab26_baseline/fold_predictions.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/rehab26_demo"))
    args = parser.parse_args()

    examples = prepare_examples(pd.read_csv(args.predictions))
    records, _ = load_repetitions(args.data_dir, dimensions="3d")
    record_index = {(record["physical_id"], record["camera"]): record for record in records}
    args.output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(examples), len(CAMERAS), figsize=(13, 20), constrained_layout=True)
    for row_index, item in enumerate(examples):
        for column_index, camera in enumerate(CAMERAS):
            key = (item["physical_id"], camera)
            if key not in record_index:
                raise ValueError(f"Не найдена поза {key}")
            result = item["cameras"][camera]
            render_pose(record_index[key], axes[row_index, column_index], result["prediction"], result["confidence"])
        axes[row_index, 0].text(
            -0.08, 0.5,
            f"{item['kind']}\nметка: {LABELS_RU[item['expected_true']]}\n"
            f"объединение: {LABELS_RU[item['expected_fused']]}",
            transform=axes[row_index, 0].transAxes, rotation=90, ha="right", va="center", fontsize=9,
        )
    fig.suptitle(
        "REHAB26: фиксированные успехи и ошибки модели\n"
        "В каждой ячейке: начало / оценённая нижняя фаза / конец; это 3D-позы, не RGB",
        fontsize=13,
    )
    fig.savefig(args.output / "examples.png", dpi=160)
    plt.close(fig)
    write_manifest(examples, args.output / "examples.csv")
    write_readme(examples, args.output / "README.md")
    print(f"Демонстрация сохранена: {args.output}")


if __name__ == "__main__":
    main()
