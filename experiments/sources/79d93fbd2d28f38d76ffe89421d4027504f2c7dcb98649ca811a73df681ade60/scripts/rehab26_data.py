"""Проверенный загрузчик наблюдений одного приседания REHAB26.

Одна физическая попытка имеет общий ID для трёх камер. Сырые массивы
открываются memory-map; кадры/координаты не исправляются и не обрезаются.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_rehab26 import MD5

LABEL_NAMES = {0: "correct", 1: "knees_inward", 2: "shallow_squat"}
ERROR_LABELS = {"ERR_KNEES_INSIDE": 1, "ERR_SHALLOW_SQUAT": 2}
CAMERA_OFFSETS = {"cam1": 0., "cam2": 45., "cam3": 90.}


def read_annotations(data_dir):
    for filename, expected in MD5.items():
        path = data_dir / filename
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Разметка отличается от проверенной версии: {filename}")
    with (data_dir / "Segmentation.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream, delimiter=";") if r["exercise_id"] == "6"]
    keys = set()
    for row in rows:
        first, last = int(row["first_frame"]), int(row["last_frame"])
        key = (row["video_id"], first, last)
        if key in keys or first < 0 or last <= first:
            raise ValueError(f"Неоднозначная или неверная граница повторения: {key}")
        keys.add(key)
        if row["correctness_score"] not in {"1", "5"}:
            raise ValueError(f"Неподдерживаемая корректность: {row}")
        if (row["correctness_score"] == "1") != (row["error_description"] == ""):
            raise ValueError(f"Противоречивые метки: {key}")
    return rows


def physical_id(row):
    return f"{row['video_id']}:{row['first_frame']}-{row['last_frame']}"


def inspect_array(array, filename):
    if array.ndim != 3 or array.shape[1:] != (33, 4) or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Неожиданный формат позы {filename}: {array.shape}, {array.dtype}")
    visibility = array[:, :, 3]
    valid_visibility = np.isfinite(visibility)
    if np.any((visibility[valid_visibility] < 0) | (visibility[valid_visibility] > 1)):
        raise ValueError(f"Visibility вне [0, 1] в {filename}")
    # Координаты 3D могут иметь ноль после центрирования. Ноль не является NaN/пропуском.
    xyz = array[:, :, :3]
    joints = [11, 12, 23, 24, 25, 26, 27, 28]
    torso = (xyz[:, 11] + xyz[:, 12]) / 2 - (xyz[:, 23] + xyz[:, 24]) / 2
    torso_length = np.linalg.norm(torso, axis=-1)
    edges = [(23, 25), (25, 27), (24, 26), (26, 28), (11, 23), (12, 24)]
    same_points = {f"{a}-{b}": int(np.sum(np.linalg.norm(xyz[:, a] - xyz[:, b], axis=-1) < 1e-10)) for a, b in edges}
    return {
        "filename": str(filename), "shape": list(array.shape), "dtype": str(array.dtype),
        "nonfinite_coordinates": int(np.sum(~np.isfinite(xyz))),
        "nonfinite_visibility": int(np.sum(~valid_visibility)),
        "visibility_min": float(np.nanmin(visibility)),
        "visibility_max": float(np.nanmax(visibility)),
        "fraction_visibility_below_half": float(np.mean(visibility < .5)),
        "key_joint_fraction_visibility_below_half": float(np.mean(visibility[:, joints] < .5)),
        "all_zero_frames": int(np.sum(np.all(array == 0, axis=(1, 2)))),
        "torso_length_median": float(np.nanmedian(torso_length)),
        "zero_torso_frames": int(np.sum(torso_length < 1e-10)),
        "coincident_connected_joint_frames": same_points,
    }


def load_repetitions(data_dir, dimensions="3d", include_other_errors=False):
    """Вернуть (наблюдения повторений, аудит), сохранив camera/person группы.

    Для неподдерживаемых ошибок при include_other_errors=True label=None;
    такие записи не должны передаваться обычному трёхклассовому обучению.
    Границы по умолчанию включают last_frame; без RGB это явно принятое
    соглашение загрузчика, а не заявленная визуальная проверка границы.
    """
    data_dir = Path(data_dir)
    if dimensions not in {"2d", "3d"}:
        raise ValueError("dimensions должен быть '2d' или '3d'")
    rows = read_annotations(data_dir)
    arrays, files = {}, []
    manifest_path = data_dir / f"download_manifest_{dimensions}.json"
    manifest_hashes = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hashes = {m["filename"]: m["sha256"] for m in manifest["members"]}
    source_interval_issues = []
    for video_id in sorted({r["video_id"] for r in rows}):
        for camera in CAMERA_OFFSETS:
            relative = Path("mediapipe") / "ex6" / f"{video_id}_{camera}_{dimensions}.npy"
            path = data_dir / relative
            if not path.exists():
                raise FileNotFoundError(f"Не найден обязательный файл {path}; запустите scripts/download_rehab26.py")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            file_audit = inspect_array(array, relative)
            file_audit["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if manifest_hashes and manifest_hashes.get(relative.as_posix()) != file_audit["sha256"]:
                raise ValueError(f"Файл позы отличается от проверенного скачивания: {relative}")
            for row in rows:
                if row["video_id"] == video_id and int(row["last_frame"]) >= len(array):
                    source_interval_issues.append({"physical_id": physical_id(row), "camera": camera, "array_frames": len(array), "last_frame": int(row["last_frame"])})
            files.append(file_audit)
            arrays[(video_id, camera)] = array
    records, excluded_annotations, excluded_observations, unselected_annotations = [], [], [], []
    selected_physical = set()
    for row in rows:
        pid = physical_id(row)
        additional = [v.strip() for v in row["additional_errors"].split(",") if v.strip()]
        error = row["error_description"]
        if additional:
            excluded_annotations.append({"physical_id": pid, "reason": "additional_errors_require_separate_multilabel_protocol", "codes": [error, *additional]})
            continue
        if row["correctness_score"] == "1":
            label, label_name = 0, LABEL_NAMES[0]
        elif error in ERROR_LABELS:
            label = ERROR_LABELS[error]
            label_name = LABEL_NAMES[label]
        else:
            unselected_annotations.append({"physical_id": pid, "reason": "unsupported_error", "code": error})
            if not include_other_errors:
                continue
            label, label_name = None, "unsupported:" + error
        if label is not None:
            selected_physical.add(pid)
        first, last = int(row["first_frame"]), int(row["last_frame"])
        orientation = float(row["person_orientation"])
        for camera, offset in CAMERA_OFFSETS.items():
            array = arrays[(row["video_id"], camera)]
            if last >= len(array):
                excluded_observations.append({"physical_id": pid, "person": row["person"], "camera": camera, "reason": "annotation_outside_array", "first_frame": first, "last_frame": last, "array_frames": len(array)})
                continue
            poses = array[first:last + 1]
            torso = (poses[:, 11, :3] + poses[:, 12, :3]) / 2 - (poses[:, 23, :3] + poses[:, 24, :3]) / 2
            records.append({
                "physical_id": pid, "person": row["person"], "camera": camera,
                "orientation": orientation, "view_angle": offset - orientation,
                "first_frame": first, "last_frame": last, "label": label, "label_name": label_name,
                "poses": poses, "n_frames": len(poses), "video_id": row["video_id"],
                "observation_id": f"{pid}:{camera}", "dimensions": dimensions,
                "n_nonfinite_coordinates": int(np.sum(~np.isfinite(poses[:, :, :3]))),
                "n_zero_torso_frames": int(np.sum(np.linalg.norm(torso, axis=-1) < 1e-10)),
            })
    observation_counts = Counter(r["physical_id"] for r in records)
    for record in records:
        record["view_weight"] = 1. / observation_counts[record["physical_id"]]
    selected_records = [r for r in records if r["label"] is not None]
    actual_selected_physical = {r["physical_id"] for r in selected_records}
    camera_support = Counter(observation_counts[pid] for pid in actual_selected_physical)
    expected_source_counts = {
        "physical_squats": 554, "people": 5, "selected_physical_repetitions": 388,
        "selected_camera_observations": 1164,
    }
    audit = {
        "source": "https://zenodo.org/records/23018128", "dimensions": dimensions,
        "expected_source_counts": expected_source_counts,
        "frame_interval_convention": "0-based; first_frame and last_frame included",
        "boundary_visual_verification": False,
        "boundary_note": "Включение last_frame зафиксировано для загрузчика; точная семантика концов без RGB-визуализации не подтверждена.",
        "physical_key": ["video_id", "first_frame", "last_frame"],
        "source_physical_squats": len(rows),
        "source_people": sorted({r["person"] for r in rows}),
        "selected_physical_repetitions_before_file_checks": len(selected_physical),
        "selected_physical_repetitions": len(actual_selected_physical),
        "selected_camera_observations": len(selected_records),
        "physical_repetition_camera_support": {str(k): v for k, v in sorted(camera_support.items())},
        "selected_observations_by_person": dict(sorted(Counter(r["person"] for r in selected_records).items())),
        "selected_observations_by_label": dict(sorted(Counter(r["label_name"] for r in selected_records).items())),
        "excluded_other_error_physical_repetitions": len(unselected_annotations),
        "other_error_annotations": unselected_annotations,
        "excluded_annotations": excluded_annotations,
        "excluded_observations": excluded_observations,
        "all_source_annotation_interval_issues": source_interval_issues,
        "selected_observations_with_zero_torso": sum(r["n_zero_torso_frames"] > 0 for r in selected_records),
        "selected_zero_torso_frames": sum(r["n_zero_torso_frames"] for r in selected_records),
        "files": files,
        "files_all_coordinates_finite": all(f["nonfinite_coordinates"] == 0 for f in files),
        "files_all_visibility_finite": all(f["nonfinite_visibility"] == 0 for f in files),
        "note": "Готовые MediaPipe-наблюдения; дополнительные ошибки не превращены в отрицательные примеры. Низкая visibility не служит причиной скрытого удаления примеров.",
    }
    return records, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument("--dimensions", choices=("2d", "3d"), default="3d")
    parser.add_argument("--include-other-errors", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/rehab26_data_audit.json"))
    args = parser.parse_args()
    records, audit = load_repetitions(args.data_dir, args.dimensions, args.include_other_errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{audit['selected_physical_repetitions']} физических повторений, {audit['selected_camera_observations']} наблюдений. Аудит: {args.output}")


if __name__ == "__main__":
    main()
