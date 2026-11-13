"""Проверить локальную разметку REHAB26 без обучения и загрузки полного архива."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


MD5 = {
    "Segmentation.csv": "3d1cf6b552f60ef03eb8f5db977c139f",
    "Segmentation.txt": "502475f709390d96b599da65fa45ffd3",
    "Error_types.csv": "2f89da48c2c26b4868a59deaa4fb3ed9",
    "joint_names.txt": "2d6999cedf3775d3e9dded2e87b8d035",
}
SUPPORTED_ERRORS = {"ERR_KNEES_INSIDE", "ERR_SHALLOW_SQUAT"}
SAMPLE_SHA256 = "7412b04236b2fb5473eb4d311cb0ded33d9402a89b25a2e292e6391d2ab394d2"


def audit(data_dir, sample=None):
    checksums = {}
    for filename, expected in MD5.items():
        raw = (data_dir / filename).read_bytes()
        actual = hashlib.md5(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Контрольная сумма не совпала: {filename}")
        checksums[filename] = {"md5": actual, "bytes": len(raw)}
    with (data_dir / "Segmentation.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    squats = [r for r in rows if r["exercise_id"] == "6"]
    physical_ids = [(r["video_id"], int(r["first_frame"]), int(r["last_frame"])) for r in squats]
    if len(physical_ids) != len(set(physical_ids)):
        raise ValueError("Повторяется ключ физического приседания")
    if any(first < 0 or last <= first for _, first, last in physical_ids):
        raise ValueError("Некорректная граница повторения")
    valid_codes = {"", "ERR_KNEES_INSIDE", "ERR_TORSO_TO_SIDE", "ERR_SHALLOW_SQUAT", "ERR_KNEES_FORWARD", "ERR_LIFTED_HEELS"}
    for row in squats:
        if row["correctness_score"] not in {"1", "5"} or row["error_description"] not in valid_codes:
            raise ValueError("Неизвестная метка приседания")
        if (row["correctness_score"] == "1") != (row["error_description"] == ""):
            raise ValueError("Противоречие между корректностью и основным типом ошибки")
    selected = [r for r in squats if r["correctness_score"] == "1" or r["error_description"] in SUPPORTED_ERRORS]
    report = {
        "checked_on": "2026-11-05",
        "source": "https://zenodo.org/records/23018128",
        "checksums": checksums,
        "all_exercise_physical_repetitions": len(rows),
        "squat_physical_repetitions": len(squats),
        "squat_by_person": dict(sorted(Counter(r["person"] for r in squats).items())),
        "squat_by_correctness": dict(sorted(Counter(r["correctness_score"] for r in squats).items())),
        "squat_by_primary_error": dict(sorted(Counter(r["error_description"] or "correct" for r in squats).items())),
        "squat_rows_with_additional_errors": sum(bool(r["additional_errors"]) for r in squats),
        "selected_three_category_physical_repetitions": len(selected),
        "selected_three_category_camera_observations_if_all_available": len(selected) * 3,
        "selected_by_person": dict(sorted(Counter(r["person"] for r in selected).items())),
        "excluded_other_error_repetitions": len(squats) - len(selected),
        "naive_video_plus_repetition_key_collisions": len(squats) - len({(r["video_id"], r["repetition_number"]) for r in squats}),
        "physical_repetition_key": ["video_id", "first_frame", "last_frame"],
        "camera_offsets_degrees": {"cam1": 0, "cam2": 45, "cam3": 90},
        "view_angle_formula": "camera_offset - person_orientation",
        "training_run": False,
        "scope": "Разметка проверена полностью; полный архив поз и RGB ещё не скачаны. Дополнительные ошибки допускаются схемой даже при пустых полях в текущем поднаборе.",
    }
    if sample is not None:
        import numpy as np
        if hashlib.sha256(sample.read_bytes()).hexdigest() != SAMPLE_SHA256:
            raise ValueError("Опциональный пример должен быть проверенным файлом ex6_Petra_cam3_3d.npy")
        a = np.load(sample, allow_pickle=False)
        if a.ndim != 3 or a.shape[1:] != (33, 4):
            raise ValueError("Неожиданная форма примера MediaPipe")
        related = [r for r in squats if r["video_id"] == "ex6_Petra"]
        report["inspected_sample"] = {
            "archive_member": "mediapipe/ex6/ex6_Petra_cam3_3d.npy",
            "shape": list(a.shape), "dtype": str(a.dtype),
            "coordinates_all_finite": bool(np.isfinite(a[:, :, :3]).all()),
            "visibility_all_finite": bool(np.isfinite(a[:, :, 3]).all()),
            "visibility_min": float(np.nanmin(a[:, :, 3])),
            "visibility_max": float(np.nanmax(a[:, :, 3])),
            "fraction_visibility_below_half": float(np.mean(a[:, :, 3] < .5)),
            "annotated_repetitions": len(related),
            "repetition_end_outside_array": sum(int(r["last_frame"]) >= len(a) for r in related),
            "archive_member_crc32": "13638f92",
            "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
            "note": "Изучен один файл, извлечённый HTTP Range из архива; это не аудит всех поз и не оценка модели.",
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/rehab26_audit_2026-11-05.json"))
    args = parser.parse_args()
    report = audit(args.data_dir, args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Проверено {report['squat_physical_repetitions']} физических приседаний; основной поднабор: {report['selected_three_category_physical_repetitions']}. Отчёт: {args.output}")


if __name__ == "__main__":
    main()
