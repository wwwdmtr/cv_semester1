"""RGB-видео → MediaPipe world-позы → ручной полный повтор → REHAB26-модель."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from predict_rehab26 import predict
from rehab26_video import (
    VIDEO_INPUT_CONTRACT,
    extract_world_poses,
    file_sha256,
    interpolate_short_gaps,
    resolve_manual_bounds,
)


DEFAULT_CLASSIFIER = Path("models/rehab26_baseline/squat_geometry.joblib")
DEFAULT_POSE_MODEL = Path("models/mediapipe/pose_landmarker_full.task")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="RGB-видео с одним человеком и полным телом в кадре")
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL,
                        help="официальный MediaPipe pose_landmarker_full.task")
    parser.add_argument("--model", type=Path, default=DEFAULT_CLASSIFIER,
                        help="доверенный bundle основной REHAB26-модели")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="по умолчанию data/processed/own_squats/<имя видео>")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int, help="включительная граница")
    parser.add_argument("--start-sec", type=float)
    parser.add_argument("--end-sec", type=float, help="граница времени справа")
    parser.add_argument("--max-gap-frames", type=int, default=3)
    parser.add_argument("--min-detection-coverage", type=float, default=0.9)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("data/processed/own_squats") / args.video.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = extract_world_poses(args.video, args.pose_model)
    raw_path = output_dir / "world_poses_raw.npy"
    np.save(raw_path, extracted.poses, allow_pickle=False)
    bounds = resolve_manual_bounds(
        extracted.frame_count,
        extracted.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
    metadata = {
        "status": "poses_extracted" if bounds is None else "processing_manual_repetition",
        "input_contract": VIDEO_INPUT_CONTRACT,
        "video": {
            "path": str(args.video),
            "sha256": file_sha256(args.video),
            "fps": extracted.fps,
            "frame_count": extracted.frame_count,
            "width": extracted.width,
            "height": extracted.height,
        },
        "pose_landmarker": {
            "path": str(args.pose_model),
            "sha256": file_sha256(args.pose_model),
            "mediapipe_version": extracted.mediapipe_version,
            "opencv_version": extracted.opencv_version,
            "num_poses": 1,
            "running_mode": "VIDEO",
        },
        "whole_video_pose_coverage": float(extracted.detected.mean()),
        "raw_poses": str(raw_path),
        "manual_bounds": None,
        "result": None,
        "limitations": [
            "Это development-проверка, а не оценка качества на независимом test.",
            "Границы повтора заданы вручную; автоматическое выделение ещё не используется.",
            "Классификатор проверяет только две категории ошибок из REHAB26.",
            "Score классификатора не является откалиброванной вероятностью правильности.",
        ],
    }

    if bounds is None:
        metadata["next_step"] = (
            "Повторите команду с --start-frame/--end-frame или --start-sec/--end-sec, "
            "чтобы оценить один полный повтор."
        )
        _write_json(output_dir / "metadata.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    start, end = bounds
    selected = extracted.poses[start:end + 1]
    metadata["manual_bounds"] = {
        "start_frame_inclusive": start,
        "end_frame_inclusive": end,
        "start_sec": start / extracted.fps,
        "end_sec_exclusive": (end + 1) / extracted.fps,
    }
    try:
        repaired, quality = interpolate_short_gaps(
            selected,
            max_gap_frames=args.max_gap_frames,
            min_detection_coverage=args.min_detection_coverage,
        )
    except ValueError as error:
        metadata.update({"status": "technical_rejection", "technical_rejection_reason": str(error)})
        _write_json(output_dir / "metadata.json", metadata)
        raise
    repetition_path = output_dir / "repetition_world_poses.npy"
    np.save(repetition_path, repaired, allow_pickle=False)
    if not args.model.is_file():
        error = FileNotFoundError(
            f"Классификатор не найден: {args.model}. Передайте доверенный squat_geometry.joblib; "
            "код не переобучает и не восстанавливает веса автоматически."
        )
        metadata.update({
            "status": "classifier_missing",
            "selected_pose_quality": quality,
            "repetition_poses": str(repetition_path),
            "technical_rejection_reason": str(error),
        })
        _write_json(output_dir / "metadata.json", metadata)
        raise error
    prediction = predict(args.model, repaired)
    result_path = output_dir / "result.json"
    _write_json(result_path, prediction)
    metadata.update({
        "status": "completed",
        "selected_pose_quality": quality,
        "repetition_poses": str(repetition_path),
        "classifier": {"path": str(args.model), "sha256": file_sha256(args.model)},
        "result": prediction,
        "result_file": str(result_path),
    })
    _write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
