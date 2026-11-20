"""RGB-видео → MediaPipe world-позы → полные повторы → REHAB26-модель."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from predict_rehab26 import predict
from rehab26_repetition_detector import RepetitionDetectorConfig, detect_repetitions
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


def _require_classifier(model_path: Path, metadata: dict, **details) -> None:
    if model_path.is_file():
        return
    error = FileNotFoundError(
        f"Классификатор не найден: {model_path}. Передайте доверенный squat_geometry.joblib; "
        "код не переобучает и не восстанавливает веса автоматически."
    )
    metadata.update({
        "status": "classifier_missing",
        "technical_rejection_reason": str(error),
        **details,
    })
    raise error


def _predict_automatic_repetitions(
    poses: np.ndarray,
    fps: float,
    model_path: Path,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    detection = detect_repetitions(
        poses,
        fps,
        RepetitionDetectorConfig(),
        search_stable_stance=True,
    )
    predictions = []
    for number, repetition in enumerate(detection.repetitions, start=1):
        selected = poses[repetition.start_frame:repetition.end_frame + 1]
        pose_path = output_dir / f"repetition_{number:03d}_world_poses.npy"
        result_path = output_dir / f"repetition_{number:03d}_result.json"
        np.save(pose_path, selected, allow_pickle=False)
        result = predict(model_path, selected)
        _write_json(result_path, result)
        predictions.append({
            "number": number,
            "bounds": repetition.__dict__,
            "poses": str(pose_path),
            "result_file": str(result_path),
            "result": result,
        })
    return detection.to_dict(), predictions


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
    parser.add_argument("--automatic-repetitions", action="store_true",
                        help="автоматически найти и оценить все завершённые приседания")
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
    if args.automatic_repetitions and bounds is not None:
        raise ValueError(
            "mixed_segmentation_modes: автоматический режим нельзя смешивать с ручными границами"
        )
    metadata = {
        "status": (
            "processing_automatic_repetitions" if args.automatic_repetitions
            else "poses_extracted" if bounds is None
            else "processing_manual_repetition"
        ),
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
        "automatic_segmentation": None,
        "repetitions": [],
        "result": None,
        "limitations": [
            "Это development-проверка, а не оценка качества на независимом test.",
            "Автоматические границы являются development-версией и требуют сравнения с ручной разметкой.",
            "Классификатор проверяет только две категории ошибок из REHAB26.",
            "Score классификатора не является откалиброванной вероятностью правильности.",
        ],
    }

    if args.automatic_repetitions:
        try:
            repaired, quality = interpolate_short_gaps(
                extracted.poses,
                max_gap_frames=args.max_gap_frames,
                min_detection_coverage=args.min_detection_coverage,
            )
            _require_classifier(args.model, metadata, whole_video_pose_quality=quality)
            segmentation, repetitions = _predict_automatic_repetitions(
                repaired,
                extracted.fps,
                args.model,
                output_dir,
            )
        except (ValueError, FileNotFoundError) as error:
            if metadata["status"] != "classifier_missing":
                metadata.update({
                    "status": "technical_rejection",
                    "technical_rejection_reason": str(error),
                })
            _write_json(output_dir / "metadata.json", metadata)
            raise
        metadata.update({
            "status": "completed" if repetitions else "no_complete_repetitions",
            "whole_video_pose_quality": quality,
            "classifier": {"path": str(args.model), "sha256": file_sha256(args.model)},
            "automatic_segmentation": segmentation,
            "repetitions": repetitions,
            "result": {
                "complete_repetitions": len(repetitions),
                "rejected_candidates": len(segmentation["rejected_candidates"]),
            },
        })
        _write_json(output_dir / "metadata.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

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
    try:
        _require_classifier(
            args.model,
            metadata,
            selected_pose_quality=quality,
            repetition_poses=str(repetition_path),
        )
    except FileNotFoundError:
        _write_json(output_dir / "metadata.json", metadata)
        raise
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
