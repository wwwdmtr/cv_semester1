"""Автоматически предложить границы приседаний в сохранённых MediaPipe world-позах."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rehab26_repetition_detector import RepetitionDetectorConfig, detect_repetitions
from rehab26_video import file_sha256, interpolate_short_gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path, help="NPY всего видео: (кадры, 33, 4), пропуски допустимы как NaN")
    parser.add_argument("--fps", type=float, required=True, help="частота кадров исходного видео")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-gap-frames", type=int, default=3)
    parser.add_argument("--min-detection-coverage", type=float, default=0.9)
    args = parser.parse_args()

    raw = np.load(args.poses, allow_pickle=False)
    repaired, pose_quality = interpolate_short_gaps(
        raw,
        max_gap_frames=args.max_gap_frames,
        min_detection_coverage=args.min_detection_coverage,
    )
    config = RepetitionDetectorConfig()
    detection = detect_repetitions(repaired, args.fps, config)
    result = {
        "status": "automatic_segmentation_candidate",
        "input": {
            "path": str(args.poses),
            "sha256": file_sha256(args.poses),
            "shape": list(raw.shape),
            "fps": args.fps,
        },
        "pose_quality": pose_quality,
        "segmentation": detection.to_dict(),
        "limitations": [
            "Пороги пока проверены только синтетическими тестами и не настроены по development-видео.",
            "Алгоритм выделяет движение, но не определяет правильность техники и не заменяет классификатор.",
            "Малая глубина не используется как причина удалить повтор.",
            "Границы нужно сравнить с ручной разметкой до использования в приложении.",
        ],
    }
    output = args.output or args.poses.with_name("automatic_repetitions.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
