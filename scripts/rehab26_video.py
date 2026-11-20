"""Извлечение MediaPipe world-поз из RGB-видео для REHAB26-классификатора.

Модуль не импортирует MediaPipe/OpenCV до фактической обработки видео, чтобы
геометрический baseline и его тесты продолжали работать без видеозависимостей.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


N_LANDMARKS = 33
POSE_WIDTH = 4
VIDEO_INPUT_CONTRACT = "MediaPipe Pose Landmarker world landmarks: (frames, 33, 4), xyz + visibility"


@dataclass(frozen=True)
class VideoPoseResult:
    poses: np.ndarray
    detected: np.ndarray
    fps: float
    width: int
    height: int
    frame_count: int
    mediapipe_version: str
    opencv_version: str


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_dependencies():
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )
    except ImportError as error:
        raise RuntimeError(
            "Не установлены видеозависимости. Выполните: "
            "python -m pip install -r requirements-video.txt"
        ) from error
    return cv2, mp, BaseOptions, PoseLandmarker, PoseLandmarkerOptions, RunningMode


def world_landmarks_to_array(world_landmarks: list[Any]) -> np.ndarray:
    """Преобразовать 33 MediaPipe landmark в xyz + visibility."""
    if len(world_landmarks) != N_LANDMARKS:
        raise ValueError(f"invalid_landmark_count: ожидалось 33, получено {len(world_landmarks)}")
    result = np.empty((N_LANDMARKS, POSE_WIDTH), dtype=np.float64)
    for index, landmark in enumerate(world_landmarks):
        result[index] = (
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
            float(getattr(landmark, "visibility", 0.0)),
        )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite_landmarks: MediaPipe вернул нечисловые координаты")
    return result


def extract_world_poses(
    video_path: Path,
    landmarker_model_path: Path,
    *,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> VideoPoseResult:
    """Обработать видео в режиме VIDEO и сохранить пропуски как NaN-кадры."""
    video_path = Path(video_path)
    landmarker_model_path = Path(landmarker_model_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if not landmarker_model_path.is_file():
        raise FileNotFoundError(
            f"Модель Pose Landmarker не найдена: {landmarker_model_path}. "
            "Скачайте официальный pose_landmarker_full.task по инструкции в README."
        )

    cv2, mp, BaseOptions, PoseLandmarker, PoseLandmarkerOptions, RunningMode = _video_dependencies()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"video_open_failed: OpenCV не открыл {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("invalid_video_fps: частота кадров отсутствует или некорректна")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(landmarker_model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    frames: list[np.ndarray] = []
    detected: list[bool] = []
    frame_index = 0
    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                result = landmarker.detect_for_video(image, timestamp_ms)
                if result.pose_world_landmarks:
                    frames.append(world_landmarks_to_array(result.pose_world_landmarks[0]))
                    detected.append(True)
                else:
                    frames.append(np.full((N_LANDMARKS, POSE_WIDTH), np.nan, dtype=np.float64))
                    detected.append(False)
                frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError("empty_video: видео не содержит читаемых кадров")
    return VideoPoseResult(
        poses=np.stack(frames),
        detected=np.asarray(detected, dtype=bool),
        fps=fps,
        width=width,
        height=height,
        frame_count=len(frames),
        mediapipe_version=str(getattr(mp, "__version__", "unknown")),
        opencv_version=str(getattr(cv2, "__version__", "unknown")),
    )


def resolve_manual_bounds(
    frame_count: int,
    fps: float,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[int, int] | None:
    """Вернуть включительные границы повтора либо None для extract-only режима."""
    frame_mode = start_frame is not None or end_frame is not None
    time_mode = start_sec is not None or end_sec is not None
    if frame_mode and time_mode:
        raise ValueError("mixed_bounds: задайте границы либо в кадрах, либо в секундах")
    if not frame_mode and not time_mode:
        return None
    if frame_mode:
        if start_frame is None or end_frame is None:
            raise ValueError("incomplete_frame_bounds: нужны одновременно --start-frame и --end-frame")
        start, end = int(start_frame), int(end_frame)
    else:
        if start_sec is None or end_sec is None:
            raise ValueError("incomplete_time_bounds: нужны одновременно --start-sec и --end-sec")
        if start_sec < 0 or end_sec < 0:
            raise ValueError("negative_time_bounds: время не может быть отрицательным")
        start = int(np.floor(start_sec * fps))
        end = int(np.ceil(end_sec * fps)) - 1
    if start < 0 or end < start or end >= frame_count:
        raise ValueError(
            f"invalid_bounds: допустимы 0 <= start <= end < {frame_count}, получено {start}:{end}"
        )
    return start, end


def interpolate_short_gaps(
    poses: np.ndarray,
    *,
    max_gap_frames: int = 3,
    min_detection_coverage: float = 0.9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Интерполировать только короткие внутренние пропуски полного повтора.

    Пропуск первого/последнего кадра и длинный разрыв приводят к техническому
    отказу: границы не расширяются и сложные случаи не удаляются молча.
    """
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (N_LANDMARKS, POSE_WIDTH) or len(array) < 2:
        raise ValueError("invalid_pose_shape: ожидается (frames>=2, 33, 4)")
    if max_gap_frames < 0:
        raise ValueError("invalid_max_gap: max_gap_frames должен быть >= 0")
    if not 0 < min_detection_coverage <= 1:
        raise ValueError("invalid_coverage_threshold: порог должен быть в (0, 1]")

    detected = np.isfinite(array).all(axis=(1, 2))
    coverage = float(detected.mean())
    if coverage < min_detection_coverage:
        raise ValueError(
            f"insufficient_pose_coverage: {coverage:.3f} < {min_detection_coverage:.3f}"
        )
    if not detected[0] or not detected[-1]:
        raise ValueError("missing_boundary_pose: на начале или конце выбранного повтора нет позы")

    repaired = array.copy()
    gaps: list[dict[str, int]] = []
    index = 0
    while index < len(repaired):
        if detected[index]:
            index += 1
            continue
        gap_start = index
        while index < len(repaired) and not detected[index]:
            index += 1
        gap_end = index - 1
        gap_length = gap_end - gap_start + 1
        if gap_length > max_gap_frames:
            raise ValueError(f"pose_gap_too_long: кадры {gap_start}:{gap_end}, длина {gap_length}")
        left, right = gap_start - 1, gap_end + 1
        for position in range(gap_start, gap_end + 1):
            alpha = (position - left) / (right - left)
            repaired[position] = (1.0 - alpha) * repaired[left] + alpha * repaired[right]
        gaps.append({"start_relative_frame": gap_start, "end_relative_frame": gap_end, "length": gap_length})

    return repaired, {
        "detection_coverage": coverage,
        "detected_frames": int(detected.sum()),
        "total_frames": int(len(array)),
        "interpolated_frames": int((~detected).sum()),
        "interpolated_gaps": gaps,
        "max_gap_frames": int(max_gap_frames),
        "min_detection_coverage": float(min_detection_coverage),
    }
