"""Прозрачное выделение повторов приседания по последовательности world-поз.

Алгоритм сегментирует движение во времени и не оценивает правильность техники.
Порог правильной глубины намеренно не используется: иначе класс недостаточной
глубины мог бы исчезнуть ещё до классификации.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rehab26_features import extract_signals


@dataclass(frozen=True)
class RepetitionDetectorConfig:
    calibration_seconds: float = 0.75
    smoothing_seconds: float = 0.15
    transition_seconds: float = 0.10
    pre_roll_seconds: float = 0.15
    post_roll_seconds: float = 0.15
    min_repetition_seconds: float = 0.50
    max_repetition_seconds: float = 8.0
    start_knee_flexion_deg: float = 12.0
    end_knee_flexion_deg: float = 8.0
    min_peak_knee_flexion_deg: float = 20.0
    start_compression: float = 0.03
    end_compression: float = 0.02
    min_peak_compression: float = 0.06
    max_calibration_knee_range_deg: float = 8.0
    max_calibration_distance_range: float = 0.05
    min_standing_knee_angle_deg: float = 150.0

    def validate(self) -> None:
        positive = {
            "calibration_seconds": self.calibration_seconds,
            "smoothing_seconds": self.smoothing_seconds,
            "transition_seconds": self.transition_seconds,
            "min_repetition_seconds": self.min_repetition_seconds,
            "max_repetition_seconds": self.max_repetition_seconds,
            "start_knee_flexion_deg": self.start_knee_flexion_deg,
            "min_peak_knee_flexion_deg": self.min_peak_knee_flexion_deg,
            "start_compression": self.start_compression,
            "min_peak_compression": self.min_peak_compression,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("invalid_detector_config: основные интервалы и пороги должны быть > 0")
        if self.pre_roll_seconds < 0 or self.post_roll_seconds < 0:
            raise ValueError("invalid_detector_config: pre/post-roll не могут быть отрицательными")
        if not 0 <= self.end_knee_flexion_deg < self.start_knee_flexion_deg < self.min_peak_knee_flexion_deg:
            raise ValueError("invalid_detector_config: нужны end_knee < start_knee < min_peak_knee")
        if not 0 <= self.end_compression < self.start_compression < self.min_peak_compression:
            raise ValueError("invalid_detector_config: нужны end_compression < start_compression < min_peak_compression")
        if self.min_repetition_seconds >= self.max_repetition_seconds:
            raise ValueError("invalid_detector_config: min_repetition_seconds должен быть меньше max")
        if self.max_calibration_knee_range_deg <= 0 or self.max_calibration_distance_range <= 0:
            raise ValueError("invalid_detector_config: допуски калибровки должны быть > 0")
        if not 0 < self.min_standing_knee_angle_deg < 180:
            raise ValueError("invalid_detector_config: угол стойки должен быть в (0, 180)")


@dataclass(frozen=True)
class DetectedRepetition:
    start_frame: int
    end_frame: int
    bottom_frame: int
    active_start_frame: int
    return_frame: int
    duration_seconds: float
    peak_knee_flexion_deg: float
    peak_compression: float


@dataclass(frozen=True)
class RepetitionDetectionResult:
    repetitions: tuple[DetectedRepetition, ...]
    rejected_candidates: tuple[dict[str, Any], ...]
    baseline_knee_angle_deg: float
    baseline_hip_to_ankle_distance: float
    calibration_start_frame: int
    calibration_frames: int
    smoothing_frames: int
    transition_frames: int
    config: RepetitionDetectorConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "repetitions": [asdict(item) for item in self.repetitions],
            "rejected_candidates": list(self.rejected_candidates),
            "baseline": {
                "knee_angle_deg": self.baseline_knee_angle_deg,
                "hip_to_ankle_distance_in_torso_lengths": self.baseline_hip_to_ankle_distance,
                "start_frame": self.calibration_start_frame,
                "calibration_frames": self.calibration_frames,
            },
            "smoothing_frames": self.smoothing_frames,
            "transition_frames": self.transition_frames,
            "config": asdict(self.config),
        }


def _frames(seconds: float, fps: float, *, minimum: int = 1) -> int:
    return max(minimum, int(round(seconds * fps)))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def find_stable_stance_start_from_signals(
    knee_angle_radians: np.ndarray,
    hip_to_ankle_distance: np.ndarray,
    fps: float,
    config: RepetitionDetectorConfig | None = None,
) -> int:
    """Найти первую устойчивую почти выпрямленную стойку для калибровки.

    Проверка абсолютного угла колена не оценивает глубину приседания. Она не
    позволяет принять за исходную стойку устойчивый кадр в нижней точке.
    """
    config = config or RepetitionDetectorConfig()
    config.validate()
    knee = np.asarray(knee_angle_radians, dtype=np.float64)
    distance = np.asarray(hip_to_ankle_distance, dtype=np.float64)
    if knee.ndim != 1 or distance.ndim != 1 or len(knee) != len(distance):
        raise ValueError("invalid_signal_shape: ожидаются два одномерных сигнала одинаковой длины")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid_fps: fps должен быть положительным")
    if not np.isfinite(knee).all() or not np.isfinite(distance).all():
        raise ValueError("nonfinite_signals: перед поиском стойки нужно обработать пропуски позы")

    calibration_frames = _frames(config.calibration_seconds, fps, minimum=3)
    if len(knee) <= calibration_frames + 2:
        raise ValueError("video_too_short: недостаточно кадров для поиска стойки")
    smoothing_frames = _frames(config.smoothing_seconds, fps)
    knee_smooth = _smooth(knee, smoothing_frames)
    distance_smooth = _smooth(distance, smoothing_frames)
    for start in range(0, len(knee) - calibration_frames + 1):
        stop = start + calibration_frames
        knee_window = knee_smooth[start:stop]
        distance_window = distance_smooth[start:stop]
        if float(np.rad2deg(np.median(knee_window))) < config.min_standing_knee_angle_deg:
            continue
        if float(np.rad2deg(np.ptp(knee_window))) > config.max_calibration_knee_range_deg:
            continue
        if float(np.ptp(distance_window)) > config.max_calibration_distance_range:
            continue
        return start
    raise ValueError("stable_stance_not_found: не найдена устойчивая стойка с почти прямыми коленями")


def detect_repetitions_from_signals(
    knee_angle_radians: np.ndarray,
    hip_to_ankle_distance: np.ndarray,
    fps: float,
    config: RepetitionDetectorConfig | None = None,
    *,
    calibration_start_frame: int = 0,
) -> RepetitionDetectionResult:
    """Выделить повторения по сгибанию коленей и сжатию вертикальной геометрии."""
    config = config or RepetitionDetectorConfig()
    config.validate()
    knee = np.asarray(knee_angle_radians, dtype=np.float64)
    distance = np.asarray(hip_to_ankle_distance, dtype=np.float64)
    if knee.ndim != 1 or distance.ndim != 1 or len(knee) != len(distance):
        raise ValueError("invalid_signal_shape: ожидаются два одномерных сигнала одинаковой длины")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid_fps: fps должен быть положительным")
    if not np.isfinite(knee).all() or not np.isfinite(distance).all():
        raise ValueError("nonfinite_signals: перед сегментацией нужно обработать пропуски позы")

    calibration_frames = _frames(config.calibration_seconds, fps, minimum=3)
    if calibration_start_frame < 0:
        raise ValueError("invalid_calibration_start: номер кадра должен быть >= 0")
    calibration_end = calibration_start_frame + calibration_frames
    if len(knee) <= calibration_end + 2:
        raise ValueError("video_too_short: недостаточно кадров после начальной калибровки")
    smoothing_frames = _frames(config.smoothing_seconds, fps)
    transition_frames = _frames(config.transition_seconds, fps)
    knee_smooth = _smooth(knee, smoothing_frames)
    distance_smooth = _smooth(distance, smoothing_frames)

    calibration_knee = knee_smooth[calibration_start_frame:calibration_end]
    calibration_distance = distance_smooth[calibration_start_frame:calibration_end]
    knee_range_deg = float(np.rad2deg(np.ptp(calibration_knee)))
    distance_range = float(np.ptp(calibration_distance))
    if knee_range_deg > config.max_calibration_knee_range_deg:
        raise ValueError(
            f"unstable_calibration_knee: диапазон {knee_range_deg:.2f}° больше "
            f"{config.max_calibration_knee_range_deg:.2f}°"
        )
    if distance_range > config.max_calibration_distance_range:
        raise ValueError(
            f"unstable_calibration_distance: диапазон {distance_range:.4f} больше "
            f"{config.max_calibration_distance_range:.4f}"
        )

    baseline_knee = float(np.median(calibration_knee))
    baseline_distance = float(np.median(calibration_distance))
    knee_flexion_deg = np.rad2deg(baseline_knee - knee_smooth)
    compression = baseline_distance - distance_smooth
    start_condition = (
        (knee_flexion_deg >= config.start_knee_flexion_deg)
        & (compression >= config.start_compression)
    )
    return_condition = (
        (knee_flexion_deg <= config.end_knee_flexion_deg)
        & (compression <= config.end_compression)
    )

    min_frames = _frames(config.min_repetition_seconds, fps)
    max_frames = _frames(config.max_repetition_seconds, fps)
    pre_roll = _frames(config.pre_roll_seconds, fps, minimum=0)
    post_roll = _frames(config.post_roll_seconds, fps, minimum=0)
    repetitions: list[DetectedRepetition] = []
    rejected: list[dict[str, Any]] = []
    state = "standing"
    start_run = 0
    return_run = 0
    active_start: int | None = None

    for frame in range(calibration_end, len(knee)):
        if state == "standing":
            start_run = start_run + 1 if start_condition[frame] else 0
            if start_run >= transition_frames:
                active_start = frame - transition_frames + 1
                state = "active"
                return_run = 0
            continue

        if state == "waiting_return":
            return_run = return_run + 1 if return_condition[frame] else 0
            if return_run >= transition_frames:
                state = "standing"
                start_run = 0
                return_run = 0
                active_start = None
            continue

        if active_start is None:
            raise RuntimeError("internal_detector_state: active без начала")
        if frame - active_start + 1 > max_frames:
            rejected.append({
                "reason": "too_long",
                "active_start_frame": active_start,
                "observed_until_frame": frame,
            })
            state = "waiting_return"
            return_run = 0
            continue

        return_run = return_run + 1 if return_condition[frame] else 0
        if return_run < transition_frames:
            continue

        return_frame = frame - transition_frames + 1
        active_length = return_frame - active_start + 1
        segment = slice(active_start, return_frame + 1)
        peak_knee = float(np.max(knee_flexion_deg[segment]))
        peak_compression = float(np.max(compression[segment]))
        combined = (
            knee_flexion_deg[segment] / config.min_peak_knee_flexion_deg
            + compression[segment] / config.min_peak_compression
        )
        bottom_frame = active_start + int(np.argmax(combined))
        if active_length < min_frames:
            rejected.append({
                "reason": "too_short",
                "active_start_frame": active_start,
                "return_frame": return_frame,
                "duration_seconds": active_length / fps,
            })
        elif peak_knee < config.min_peak_knee_flexion_deg or peak_compression < config.min_peak_compression:
            rejected.append({
                "reason": "small_motion",
                "active_start_frame": active_start,
                "return_frame": return_frame,
                "peak_knee_flexion_deg": peak_knee,
                "peak_compression": peak_compression,
            })
        else:
            repetitions.append(DetectedRepetition(
                start_frame=max(calibration_end, active_start - pre_roll),
                end_frame=min(len(knee) - 1, return_frame + post_roll),
                bottom_frame=bottom_frame,
                active_start_frame=active_start,
                return_frame=return_frame,
                duration_seconds=active_length / fps,
                peak_knee_flexion_deg=peak_knee,
                peak_compression=peak_compression,
            ))
        state = "standing"
        start_run = 0
        return_run = 0
        active_start = None

    if state == "active" and active_start is not None:
        rejected.append({
            "reason": "incomplete_at_video_end",
            "active_start_frame": active_start,
            "observed_until_frame": len(knee) - 1,
        })

    return RepetitionDetectionResult(
        repetitions=tuple(repetitions),
        rejected_candidates=tuple(rejected),
        baseline_knee_angle_deg=float(np.rad2deg(baseline_knee)),
        baseline_hip_to_ankle_distance=baseline_distance,
        calibration_start_frame=calibration_start_frame,
        calibration_frames=calibration_frames,
        smoothing_frames=smoothing_frames,
        transition_frames=transition_frames,
        config=config,
    )


def detect_repetitions(
    poses: np.ndarray,
    fps: float,
    config: RepetitionDetectorConfig | None = None,
    *,
    search_stable_stance: bool = False,
) -> RepetitionDetectionResult:
    """Извлечь два наблюдаемых сигнала из world-поз и сегментировать повторы."""
    signals = extract_signals(poses)
    mean_knee_angle = np.mean(signals[:, :2], axis=1)
    hip_to_ankle_distance = signals[:, 6]
    calibration_start = 0
    if search_stable_stance:
        calibration_start = find_stable_stance_start_from_signals(
            mean_knee_angle,
            hip_to_ankle_distance,
            fps,
            config,
        )
    return detect_repetitions_from_signals(
        mean_knee_angle,
        hip_to_ankle_distance,
        fps,
        config,
        calibration_start_frame=calibration_start,
    )
