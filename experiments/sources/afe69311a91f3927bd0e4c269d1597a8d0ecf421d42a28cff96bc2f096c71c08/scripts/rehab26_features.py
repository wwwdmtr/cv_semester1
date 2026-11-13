"""Фиксированный исходный набор геометрических признаков MediaPipe 3D.

Видимость сохраняется в исходных данных, но не участвует в признаках и
фильтрации этого baseline: её полезность проверяется отдельным опытом.
"""

import numpy as np


FEATURE_VERSION = "rehab26_global_geometry_v1"
QUANTILES = (0., .1, .5, .9, 1.)
SIGNAL_NAMES = (
    "right_knee_angle", "left_knee_angle", "right_hip_angle", "left_hip_angle",
    "knee_distance", "ankle_distance", "hip_to_ankle_distance",
)
FEATURE_NAMES = [f"{signal}_q{q:g}" for q in QUANTILES for signal in SIGNAL_NAMES]
FEATURE_METADATA = {
    "version": FEATURE_VERSION,
    "input": "MediaPipe world landmarks; shape (frames, 33, 4); xyz in meters + visibility",
    "quantiles": list(QUANTILES), "signals": list(SIGNAL_NAMES),
    "angle_units": "radians", "distance_units": "median torso length",
    "normalization": "median distance between shoulder midpoint and hip midpoint over the observed repetition",
    "visibility_used_in_predictors": False,
    "temporal_order_used": False,
    "joint_indices": {"left_shoulder": 11, "right_shoulder": 12, "left_hip": 23,
                      "right_hip": 24, "left_knee": 25, "right_knee": 26,
                      "left_ankle": 27, "right_ankle": 28},
}


def _angle(first, center, last):
    a, b = first - center, last - center
    norm_a, norm_b = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if np.any(norm_a <= 1e-10) or np.any(norm_b <= 1e-10):
        raise ValueError("degenerate_relevant_limb: нулевая длина сегмента сустава")
    cosine = np.sum(a * b, axis=-1) / (norm_a * norm_b)
    return np.arccos(np.clip(cosine, -1., 1.))


def extract_signals(poses):
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (33, 4) or len(array) < 2:
        raise ValueError("invalid_shape: ожидается (frames>=2, 33, 4)")
    xyz = array[:, :, :3]
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite_coordinates: есть пропуски или бесконечные координаты")
    shoulders = (xyz[:, 11] + xyz[:, 12]) / 2
    hips = (xyz[:, 23] + xyz[:, 24]) / 2
    ankles = (xyz[:, 27] + xyz[:, 28]) / 2
    scale = np.median(np.linalg.norm(shoulders - hips, axis=-1))
    if not np.isfinite(scale) or scale <= 1e-10:
        raise ValueError("degenerate_torso: нулевая медианная длина корпуса")
    signals = np.stack([
        _angle(xyz[:, 24], xyz[:, 26], xyz[:, 28]),
        _angle(xyz[:, 23], xyz[:, 25], xyz[:, 27]),
        _angle(shoulders, xyz[:, 24], xyz[:, 26]),
        _angle(shoulders, xyz[:, 23], xyz[:, 25]),
        np.linalg.norm(xyz[:, 25] - xyz[:, 26], axis=-1) / scale,
        np.linalg.norm(xyz[:, 27] - xyz[:, 28], axis=-1) / scale,
        np.linalg.norm(hips - ankles, axis=-1) / scale,
    ], axis=1)
    if not np.isfinite(signals).all():
        raise ValueError("nonfinite_signals: не удалось вычислить геометрию")
    return signals


def extract_features(poses):
    """Вернуть 35 признаков; порядок совпадает с FEATURE_NAMES."""
    return np.quantile(extract_signals(poses), QUANTILES, axis=0).ravel()
