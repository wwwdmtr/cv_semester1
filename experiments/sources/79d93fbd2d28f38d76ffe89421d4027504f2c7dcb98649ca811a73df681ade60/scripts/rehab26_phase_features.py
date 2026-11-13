"""Одна фиксированная добавка фазы к исходным 35 признакам REHAB26.

Семь добавок — mean(bottom)-mean(edges) для тех же семи 3D-сигналов.
Нижняя фаза определяется по наблюдаемому углу коленей, без меток класса.
Это статистики фазы и границ, а не модель полного временного порядка.
"""

import numpy as np

from rehab26_features import (
    FEATURE_METADATA as BASE_METADATA,
    FEATURE_NAMES as BASE_NAMES,
    FEATURE_VERSION as BASE_VERSION,
    SIGNAL_NAMES,
    extract_features as extract_global_features,
    extract_signals,
)


FEATURE_VERSION = "rehab26_global_geometry_plus_phase_relative_v1"
BOTTOM_QUANTILE = 0.20
EDGE_FRACTION = 0.15
FEATURE_NAMES = list(BASE_NAMES) + [f"{signal}_mean_bottom_minus_edges" for signal in SIGNAL_NAMES]
FEATURE_METADATA = {
    **BASE_METADATA,
    "version": FEATURE_VERSION,
    "dimension": 42,
    "baseline_prefix_version": BASE_VERSION,
    "baseline_prefix_dimension": 35,
    "additional_signals": list(SIGNAL_NAMES),
    "additional_recipe": "mean(signal[bottom])-mean(signal[edges]) for each of seven baseline signals",
    "bottom_recipe": "mean(right_knee_angle,left_knee_angle) <= np.quantile(mean_knee_angle,0.20)",
    "bottom_quantile": BOTTOM_QUANTILE,
    "bottom_ties": "all values equal to cutoff included; bottom may contain more than 20 percent of frames",
    "edges_recipe": "unique boolean union of first and last max(1,round(0.15*T)) frames",
    "edge_fraction": EDGE_FRACTION,
    "edge_rounding": "Python round (ties to even), minimum one frame at each end",
    "phase_edge_overlap": "allowed, including two-frame sequences; no duplicated edge frames",
    "minimum_frames": 2,
    "visibility_used_in_predictors": False,
    "labels_used_in_predictors": False,
    "temporal_order_used": True,
    "temporal_scope": "edge membership only; invariant to full sequence reversal, no descent/ascent sequence model",
    "standing_assumption": "edges are only a proxy for initial/final stance; complete repetition input required",
}


def phase_masks(signals):
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != 7 or len(signals) < 2 or not np.isfinite(signals).all():
        raise ValueError("expected_finite_frames_ge_2_by_7_signals")
    mean_knee_angle = signals[:, :2].mean(axis=1)
    bottom = mean_knee_angle <= np.quantile(mean_knee_angle, BOTTOM_QUANTILE)
    edge_count = max(1, round(EDGE_FRACTION * len(signals)))
    edges = np.zeros(len(signals), dtype=bool)
    edges[:edge_count] = True
    edges[-edge_count:] = True
    return bottom, edges


def extract_features(poses):
    """Вернуть 42 числа; первые 35 побитно соответствуют старому extractor."""
    prefix = extract_global_features(poses)
    signals = extract_signals(poses)
    bottom, edges = phase_masks(signals)
    differences = signals[bottom].mean(axis=0) - signals[edges].mean(axis=0)
    result = np.concatenate([prefix, differences])
    if result.shape != (42,) or not np.isfinite(result).all():
        raise ValueError("invalid_phase_relative_feature_vector")
    return result
