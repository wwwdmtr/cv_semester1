"""Маленький CPU temporal CNN над семью геометрическими сигналами.

Вход — полный повтор после индивидуальной интерполяции в 64 точки.
Это собственный ограниченный эксперимент, не воспроизведение GCN/RGB.
Нормировка каналов и балансировка вычисляются только на переданном train.
Порядок кадров меняет обёртка опыта, а не сам классификатор.
"""

import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from rehab26_features import SIGNAL_NAMES, extract_signals


FEATURE_VERSION = "rehab26_temporal_geometry64_v1"
N_FRAMES = 64
TRAIN_EPOCHS = 120
ENSEMBLE_SEEDS = (0, 1, 2)
SHUFFLE_SEED = 1729
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
FEATURE_METADATA = {
    "version": FEATURE_VERSION,
    "shape": [N_FRAMES, 7],
    "signals": list(SIGNAL_NAMES),
    "input": "complete MediaPipe world pose repetition (T>=2,33,4)",
    "resampling": "linear interpolation of each signal on normalized [0,1] time, endpoints included",
    "duration_used": False,
    "visibility_used": False,
    "channel_scaling": "weighted mean and population standard deviation over train observations and time; constant scale=1",
    "architecture": "Conv1d(7,16,5,pad2),ReLU,Conv1d(16,16,5,pad2),ReLU,AdaptiveAvgPool1d(4),Flatten,Linear(64,3)",
    "temporal_scope": "local adjacency plus four coarse ordered temporal pooling bins",
    "epochs": TRAIN_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "ensemble_seeds": list(ENSEMBLE_SEEDS),
    "ensemble_recipe": "mean of three member probabilities; not logits or votes",
    "selection": "only balanced false/true; no validation/test early stopping",
    "shuffle_recipe": "seed1729 plus stable physical_id hash, one permutation shared by all camera observations of the physical repetition",
    "device": "CPU, one thread, deterministic algorithms",
}


def resample_signal_array(signals):
    """Сохранить форму/концы семи сигналов при линейном T→64."""
    signals = np.asarray(signals, dtype=np.float64)
    if (signals.ndim != 2 or signals.shape[1] != 7 or len(signals) < 2
            or not np.isfinite(signals).all()):
        raise ValueError("expected_finite_frames_ge_2_by_7_signals")
    source_time = np.linspace(0.0, 1.0, len(signals))
    target_time = np.linspace(0.0, 1.0, N_FRAMES)
    return np.column_stack([
        np.interp(target_time, source_time, signals[:, channel]) for channel in range(7)
    ])


def resample_signals(poses):
    return resample_signal_array(extract_signals(poses))


def prepare_sequences(records):
    """Одна последовательность на каждую строку; никакой фильтрации сбоев."""
    if not records:
        raise ValueError("empty_sequence_records")
    rows, failures = [], []
    for index, record in enumerate(records):
        try:
            rows.append(resample_signals(record["poses"]))
        except (ValueError, KeyError, FloatingPointError) as exc:
            failures.append({"index": index, "physical_id": record.get("physical_id"),
                             "camera": record.get("camera"), "reason": str(exc)})
    if failures:
        error = ValueError(f"sequence_preparation_failed_without_filtering: {failures}")
        error.failures = failures
        raise error
    return np.stack(rows)


def _validated_sequences(x):
    result = np.asarray(x, dtype=np.float64)
    if (result.ndim != 3 or result.shape[1:] != (N_FRAMES, 7)
            or len(result) == 0 or not np.isfinite(result).all()):
        raise ValueError("expected_nonempty_finite_N_by_64_by_7")
    return result


def transform_order(x, ids, shuffled=False, seed=SHUFFLE_SEED):
    """Фиксированная отдельная перестановка кадров каждого физического повтора.

    Семь каналов одного кадра перемещаются вместе. ids — physical_id без
    camera, поэтому виды одного движения получают общую перестановку.
    Ключ не является
    предиктором; он лишь делает абляцию независимой от порядка строк/батчей.
    Перестановка выполняется после интерполяции, сохраняя те же 64 вектора.
    """
    values = _validated_sequences(x)
    if len(ids) != len(values):
        raise ValueError("one_stable_observation_id_required_per_sequence")
    result = values.copy()
    if not shuffled:
        return result
    for index, key in enumerate(ids):
        serialized = json.dumps(key, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{seed}|{serialized}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        permutation = np.random.default_rng(local_seed).permutation(N_FRAMES)
        result[index] = values[index, permutation]
    return result


def effective_training_weights(y, weights, balanced):
    y, weights = np.asarray(y), np.asarray(weights, dtype=np.float64)
    if (y.ndim != 1 or weights.shape != y.shape or len(y) == 0
            or set(np.unique(y)) != {0, 1, 2} or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError("expected_three_classes_and_positive_finite_train_weights")
    class_weights = np.ones(3, dtype=np.float64)
    if balanced:
        for label in range(3):
            class_weights[label] = weights.sum() / (3.0 * weights[y == label].sum())
    effective = weights * class_weights[y.astype(int)]
    return effective / effective.sum(), class_weights


def _network():
    return nn.Sequential(
        nn.Conv1d(7, 16, kernel_size=5, padding=2), nn.ReLU(),
        nn.Conv1d(16, 16, kernel_size=5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(64, 3),
    )


def _probabilities_from_logits(logits):
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class TemporalClassifier:
    """Три seed-модели с общей train-нормировкой и mean-proba ответом."""

    def __init__(self, spec):
        self.spec = dict(spec)

    def fit(self, x, y, weights):
        x = _validated_sequences(x)
        original_y = np.asarray(y)
        if original_y.shape != (len(x),) or not np.all(original_y == original_y.astype(int)):
            raise ValueError("one_integer_train_label_required_per_sequence")
        y = original_y.astype(int)
        observation_weights = np.asarray(weights, dtype=np.float64)
        effective, class_weights = effective_training_weights(
            y, observation_weights, bool(self.spec.get("balanced", False)))
        channel_weights = observation_weights / observation_weights.sum()
        self.mean_ = np.einsum("n,ntc->c", channel_weights, x) / N_FRAMES
        centered = x - self.mean_[None, None, :]
        variance = np.einsum("n,ntc->c", channel_weights, centered ** 2) / N_FRAMES
        self.scale_ = np.sqrt(np.maximum(variance, 0.0))
        self.scale_[self.scale_ <= 1e-12] = 1.0
        self.class_weights_ = class_weights
        self.classes_ = np.array([0, 1, 2])
        self.seeds_ = tuple(ENSEMBLE_SEEDS)
        self.training_config_ = {
            "epochs": TRAIN_EPOCHS, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "optimizer": "Adam", "batching": "full_batch",
            "seeds": list(self.seeds_), "balanced": bool(self.spec.get("balanced", False)),
            "device": "cpu", "threads": 1, "deterministic_algorithms": True,
            "validation_or_test_used": False,
        }
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        features = torch.from_numpy(np.ascontiguousarray(
            (centered / self.scale_[None, None, :]).transpose(0, 2, 1), dtype=np.float32))
        targets = torch.from_numpy(y.astype(np.int64))
        loss_weights = torch.from_numpy(effective.astype(np.float32))
        self.members_, self.training_losses_, self.training_log_ = [], [], []
        for seed in self.seeds_:
            member_started = time.perf_counter()
            torch.manual_seed(seed)
            model = _network().cpu()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            model.train()
            with torch.no_grad():
                initial = float((nn.functional.cross_entropy(model(features), targets, reduction="none")
                                 * loss_weights).sum().item())
            final = initial
            epoch_losses = []
            for _ in range(TRAIN_EPOCHS):
                optimizer.zero_grad(set_to_none=True)
                loss = (nn.functional.cross_entropy(model(features), targets, reduction="none")
                        * loss_weights).sum()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite_temporal_training_loss")
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                final = float((nn.functional.cross_entropy(model(features), targets, reduction="none")
                               * loss_weights).sum().item())
            if not np.isfinite(final):
                raise ValueError("nonfinite_temporal_final_loss")
            self.members_.append(model)
            self.training_losses_.append({"seed": seed, "initial": initial, "final": final})
            self.training_log_.append({
                "seed": seed, "initial": initial, "final": final,
                "epoch_losses": epoch_losses,
                "epoch_loss_timing": "train-mode weighted CE before optimizer step; final separately after last step",
                "seconds": time.perf_counter() - member_started,
            })
        self.n_parameters_per_member_ = sum(p.numel() for p in self.members_[0].parameters())
        return self

    def member_probabilities(self, x):
        x = _validated_sequences(x)
        if not hasattr(self, "members_"):
            raise ValueError("temporal_classifier_not_fitted")
        features = torch.from_numpy(np.ascontiguousarray(
            ((x - self.mean_[None, None, :]) / self.scale_[None, None, :]).transpose(0, 2, 1),
            dtype=np.float32))
        with torch.no_grad():
            return np.stack([_probabilities_from_logits(model(features).cpu().numpy())
                             for model in self.members_])

    def predict_proba(self, x):
        return self.member_probabilities(x).mean(axis=0)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


def fit_temporal(x, y, weights, spec):
    return TemporalClassifier(spec).fit(x, y, weights)
