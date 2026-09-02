from __future__ import annotations

import numpy as np

from .label_firewall import LabelOperation, PartitionRole, enforce_label_access


def _vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def apply_fixed_blend(descriptor: object, neural: object, alpha: float) -> np.ndarray:
    descriptor_array = _vector(descriptor, "descriptor")
    neural_array = _vector(neural, "neural")
    if descriptor_array.shape != neural_array.shape:
        raise ValueError("component predictions must have identical shape")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be a finite scalar in [0, 1]")
    return (1.0 - alpha) * descriptor_array + alpha * neural_array


def fit_convex_weight(
    observed: object,
    descriptor: object,
    neural: object,
    *,
    partition: PartitionRole,
    default: float = 0.5,
    tolerance: float = 1e-12,
) -> float:
    enforce_label_access(partition, LabelOperation.COEFFICIENT_FITTING)
    observed_array = _vector(observed, "observed")
    descriptor_array = _vector(descriptor, "descriptor")
    neural_array = _vector(neural, "neural")
    if not (observed_array.shape == descriptor_array.shape == neural_array.shape):
        raise ValueError("observations and component predictions must have identical shape")
    direction = neural_array - descriptor_array
    denominator = float(direction @ direction)
    if denominator <= tolerance:
        default = float(default)
        if not np.isfinite(default) or not 0.0 <= default <= 1.0:
            raise ValueError("default must be a finite scalar in [0, 1]")
        return default
    raw = float(direction @ (observed_array - descriptor_array) / denominator)
    return float(np.clip(raw, 0.0, 1.0))
