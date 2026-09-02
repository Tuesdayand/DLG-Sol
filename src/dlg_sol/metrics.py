from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    rmse: float
    mae: float
    bias_prediction_minus_observed: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    n: int
    delta_rmse_a_minus_b: float
    bootstrap_mean_delta_rmse: float
    ci95_low: float
    ci95_high: float
    inference: str
    bootstrap_replicates: int
    bootstrap_seed: int


def _vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def regression_metrics(observed: object, prediction: object) -> RegressionMetrics:
    observed_array = _vector(observed, "observed")
    prediction_array = _vector(prediction, "prediction")
    if observed_array.shape != prediction_array.shape:
        raise ValueError("observations and predictions must have identical shape")
    residual = prediction_array - observed_array
    return RegressionMetrics(
        n=len(observed_array),
        rmse=float(np.sqrt(np.mean(residual * residual))),
        mae=float(np.mean(np.abs(residual))),
        bias_prediction_minus_observed=float(np.mean(residual)),
    )


def paired_rmse_bootstrap(
    observed: object,
    prediction_a: object,
    prediction_b: object,
    *,
    seed: int,
    replicates: int = 10000,
    batch_size: int = 250,
) -> PairedBootstrapResult:
    observed_array = _vector(observed, "observed")
    prediction_a_array = _vector(prediction_a, "prediction_a")
    prediction_b_array = _vector(prediction_b, "prediction_b")
    if not (observed_array.shape == prediction_a_array.shape == prediction_b_array.shape):
        raise ValueError("observations and paired predictions must have identical shape")
    if replicates < 1 or batch_size < 1:
        raise ValueError("replicates and batch_size must be positive")
    squared_a = (prediction_a_array - observed_array) ** 2
    squared_b = (prediction_b_array - observed_array) ** 2
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    cursor = 0
    while cursor < replicates:
        width = min(batch_size, replicates - cursor)
        indices = rng.integers(0, len(observed_array), size=(width, len(observed_array)))
        values[cursor : cursor + width] = np.sqrt(np.mean(squared_a[indices], axis=1)) - np.sqrt(
            np.mean(squared_b[indices], axis=1)
        )
        cursor += width
    metric_a = regression_metrics(observed_array, prediction_a_array)
    metric_b = regression_metrics(observed_array, prediction_b_array)
    low, high = np.quantile(values, [0.025, 0.975])
    inference = "resolved" if low > 0.0 or high < 0.0 else "inconclusive"
    return PairedBootstrapResult(
        n=len(observed_array),
        delta_rmse_a_minus_b=metric_a.rmse - metric_b.rmse,
        bootstrap_mean_delta_rmse=float(np.mean(values)),
        ci95_low=float(low),
        ci95_high=float(high),
        inference=inference,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )
