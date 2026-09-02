from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MeanEmbeddingFallback:
    values: np.ndarray

    def transform(self, embeddings, failed_mask=None):
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.values):
            raise ValueError("embedding matrix has an incompatible shape")
        inferred = ~np.isfinite(matrix).all(axis=1)
        failed = inferred if failed_mask is None else np.asarray(failed_mask, dtype=bool)
        if failed.shape != (len(matrix),):
            raise ValueError("failed-row mask has an incompatible shape")
        if np.any(inferred & ~failed):
            raise ValueError("non-finite embeddings must be marked as failed")
        result = matrix.copy()
        result[failed] = self.values
        if not np.isfinite(result).all():
            raise ValueError("fallback did not produce finite embeddings")
        return result


@dataclass(frozen=True)
class FusionPreprocessor:
    method: str
    language_scaler: StandardScaler
    geometry_scaler: StandardScaler
    language_pca: PCA | None
    geometry_pca: PCA | None
    pca_dimension: int | None
    language_limits: tuple[np.ndarray, np.ndarray] | None
    geometry_limits: tuple[np.ndarray, np.ndarray] | None

    def transform(self, language_embeddings, geometry_embeddings):
        language = _matrix(language_embeddings, "language")
        geometry = _matrix(geometry_embeddings, "geometry")
        if len(language) != len(geometry):
            raise ValueError("language and geometry row counts differ")
        language = _clip(language, self.language_limits)
        geometry = _clip(geometry, self.geometry_limits)
        language = self.language_scaler.transform(language)
        geometry = self.geometry_scaler.transform(geometry)
        if self.method == "raw_standardized_concat":
            result = np.concatenate([language, geometry], axis=1)
        elif self.method == "separate_pca384_l2_concat":
            language = _pad(_safe_l2(self.language_pca.transform(language)), self.pca_dimension)
            geometry = _pad(_safe_l2(self.geometry_pca.transform(geometry)), self.pca_dimension)
            result = np.concatenate([language, geometry], axis=1) / np.sqrt(2.0)
        else:
            raise ValueError("unsupported fusion preprocessing method")
        result = np.asarray(result, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("fusion preprocessing produced non-finite values")
        return result


def _matrix(values, name):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} embeddings must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} embeddings must be finite before fusion preprocessing")
    return matrix


def _mask(values, rows):
    mask = np.asarray(values, dtype=bool)
    if mask.shape != (rows,) or not mask.any():
        raise ValueError("fit mask must select at least one row")
    return mask


def _limits(matrix, fit_mask, quantile):
    if quantile is None:
        return None
    value = float(quantile)
    if not 0.0 <= value < 0.5:
        raise ValueError("winsor quantile must be in [0, 0.5)")
    return np.quantile(matrix[fit_mask], value, axis=0), np.quantile(matrix[fit_mask], 1.0 - value, axis=0)


def _clip(matrix, limits):
    if limits is None:
        return matrix
    return np.clip(matrix, limits[0], limits[1]).astype(np.float32)


def _safe_l2(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / np.where(norms < 1e-12, 1.0, norms), dtype=np.float32)


def _pad(matrix, dimension):
    if matrix.shape[1] >= dimension:
        return np.asarray(matrix[:, :dimension], dtype=np.float32)
    return np.pad(matrix, ((0, 0), (0, dimension - matrix.shape[1]))).astype(np.float32)


def _fit_pca(matrix, fit_mask, scaler, dimension, seed):
    fit_values = scaler.transform(matrix[fit_mask]).astype(np.float32)
    components = min(int(dimension), fit_values.shape[1], int(fit_mask.sum()) - 1)
    if components < 1:
        raise ValueError("PCA requires at least two fit rows")
    solver = "randomized" if components < min(fit_values.shape) else "full"
    return PCA(n_components=components, svd_solver=solver, random_state=int(seed)).fit(fit_values)


def fit_fusion_preprocessor(language_embeddings, geometry_embeddings, fit_mask, method, seed, pca_dimension=384, winsor_quantile=None):
    language = _matrix(language_embeddings, "language")
    geometry = _matrix(geometry_embeddings, "geometry")
    if len(language) != len(geometry):
        raise ValueError("language and geometry row counts differ")
    fit = _mask(fit_mask, len(language))
    language_limits = _limits(language, fit, winsor_quantile)
    geometry_limits = _limits(geometry, fit, winsor_quantile)
    clipped_language = _clip(language, language_limits)
    clipped_geometry = _clip(geometry, geometry_limits)
    language_scaler = StandardScaler().fit(clipped_language[fit])
    geometry_scaler = StandardScaler().fit(clipped_geometry[fit])
    language_pca = None
    geometry_pca = None
    dimension = None
    if method == "separate_pca384_l2_concat":
        dimension = int(pca_dimension)
        if dimension != 384:
            raise ValueError("released PCA fusion member requires dimension 384")
        language_pca = _fit_pca(clipped_language, fit, language_scaler, dimension, seed)
        geometry_pca = _fit_pca(clipped_geometry, fit, geometry_scaler, dimension, int(seed) + 101)
    elif method != "raw_standardized_concat":
        raise ValueError("unsupported fusion preprocessing method")
    return FusionPreprocessor(method, language_scaler, geometry_scaler, language_pca, geometry_pca, dimension, language_limits, geometry_limits)


def fit_training_mean_fallback(embeddings, fit_mask):
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("embedding matrix must be non-empty and two-dimensional")
    fit = np.asarray(fit_mask, dtype=bool)
    if fit.shape != (len(matrix),) or not fit.any():
        raise ValueError("fit mask has an incompatible shape or is empty")
    successful = fit & np.isfinite(matrix).all(axis=1)
    if not successful.any():
        raise ValueError("no successful fit embeddings are available for fallback")
    return MeanEmbeddingFallback(np.asarray(matrix[successful].mean(axis=0), dtype=np.float32))
