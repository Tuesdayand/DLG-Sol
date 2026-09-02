from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


VALID_CORRELATION_COMPARISONS = {"gt", "ge"}


@dataclass(frozen=True)
class DescriptorFilterConfig:
    variance_threshold: float = 1e-12
    variance_ddof: int = 1
    correlation_threshold: float | None = 0.98
    correlation_comparison: str | None = "ge"


@dataclass(frozen=True)
class DescriptorSchema:
    input_columns: tuple[str, ...]
    retained_columns: tuple[str, ...]
    dropped_missing: tuple[str, ...]
    dropped_constant: tuple[str, ...]
    dropped_low_variance: tuple[str, ...]
    dropped_high_correlation: tuple[str, ...]
    config: DescriptorFilterConfig


def _validate_config(config: DescriptorFilterConfig) -> None:
    if config.variance_threshold < 0:
        raise ValueError("variance threshold cannot be negative")
    if config.variance_ddof not in {0, 1}:
        raise ValueError("variance ddof must be zero or one")
    if config.correlation_threshold is None:
        if config.correlation_comparison is not None:
            raise ValueError("correlation comparison must be null when correlation filtering is disabled")
        return
    if not 0 <= config.correlation_threshold <= 1:
        raise ValueError("correlation threshold must be between zero and one")
    if config.correlation_comparison not in VALID_CORRELATION_COMPARISONS:
        raise ValueError("unsupported correlation comparison")


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.has_duplicates:
        raise ValueError("descriptor columns must be unique")
    if len(frame.columns) == 0:
        raise ValueError("descriptor frame must contain at least one column")
    return frame.apply(pd.to_numeric, errors="coerce")


def fit_descriptor_schema(frame: pd.DataFrame, config: DescriptorFilterConfig) -> DescriptorSchema:
    _validate_config(config)
    numeric = _numeric_frame(frame)
    dropped_missing = tuple(numeric.columns[numeric.isna().any(axis=0)])
    current = numeric.drop(columns=list(dropped_missing))
    dropped_constant = tuple(current.columns[current.nunique(dropna=False) <= 1])
    current = current.drop(columns=list(dropped_constant))
    variance = current.var(axis=0, ddof=config.variance_ddof)
    dropped_low_variance = tuple(variance.index[variance <= config.variance_threshold])
    current = current.drop(columns=list(dropped_low_variance))
    dropped_high_correlation: tuple[str, ...] = ()
    if config.correlation_threshold is not None and len(current.columns) > 1:
        correlation = current.corr(method="pearson").abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        if config.correlation_comparison == "gt":
            selected = [column for column in upper.columns if (upper[column] > config.correlation_threshold).any()]
        else:
            selected = [column for column in upper.columns if (upper[column] >= config.correlation_threshold).any()]
        dropped_high_correlation = tuple(selected)
        current = current.drop(columns=selected)
    retained = tuple(current.columns)
    if not retained:
        raise ValueError("descriptor filtering removed every column")
    return DescriptorSchema(
        input_columns=tuple(numeric.columns),
        retained_columns=retained,
        dropped_missing=dropped_missing,
        dropped_constant=dropped_constant,
        dropped_low_variance=dropped_low_variance,
        dropped_high_correlation=dropped_high_correlation,
        config=config,
    )


def transform_descriptor_frame(frame: pd.DataFrame, schema: DescriptorSchema, dtype: object = np.float32) -> np.ndarray:
    missing = [column for column in schema.retained_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"descriptor frame is missing {len(missing)} retained columns")
    numeric = frame.loc[:, list(schema.retained_columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=dtype)
    if not np.isfinite(values).all():
        raise ValueError("retained descriptor matrix contains non-finite values")
    return values
