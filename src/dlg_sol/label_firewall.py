from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


class PartitionRole(str, Enum):
    MODEL_DEVELOPMENT = "model_development"
    COEFFICIENT_DEVELOPMENT = "coefficient_development"
    SCORED = "scored"


class LabelOperation(str, Enum):
    PREPROCESSING = "preprocessing"
    HYPERPARAMETER_SELECTION = "hyperparameter_selection"
    MODEL_FITTING = "model_fitting"
    COEFFICIENT_FITTING = "coefficient_fitting"
    SCORING = "scoring"


ALLOWED_OPERATIONS = {
    PartitionRole.MODEL_DEVELOPMENT: frozenset(LabelOperation),
    PartitionRole.COEFFICIENT_DEVELOPMENT: frozenset(
        {LabelOperation.COEFFICIENT_FITTING, LabelOperation.SCORING}
    ),
    PartitionRole.SCORED: frozenset({LabelOperation.SCORING}),
}
DEFAULT_TARGET_COLUMNS = frozenset({"label", "logS", "observed", "target", "y"})


def enforce_label_access(partition: PartitionRole, operation: LabelOperation) -> None:
    if operation not in ALLOWED_OPERATIONS[partition]:
        raise PermissionError(f"{operation.value} cannot use labels from {partition.value} rows")


def ensure_label_free(
    columns: Iterable[str], target_columns: Iterable[str] = DEFAULT_TARGET_COLUMNS
) -> None:
    overlap = set(columns) & set(target_columns)
    if overlap:
        raise ValueError(f"prediction input contains label columns: {sorted(overlap)}")


def bind_labels_for_scoring(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    prediction_columns: Iterable[str],
    key: str = "record_id",
    target: str = "logS",
) -> pd.DataFrame:
    enforce_label_access(PartitionRole.SCORED, LabelOperation.SCORING)
    prediction_columns = tuple(prediction_columns)
    ensure_label_free(predictions.columns, set(DEFAULT_TARGET_COLUMNS) | {target})
    required_predictions = {key, *prediction_columns}
    required_labels = {key, target}
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("prediction table is missing required columns")
    if not required_labels.issubset(labels.columns):
        raise ValueError("label table is missing required columns")
    if predictions[key].isna().any() or labels[key].isna().any():
        raise ValueError("record identifiers cannot be missing")
    if predictions[key].duplicated().any() or labels[key].duplicated().any():
        raise ValueError("record identifiers must be unique")
    left = predictions[[key, *prediction_columns]].copy()
    left["_prediction_order"] = np.arange(len(left))
    joined = left.merge(
        labels[[key, target]],
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("prediction and label rows do not match exactly")
    joined = joined.sort_values("_prediction_order").drop(columns=["_prediction_order", "_merge"])
    numeric = joined[[target, *prediction_columns]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("labels and predictions must be finite")
    return joined[[key, target, *prediction_columns]].reset_index(drop=True)
