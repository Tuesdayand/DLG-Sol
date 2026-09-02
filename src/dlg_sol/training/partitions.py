from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..label_firewall import LabelOperation, PartitionRole, enforce_label_access, ensure_label_free


@dataclass(frozen=True)
class PartitionIds:
    fit: tuple[str, ...]
    selection: tuple[str, ...]
    scored: tuple[str, ...]


def _ids(frame: pd.DataFrame, key: str, name: str) -> tuple[str, ...]:
    if key not in frame.columns:
        raise ValueError(f"{name} rows are missing the record key")
    if frame[key].isna().any():
        raise ValueError(f"{name} record identifiers cannot be missing")
    normalized = frame[key].astype(str)
    if normalized.duplicated().any():
        raise ValueError(f"{name} record identifiers must be unique")
    values = tuple(normalized)
    if not values:
        raise ValueError(f"{name} partition cannot be empty")
    return values


def _labels(frame: pd.DataFrame, target: str, name: str) -> None:
    if target not in frame.columns:
        raise ValueError(f"{name} rows are missing the target column")
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} targets must be finite")


def validate_training_partitions(
    fit_rows: pd.DataFrame,
    selection_rows: pd.DataFrame,
    scored_rows: pd.DataFrame,
    key: str = "record_id",
    target: str = "logS",
) -> PartitionIds:
    enforce_label_access(PartitionRole.MODEL_DEVELOPMENT, LabelOperation.PREPROCESSING)
    enforce_label_access(PartitionRole.MODEL_DEVELOPMENT, LabelOperation.HYPERPARAMETER_SELECTION)
    enforce_label_access(PartitionRole.MODEL_DEVELOPMENT, LabelOperation.MODEL_FITTING)
    _labels(fit_rows, target, "fit")
    _labels(selection_rows, target, "selection")
    ensure_label_free(scored_rows.columns, {target, "label", "observed", "target", "y"})
    fit = _ids(fit_rows, key, "fit")
    selection = _ids(selection_rows, key, "selection")
    scored = _ids(scored_rows, key, "scored")
    sets = {"fit": set(fit), "selection": set(selection), "scored": set(scored)}
    for left, right in (("fit", "selection"), ("fit", "scored"), ("selection", "scored")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"{left} and {right} partitions overlap on {len(overlap)} record identifiers")
    return PartitionIds(fit=fit, selection=selection, scored=scored)
