from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..descriptors.filtering import DescriptorFilterConfig, DescriptorSchema, fit_descriptor_schema, transform_descriptor_frame
from ..descriptors.xgboost_model import fit_xgb_regressor
from ..label_firewall import ensure_label_free
from ..metrics import regression_metrics
from .partitions import PartitionIds, validate_training_partitions
from .selection import select_validation_trial


@dataclass(frozen=True)
class PreparedDescriptorUnit:
    partitions: PartitionIds
    schema: DescriptorSchema
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_selection: np.ndarray
    y_selection: np.ndarray
    x_scored: np.ndarray


@dataclass(frozen=True)
class ValidationSearchResult:
    selected_trial: int
    selected_parameters: dict[str, int | float]
    selected_best_iteration: int | None
    trial_records: tuple[dict[str, int | float | None], ...]
    scored_predictions: np.ndarray
    selected_model: Any


def _indexed_descriptors(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in frame.columns:
        raise ValueError("descriptor table is missing the record key")
    if frame[key].isna().any():
        raise ValueError("descriptor record identifiers must be unique and non-missing")
    normalized = frame.copy()
    normalized[key] = normalized[key].astype(str)
    if normalized[key].duplicated().any():
        raise ValueError("descriptor record identifiers must be unique and non-missing")
    ensure_label_free(frame.columns)
    feature_columns = [column for column in frame.columns if column != key]
    if not feature_columns:
        raise ValueError("descriptor table contains no feature columns")
    return normalized.set_index(key, drop=True).loc[:, feature_columns]


def prepare_descriptor_unit(
    descriptors: pd.DataFrame,
    fit_rows: pd.DataFrame,
    selection_rows: pd.DataFrame,
    scored_rows: pd.DataFrame,
    config: DescriptorFilterConfig,
    key: str = "record_id",
    target: str = "logS",
) -> PreparedDescriptorUnit:
    partitions = validate_training_partitions(fit_rows, selection_rows, scored_rows, key, target)
    indexed = _indexed_descriptors(descriptors, key)
    required = set(partitions.fit) | set(partitions.selection) | set(partitions.scored)
    missing = required - set(indexed.index)
    if missing:
        raise ValueError(f"descriptor table is missing {len(missing)} required records")
    schema = fit_descriptor_schema(indexed.loc[list(partitions.fit)], config)
    x_fit = transform_descriptor_frame(indexed.loc[list(partitions.fit)], schema)
    x_selection = transform_descriptor_frame(indexed.loc[list(partitions.selection)], schema)
    x_scored = transform_descriptor_frame(indexed.loc[list(partitions.scored)], schema)
    y_fit = pd.to_numeric(fit_rows[target], errors="coerce").to_numpy(dtype=float)
    y_selection = pd.to_numeric(selection_rows[target], errors="coerce").to_numpy(dtype=float)
    return PreparedDescriptorUnit(partitions, schema, x_fit, y_fit, x_selection, y_selection, x_scored)


def _best_iteration(model: Any, early_stopping_rounds: int | None) -> int | None:
    if early_stopping_rounds is None:
        return None
    return int(model.best_iteration)


def evaluate_validation_candidates(
    unit: PreparedDescriptorUnit,
    candidates: Sequence[Mapping[str, int | float]],
    seed_for_trial: Callable[[int], int],
    device: str = "cpu",
    n_jobs: int = 4,
    early_stopping_rounds: int | None = None,
    tree_budget: int | None = None,
) -> ValidationSearchResult:
    if not candidates:
        raise ValueError("candidate collection cannot be empty")
    records: list[dict[str, int | float | None]] = []
    parameters_by_trial: dict[int, dict[str, int | float]] = {}
    observed_trials: set[int] = set()
    selected_model: Any | None = None
    selected_key: tuple[float, int] | None = None
    for position, source in enumerate(candidates):
        parameters = dict(source)
        trial_value = parameters.pop("trial", position)
        trial = int(trial_value)
        if trial != trial_value:
            raise ValueError("candidate trial identifiers must be integers")
        if trial < 0 or trial in observed_trials:
            raise ValueError("candidate trial identifiers must be unique non-negative integers")
        observed_trials.add(trial)
        if tree_budget is not None:
            if "n_estimators" in parameters:
                raise ValueError("tree budget conflicts with candidate n_estimators")
            parameters["n_estimators"] = int(tree_budget)
        model = fit_xgb_regressor(
            unit.x_fit,
            unit.y_fit,
            parameters,
            random_state=int(seed_for_trial(trial)),
            device=device,
            n_jobs=n_jobs,
            x_validation=unit.x_selection if early_stopping_rounds is not None else None,
            y_validation=unit.y_selection if early_stopping_rounds is not None else None,
            early_stopping_rounds=early_stopping_rounds,
        )
        validation_prediction = np.asarray(model.predict(unit.x_selection), dtype=float)
        metrics = regression_metrics(unit.y_selection, validation_prediction)
        best_iteration = _best_iteration(model, early_stopping_rounds)
        records.append(
            {
                "trial": trial,
                "n": len(unit.y_selection),
                "sse": float(np.sum((unit.y_selection - validation_prediction) ** 2)),
                "val_rmse": metrics.rmse,
                "val_mae": metrics.mae,
                "best_iteration": best_iteration,
            }
        )
        parameters_by_trial[trial] = parameters
        key = (metrics.rmse, trial)
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_model = model
    frame = pd.DataFrame(records)
    selected = select_validation_trial(
        frame,
        rmse_column="val_rmse",
        best_iteration_column="best_iteration" if early_stopping_rounds is not None else None,
    )
    if selected_model is None or selected_key != (selected.validation_rmse, selected.trial):
        raise RuntimeError("candidate ranking and retained model disagree")
    model = selected_model
    scored_predictions = np.asarray(model.predict(unit.x_scored), dtype=float)
    if not np.isfinite(scored_predictions).all():
        raise ValueError("selected model produced non-finite scored predictions")
    return ValidationSearchResult(
        selected_trial=selected.trial,
        selected_parameters=parameters_by_trial[selected.trial],
        selected_best_iteration=selected.best_iteration,
        trial_records=tuple(records),
        scored_predictions=scored_predictions,
        selected_model=model,
    )
