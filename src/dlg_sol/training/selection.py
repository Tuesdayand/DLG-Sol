from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationSelection:
    trial: int
    validation_rmse: float
    best_iteration: int | None


@dataclass(frozen=True)
class NestedSelection:
    trial: int
    pooled_inner_rmse: float
    mean_fold_rmse: float
    mean_fold_mae: float
    final_tree_count: int
    inner_best_iterations: tuple[int, ...]


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"trial records are missing columns: {missing}")
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("trial records contain non-finite values")
    return values


def select_validation_trial(
    trial_records: pd.DataFrame,
    rmse_column: str = "val_rmse",
    best_iteration_column: str | None = "best_iteration",
) -> ValidationSelection:
    columns = ("trial", rmse_column) if best_iteration_column is None else ("trial", rmse_column, best_iteration_column)
    numeric = _numeric(trial_records, columns)
    trials = numeric["trial"].astype(int)
    if numeric["trial"].ne(trials).any() or trials.duplicated().any() or (trials < 0).any():
        raise ValueError("validation trial identifiers must be unique non-negative integers")
    ranked = numeric.assign(trial=trials).sort_values([rmse_column, "trial"], kind="stable")
    selected = ranked.iloc[0]
    best_iteration = None if best_iteration_column is None else int(selected[best_iteration_column])
    return ValidationSelection(int(selected["trial"]), float(selected[rmse_column]), best_iteration)


def select_nested_trial(trial_records: pd.DataFrame) -> NestedSelection:
    columns = ("trial", "inner_validation_fold", "n", "sse", "rmse", "mae", "best_iteration")
    numeric = _numeric(trial_records, columns)
    integer_columns = ("trial", "inner_validation_fold", "n", "best_iteration")
    converted = {column: numeric[column].astype(int) for column in integer_columns}
    if any(numeric[column].ne(converted[column]).any() for column in integer_columns):
        raise ValueError("nested trial identifiers, counts, and iterations must be integers")
    numeric = numeric.assign(**converted)
    if (numeric[["trial", "inner_validation_fold", "n", "best_iteration"]] < 0).any().any():
        raise ValueError("nested trial identifiers, counts, and iterations cannot be negative")
    if (numeric["n"] < 1).any():
        raise ValueError("nested validation folds must contain at least one row")
    if numeric.duplicated(["trial", "inner_validation_fold"]).any():
        raise ValueError("nested trial-fold records must be unique")
    fold_sets = numeric.groupby("trial")["inner_validation_fold"].apply(lambda values: tuple(sorted(values)))
    if len(set(fold_sets)) != 1:
        raise ValueError("every nested trial must cover the same validation folds")
    summary = numeric.groupby("trial", as_index=False).agg(
        inner_n=("n", "sum"),
        inner_sse=("sse", "sum"),
        mean_fold_rmse=("rmse", "mean"),
        mean_fold_mae=("mae", "mean"),
    )
    summary["pooled_inner_rmse"] = np.sqrt(summary["inner_sse"] / summary["inner_n"])
    ranked = summary.sort_values(["pooled_inner_rmse", "mean_fold_rmse", "trial"], kind="stable")
    selected = ranked.iloc[0]
    trial = int(selected["trial"])
    iterations = tuple(numeric.loc[numeric["trial"].eq(trial), "best_iteration"].astype(int))
    final_tree_count = int(np.median(np.asarray(iterations, dtype=int) + 1))
    return NestedSelection(
        trial=trial,
        pooled_inner_rmse=float(selected["pooled_inner_rmse"]),
        mean_fold_rmse=float(selected["mean_fold_rmse"]),
        mean_fold_mae=float(selected["mean_fold_mae"]),
        final_tree_count=final_tree_count,
        inner_best_iterations=iterations,
    )
