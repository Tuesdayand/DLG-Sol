from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def build_xgb_regressor(
    parameters: Mapping[str, int | float],
    random_state: int,
    device: str = "cpu",
    n_jobs: int = 4,
    early_stopping_rounds: int | None = None,
):
    from xgboost import XGBRegressor

    values = dict(parameters)
    values.pop("trial", None)
    fixed = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "n_jobs": int(n_jobs),
        "random_state": int(random_state),
        "verbosity": 0,
    }
    if early_stopping_rounds is not None:
        fixed["eval_metric"] = "rmse"
        fixed["early_stopping_rounds"] = int(early_stopping_rounds)
    overlap = set(values).intersection(fixed)
    if overlap:
        raise ValueError(f"parameters override fixed XGBoost settings: {sorted(overlap)}")
    return XGBRegressor(**fixed, **values)


def fit_xgb_regressor(
    x_train: np.ndarray,
    y_train: Sequence[float],
    parameters: Mapping[str, int | float],
    random_state: int,
    device: str = "cpu",
    n_jobs: int = 4,
    x_validation: np.ndarray | None = None,
    y_validation: Sequence[float] | None = None,
    early_stopping_rounds: int | None = None,
):
    if (x_validation is None) != (y_validation is None):
        raise ValueError("validation features and labels must be supplied together")
    if early_stopping_rounds is not None and x_validation is None:
        raise ValueError("early stopping requires a validation partition")
    model = build_xgb_regressor(parameters, random_state, device, n_jobs, early_stopping_rounds)
    options = {}
    if x_validation is not None and y_validation is not None:
        options["eval_set"] = [(x_validation, np.asarray(y_validation, dtype=float))]
        options["verbose"] = False
    model.fit(np.asarray(x_train), np.asarray(y_train, dtype=float), **options)
    return model
