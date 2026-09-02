from __future__ import annotations

import numpy as np
import pandas as pd

from ..label_firewall import ensure_label_free


def validate_oof_predictions(
    predictions: pd.DataFrame,
    expected_rows: int,
    key: str = "record_id",
    prediction: str = "prediction",
) -> pd.DataFrame:
    ensure_label_free(predictions.columns)
    required = {key, prediction}
    if not required.issubset(predictions.columns):
        raise ValueError("OOF predictions are missing required columns")
    if len(predictions) != expected_rows:
        raise ValueError("OOF prediction row count does not match the evaluation contract")
    if predictions[key].isna().any() or predictions[key].duplicated().any():
        raise ValueError("OOF predictions require unique non-missing record identifiers")
    values = pd.to_numeric(predictions[prediction], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OOF predictions must be finite")
    return predictions[[key, prediction]].reset_index(drop=True).copy()


def aggregate_member_predictions(
    predictions: pd.DataFrame,
    expected_members: int,
    expected_rows: int,
    key: str = "record_id",
    member: str = "member",
    prediction: str = "prediction",
) -> pd.DataFrame:
    ensure_label_free(predictions.columns)
    required = {key, member, prediction}
    if not required.issubset(predictions.columns):
        raise ValueError("member predictions are missing required columns")
    if expected_members < 1 or expected_rows < 1:
        raise ValueError("expected member and row counts must be positive")
    if predictions[[key, member]].isna().any().any():
        raise ValueError("member prediction identifiers cannot be missing")
    if predictions.duplicated([key, member]).any():
        raise ValueError("member predictions contain duplicate record-member pairs")
    values = pd.to_numeric(predictions[prediction], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("member predictions must be finite")
    member_sets = predictions.groupby(key, sort=False)[member].apply(lambda items: tuple(sorted(items.astype(str))))
    if len(member_sets) != expected_rows or member_sets.map(len).ne(expected_members).any():
        raise ValueError("member predictions do not satisfy expected row or member counts")
    if len(set(member_sets)) != 1:
        raise ValueError("every scored row must contain the same member identifiers")
    order = predictions[[key]].drop_duplicates().reset_index(drop=True)
    averaged = predictions.assign(**{prediction: values}).groupby(key, sort=False, as_index=False)[prediction].mean()
    result = order.merge(averaged, on=key, how="left", validate="one_to_one", sort=False)
    return result[[key, prediction]]
