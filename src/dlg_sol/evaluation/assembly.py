from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..blend import fit_convex_weight
from ..fusion.adapters import EvaluationAdapter, aggregate_scored_predictions
from ..label_firewall import PartitionRole, ensure_label_free


@dataclass(frozen=True)
class EvaluationAssembly:
    coefficients: pd.DataFrame
    predictions: pd.DataFrame


def _numeric_predictions(frame: pd.DataFrame, fold_column: str) -> pd.DataFrame:
    ensure_label_free(frame.columns)
    required = {"record_id", fold_column, "descriptor_prediction", "aug2_head4"}
    if set(frame.columns) != required:
        raise ValueError("component prediction columns differ from the evaluation contract")
    output = frame[["record_id", fold_column, "descriptor_prediction", "aug2_head4"]].copy()
    output["record_id"] = output["record_id"].astype(str)
    output[fold_column] = pd.to_numeric(output[fold_column], errors="raise").astype(int)
    if output[["record_id", fold_column]].isna().any().any() or output.duplicated(["record_id", fold_column]).any():
        raise ValueError("component predictions require unique record-fold pairs")
    values = output[["descriptor_prediction", "aug2_head4"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("component predictions must be finite")
    output[["descriptor_prediction", "aug2_head4"]] = values
    return output


def _coefficient_folds(adapter: EvaluationAdapter) -> tuple[int, ...]:
    if adapter.coefficient_policy in {"one_coefficient_from_complete_tdc_oof", "one_coefficient_from_complete_complat_oof"}:
        return (0,)
    return adapter.fold_indices


def _join_coefficient_rows(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    prediction_frame = _numeric_predictions(predictions, "deployment_fold")
    if set(labels.columns) != {"deployment_fold", "record_id", "logS"}:
        raise ValueError("coefficient-label columns differ from the evaluation contract")
    label_frame = labels[["deployment_fold", "record_id", "logS"]].copy()
    label_frame["deployment_fold"] = pd.to_numeric(label_frame["deployment_fold"], errors="raise").astype(int)
    label_frame["record_id"] = label_frame["record_id"].astype(str)
    label_frame["logS"] = pd.to_numeric(label_frame["logS"], errors="coerce")
    if label_frame[["deployment_fold", "record_id"]].isna().any().any() or label_frame.duplicated(["deployment_fold", "record_id"]).any() or not np.isfinite(label_frame["logS"]).all():
        raise ValueError("coefficient labels require unique finite record-fold rows")
    joined = prediction_frame.merge(label_frame, on=["deployment_fold", "record_id"], how="outer", validate="one_to_one", indicator=True, sort=False)
    if not joined["_merge"].eq("both").all():
        raise ValueError("coefficient predictions and labels do not match exactly")
    return joined.drop(columns="_merge")


def fit_evaluation_coefficients(adapter: EvaluationAdapter, scored_predictions: pd.DataFrame, coefficient_predictions: pd.DataFrame | None = None, coefficient_labels: pd.DataFrame | None = None) -> pd.DataFrame:
    scored = _numeric_predictions(scored_predictions, "fold_index")
    if adapter.coefficient_policy == "prespecified_0.5":
        if coefficient_predictions is not None or coefficient_labels is not None:
            raise ValueError("prespecified-coefficient evaluation cannot accept coefficient files")
        return pd.DataFrame({"deployment_fold": list(adapter.fold_indices), "alpha": [0.5] * len(adapter.fold_indices), "coefficient_rows": [0] * len(adapter.fold_indices)})
    if coefficient_predictions is None or coefficient_labels is None:
        raise ValueError("evaluation requires coefficient predictions and labels")
    joined = _join_coefficient_rows(coefficient_predictions, coefficient_labels)
    expected_folds = _coefficient_folds(adapter)
    if tuple(sorted(joined["deployment_fold"].unique())) != expected_folds:
        raise ValueError("coefficient sources do not cover every required deployment fold")
    rows = []
    for fold in expected_folds:
        group = joined[joined["deployment_fold"].eq(fold)]
        if group.empty:
            raise ValueError("coefficient source is empty")
        scored_ids = set(scored.loc[scored["fold_index"].eq(fold), "record_id"])
        if adapter.coefficient_policy in {"one_coefficient_from_complete_tdc_oof", "one_coefficient_from_complete_complat_oof"}:
            scored_ids = set(scored["record_id"])
        if scored_ids & set(group["record_id"]):
            raise ValueError("coefficient rows overlap the rows scored by the same deployed model")
        alpha = fit_convex_weight(group["logS"], group["descriptor_prediction"], group["aug2_head4"], partition=PartitionRole.COEFFICIENT_DEVELOPMENT)
        rows.append({"deployment_fold": int(fold), "alpha": alpha, "coefficient_rows": len(group)})
    return pd.DataFrame(rows)


def _aggregate_column(frame: pd.DataFrame, adapter: EvaluationAdapter, column: str) -> pd.DataFrame:
    renamed = frame[["record_id", "fold_index", column]].rename(columns={column: "aug2_head4"})
    aggregated = aggregate_scored_predictions(renamed, adapter)
    return aggregated.rename(columns={"aug2_head4": column})


def assemble_evaluation_predictions(adapter: EvaluationAdapter, scored_predictions: pd.DataFrame, coefficient_predictions: pd.DataFrame | None = None, coefficient_labels: pd.DataFrame | None = None) -> EvaluationAssembly:
    scored = _numeric_predictions(scored_predictions, "fold_index")
    if not set(scored["fold_index"]).issubset(set(adapter.fold_indices)):
        raise ValueError("scored component predictions contain undeclared folds")
    coefficients = fit_evaluation_coefficients(adapter, scored, coefficient_predictions, coefficient_labels)
    alpha_by_fold = dict(zip(coefficients["deployment_fold"].astype(int), coefficients["alpha"].astype(float)))
    global_policy = adapter.coefficient_policy in {"one_coefficient_from_complete_tdc_oof", "one_coefficient_from_complete_complat_oof"}
    if global_policy:
        alpha = np.full(len(scored), alpha_by_fold[0], dtype=float)
    else:
        alpha = scored["fold_index"].map(alpha_by_fold).to_numpy(dtype=float)
    if not np.isfinite(alpha).all():
        raise ValueError("scored rows cannot be mapped to fitted coefficients")
    scored["dlg_sol"] = scored["descriptor_prediction"].to_numpy(float) + alpha * (scored["aug2_head4"].to_numpy(float) - scored["descriptor_prediction"].to_numpy(float))
    descriptor = _aggregate_column(scored, adapter, "descriptor_prediction")
    neural = _aggregate_column(scored, adapter, "aug2_head4")
    blend = _aggregate_column(scored, adapter, "dlg_sol")
    predictions = descriptor.merge(neural, on="record_id", validate="one_to_one").merge(blend, on="record_id", validate="one_to_one")
    return EvaluationAssembly(coefficients.reset_index(drop=True), predictions[["record_id", "descriptor_prediction", "aug2_head4", "dlg_sol"]].reset_index(drop=True))
