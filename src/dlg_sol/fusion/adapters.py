from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..label_firewall import ensure_label_free
from .config import FusionConfig
from .preprocessing import FusionPreprocessor, fit_fusion_preprocessor


@dataclass(frozen=True)
class EvaluationAdapter:
    evaluation_id: str
    dataset_id: str
    source_design: str
    partition_column: str
    fit_values: tuple[str, ...]
    coefficient_values: tuple[str, ...]
    scored_values: tuple[str, ...]
    reserved_values: tuple[str, ...]
    fold_indices: tuple[int, ...]
    winsorization: bool
    winsor_quantile: float | None
    scored_prediction_aggregation: str
    expected_scored_rows: int
    coefficient_policy: str
    historical_neural_column: str


@dataclass(frozen=True)
class AdapterRegistry:
    model_identity: str
    fold_seed_multiplier: int
    record_key: str
    fold_column: str
    member_columns: tuple[str, ...]
    canonical_neural_column: str
    adapters: dict[str, EvaluationAdapter]


@dataclass(frozen=True)
class AdapterMasks:
    fit: np.ndarray
    coefficient: np.ndarray
    scored: np.ndarray
    reserved: np.ndarray


EXPECTED_ADAPTERS = {
    "R01": ("aqsoldbc", "split", ("Train",), (), ("External",), ("Val",), (0, 1, 2, 3, 4), True, 0.001, "one_held_out_prediction", 8047, "prespecified_0.5", "aug2_head4"),
    "R02": ("complat", "split", ("Train",), (), ("External",), (), (0, 1, 2, 3, 4), True, 0.001, "one_held_out_prediction", 17937, "cross_fitted_from_training_side_oof", "aug2_head4"),
    "R03": ("tdc", "partition", ("fit",), (), ("oof_holdout",), ("selection", "official_test"), (0, 1, 2, 3, 4), True, 0.001, "one_held_out_prediction", 7985, "cross_fitted_from_training_side_oof", "aug2_head4"),
    "R04": ("tdc", "partition", ("fit",), (), ("official_test",), ("selection", "oof_holdout"), (0, 1, 2, 3, 4), True, 0.001, "mean_across_split_models", 1997, "one_coefficient_from_complete_tdc_oof", "aug2_head4"),
    "R05": ("jchem", "split", ("Train",), ("Val",), ("External",), (), (1, 2, 3, 4, 5), False, None, "mean_across_split_models", 980, "validation_only_within_each_reported_split", "aug2_head4"),
    "R09": ("complat", "split", ("Train",), (), ("External",), (), (0,), True, 0.001, "single_full_refit", 1282, "one_coefficient_from_complete_complat_oof", "polarh_aug2_head4"),
}


def _tuple(values) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def load_adapter_registry(path: str | Path) -> AdapterRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("model_identity") != "aug2_head4":
        raise ValueError("unsupported or noncanonical adapter configuration")
    output = payload.get("common_output_schema", {})
    if output != {"record_key": "record_id", "fold_column": "fold_index", "member_columns": ["head_0", "head_1", "head_2", "head_3"], "canonical_neural_column": "aug2_head4"}:
        raise ValueError("adapter output schema differs from the recorded contract")
    records = {}
    for item in payload.get("evaluation_adapters", []):
        adapter = EvaluationAdapter(
            str(item["evaluation_id"]), str(item["dataset_id"]), str(item["source_design"]), str(item["partition_column"]),
            _tuple(item["fit_values"]), _tuple(item["coefficient_values"]), _tuple(item["scored_values"]), _tuple(item["reserved_values"]),
            tuple(int(value) for value in item["fold_indices"]), bool(item["winsorization"]),
            None if item["winsor_quantile"] is None else float(item["winsor_quantile"]),
            str(item["scored_prediction_aggregation"]), int(item["expected_scored_rows"]), str(item["coefficient_policy"]), str(item["historical_neural_column"]),
        )
        if adapter.evaluation_id in records:
            raise ValueError("adapter evaluation identifiers must be unique")
        records[adapter.evaluation_id] = adapter
    if set(records) != set(EXPECTED_ADAPTERS):
        raise ValueError("adapter registry does not contain the exact primary evaluation set")
    for evaluation_id, expected in EXPECTED_ADAPTERS.items():
        item = records[evaluation_id]
        observed = (item.dataset_id, item.partition_column, item.fit_values, item.coefficient_values, item.scored_values, item.reserved_values, item.fold_indices, item.winsorization, item.winsor_quantile, item.scored_prediction_aggregation, item.expected_scored_rows, item.coefficient_policy, item.historical_neural_column)
        if observed != expected:
            raise ValueError(f"adapter contract mismatch: {evaluation_id}")
    multiplier = int(payload.get("fold_seed_multiplier", 0))
    if multiplier != 100000:
        raise ValueError("adapter fold-seed multiplier differs from the recorded contract")
    return AdapterRegistry("aug2_head4", multiplier, "record_id", "fold_index", ("head_0", "head_1", "head_2", "head_3"), "aug2_head4", records)


def build_adapter_masks(metadata: pd.DataFrame, adapter: EvaluationAdapter, key: str = "record_id") -> AdapterMasks:
    ensure_label_free(metadata.columns)
    if key not in metadata or adapter.partition_column not in metadata:
        raise ValueError("adapter metadata is missing its record or partition column")
    if metadata[key].isna().any() or metadata[key].astype(str).duplicated().any():
        raise ValueError("adapter metadata requires unique non-missing record identifiers")
    partition = metadata[adapter.partition_column].astype(str)
    groups = tuple(adapter.fit_values + adapter.coefficient_values + adapter.scored_values + adapter.reserved_values)
    if len(groups) != len(set(groups)):
        raise ValueError("adapter partition roles overlap")
    unknown = sorted(set(partition) - set(groups))
    if unknown:
        raise ValueError(f"adapter metadata contains undeclared partition values: {unknown}")
    masks = AdapterMasks(partition.isin(adapter.fit_values).to_numpy(), partition.isin(adapter.coefficient_values).to_numpy(), partition.isin(adapter.scored_values).to_numpy(), partition.isin(adapter.reserved_values).to_numpy())
    stacked = np.vstack([masks.fit, masks.coefficient, masks.scored, masks.reserved]).astype(np.int8)
    if not masks.fit.any() or not masks.scored.any() or not np.all(stacked.sum(axis=0) == 1):
        raise ValueError("adapter roles must classify every row exactly once with non-empty fit and scored sets")
    if adapter.coefficient_values and not masks.coefficient.any():
        raise ValueError("adapter coefficient partition is empty")
    return masks


def adapter_head_schedule(adapter: EvaluationAdapter, fusion: FusionConfig, fold_index: int, fold_seed_multiplier: int = 100000) -> tuple[tuple[int, int, int], ...]:
    fold = int(fold_index)
    if fold not in adapter.fold_indices:
        raise ValueError("fold index is not declared by the adapter")
    if int(fold_seed_multiplier) != 100000:
        raise ValueError("fold-seed multiplier differs from the recorded contract")
    return tuple((member.member_index, member.base_seed + fold * int(fold_seed_multiplier), member.fixed_epochs) for member in fusion.members)


def fit_adapter_preprocessor(language_embeddings, geometry_embeddings, masks: AdapterMasks, adapter: EvaluationAdapter, seed: int) -> FusionPreprocessor:
    quantile = adapter.winsor_quantile if adapter.winsorization else None
    return fit_fusion_preprocessor(language_embeddings, geometry_embeddings, masks.fit, "raw_standardized_concat", int(seed), winsor_quantile=quantile)


def aggregate_scored_predictions(predictions: pd.DataFrame, adapter: EvaluationAdapter, key: str = "record_id", fold: str = "fold_index", prediction: str = "aug2_head4") -> pd.DataFrame:
    ensure_label_free(predictions.columns)
    required = {key, fold, prediction}
    if not required.issubset(predictions.columns):
        raise ValueError("scored predictions are missing adapter output columns")
    if predictions[[key, fold]].isna().any().any() or predictions.duplicated([key, fold]).any():
        raise ValueError("scored predictions require unique record-fold pairs")
    frame = predictions[[key, fold, prediction]].copy()
    frame[key] = frame[key].astype(str)
    frame[fold] = pd.to_numeric(frame[fold], errors="raise").astype(int)
    values = pd.to_numeric(frame[prediction], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not set(frame[fold]).issubset(set(adapter.fold_indices)):
        raise ValueError("scored prediction values or fold identifiers violate the adapter contract")
    frame[prediction] = values
    mode = adapter.scored_prediction_aggregation
    if mode in {"one_held_out_prediction", "single_full_refit"}:
        if len(frame) != adapter.expected_scored_rows or frame[key].duplicated().any():
            raise ValueError("single-prediction adapter row coverage is incomplete")
        return frame[[key, prediction]].reset_index(drop=True)
    if mode != "mean_across_split_models":
        raise ValueError("unsupported scored-prediction aggregation")
    fold_sets = frame.groupby(key, sort=False)[fold].apply(lambda items: tuple(sorted(items.astype(int))))
    if len(fold_sets) != adapter.expected_scored_rows or any(item != adapter.fold_indices for item in fold_sets):
        raise ValueError("split-model predictions do not cover every required fold")
    order = frame[[key]].drop_duplicates().reset_index(drop=True)
    means = frame.sort_values([key, fold]).groupby(key, sort=False)[prediction].mean().reset_index()
    return order.merge(means, on=key, how="left", validate="one_to_one", sort=False)[[key, prediction]]
