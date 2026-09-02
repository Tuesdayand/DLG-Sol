from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FusionMember:
    member_index: int
    language_variant: str
    preprocessing: str
    regressor: str
    base_seed: int
    fixed_epochs: int


@dataclass(frozen=True)
class FusionConfig:
    model_identity: str
    members: tuple[FusionMember, ...]
    geometry_representation: str
    adapter_winsor_quantile: float
    adapter_winsorization: dict[str, bool]
    head_parameters: dict[str, float | int]
    gradient_clip_norm: float
    expected_members: int


EXPECTED_MEMBERS = (
    (0, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260901, 29),
    (1, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260902, 82),
    (2, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260903, 40),
    (3, "randomized_single_task", "raw_standardized_concat", "residual_mlp", 260904, 21),
)


def load_fusion_config(path: str | Path) -> FusionConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("model_identity") != "aug2_head4":
        raise ValueError("unsupported or noncanonical fusion configuration")
    members = tuple(
        FusionMember(
            int(record["member_index"]),
            str(record["language_variant"]),
            str(record["preprocessing"]),
            str(record["regressor"]),
            int(record["base_seed"]),
            int(record["fixed_epochs"]),
        )
        for record in payload["members"]
    )
    observed = tuple(
        (item.member_index, item.language_variant, item.preprocessing, item.regressor, item.base_seed, item.fixed_epochs)
        for item in members
    )
    if observed != EXPECTED_MEMBERS:
        raise ValueError("fusion-head mapping differs from the canonical aug2_head4 contract")
    preprocessing = payload["preprocessing"]
    prediction = payload["neural_prediction"]
    training = payload["training"]
    architecture = payload["head_architecture"]
    if preprocessing.get("fit_partition_only") is not True:
        raise ValueError("fusion preprocessing must be fit-partition-only")
    if preprocessing.get("method") != "modality_wise_standardization_then_raw_concatenation":
        raise ValueError("final fusion preprocessing must use raw standardized concatenation")
    if preprocessing.get("pca_used_by_final_model") is not False or preprocessing.get("l2_normalization_used_by_final_model") is not False:
        raise ValueError("PCA and L2 normalization are not part of the final model")
    winsorization = preprocessing.get("adapter_winsorization", {})
    if winsorization != {"aqsoldbc": True, "complat": True, "tdc": True, "jchem": False}:
        raise ValueError("dataset-specific winsorization mapping differs from the recorded contract")
    if preprocessing.get("failed_geometry_fallback_uses_labels") is not False:
        raise ValueError("geometry fallback must be label-free")
    if prediction.get("aggregation") != "arithmetic_mean" or prediction.get("input_dependent_weights") is not False:
        raise ValueError("neural-head aggregation differs from the released contract")
    if prediction.get("shared_input_representation") is not True or prediction.get("expected_members") != 4:
        raise ValueError("all four heads must receive the same concatenated representation")
    if architecture != {"hidden": 384, "depth": 2, "dropout": 0.45, "output_dimension": 1}:
        raise ValueError("residual-head architecture differs from the recorded contract")
    expected_training = {
        "loss": "mean_squared_error",
        "optimizer": "AdamW",
        "learning_rate": 0.0005,
        "weight_decay": 0.0005,
        "batch_size": 128,
        "gradient_clip_norm": 5.0,
        "epoch_policy": "fixed_per_head",
        "fold_seed_rule": "base_seed_plus_100000_times_fold_index",
        "schedule_provenance": "per-seed median best epoch from AqSolDBc five-fold development",
    }
    if training != expected_training:
        raise ValueError("fusion-head training contract differs from the recorded policy")
    head_parameters = {
        "hidden": int(architecture["hidden"]),
        "depth": int(architecture["depth"]),
        "dropout": float(architecture["dropout"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "batch_size": int(training["batch_size"]),
    }
    return FusionConfig(
        str(payload["model_identity"]),
        members,
        str(payload["shared_geometry_representation"]),
        float(preprocessing["adapter_winsor_quantile"]),
        dict(winsorization),
        head_parameters,
        float(training["gradient_clip_norm"]),
        int(prediction["expected_members"]),
    )
