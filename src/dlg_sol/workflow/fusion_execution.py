from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ..fusion.adapters import adapter_head_schedule, load_adapter_registry
from ..fusion.config import load_fusion_config
from ..fusion.ensemble import mean_member_predictions
from ..fusion.modeling import build_fusion_model
from ..fusion.preprocessing import fit_fusion_preprocessor
from ..fusion.training import fit_fixed_epoch_fusion_head, predict_fusion
from .branch_io import MaterializedBranchInput, sha256_file, write_prediction_output


def _load_embeddings(path, evaluation_id, branch_id, fold_index):
    root = Path(path)
    manifest_path = root / "run_manifest.json"
    embedding_path = root / "embeddings.npz"
    if not manifest_path.is_file() or not embedding_path.is_file():
        raise ValueError("fusion dependency run is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_id") != evaluation_id or manifest.get("branch_id") != branch_id or int(manifest.get("fold_index")) != int(fold_index):
        raise ValueError("fusion dependency run identity mismatch")
    if manifest.get("scored_labels_accessed") is not False:
        raise ValueError("fusion dependency violates the scored-label firewall")
    expected = manifest.get("artifacts", {}).get("embeddings.npz", {})
    if expected.get("sha256") != sha256_file(embedding_path) or int(expected.get("bytes", -1)) != embedding_path.stat().st_size:
        raise ValueError("fusion dependency embedding identity mismatch")
    package = np.load(embedding_path, allow_pickle=False)
    identifiers = pd.Series(package["record_ids"].astype(str), dtype="string")
    matrix = np.asarray(package["embeddings"], dtype=np.float32)
    if identifiers.isna().any() or identifiers.duplicated().any() or matrix.ndim != 2 or len(matrix) != len(identifiers) or not np.isfinite(matrix).all():
        raise ValueError("fusion dependency embedding package is invalid")
    return root, manifest_path, identifiers.astype(str), matrix


def _align(identifiers, matrix, required, name):
    required_ids = [str(value) for value in required]
    index = pd.Series(np.arange(len(identifiers)), index=identifiers)
    missing = set(required_ids) - set(index.index)
    if missing:
        raise ValueError(f"{name} embeddings omit materialized fusion rows")
    return matrix[index.loc[required_ids].to_numpy()]


def _labels(frame):
    values = pd.to_numeric(frame["logS"], errors="coerce").to_numpy(np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("fusion fit labels must be finite and non-empty")
    return values


def _load_reuse(path, branch_input, fusion, input_dimension):
    root = Path(path)
    manifest_path = root / "run_manifest.json"
    preprocessor_path = root / "preprocessor.joblib"
    if not manifest_path.is_file() or not preprocessor_path.is_file():
        raise ValueError("fusion companion run is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_id") != "R03" or manifest.get("branch_id") != "fusion_head" or int(manifest.get("fold_index")) != branch_input.fold_index:
        raise ValueError("fusion companion run does not match the declared dependency")
    models = []
    for member in fusion.members:
        path = root / f"head_{member.member_index}.pt"
        if not path.is_file():
            raise ValueError("fusion companion head set is incomplete")
        model = build_fusion_model("residual_mlp", input_dimension, fusion.head_parameters)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        models.append(model)
    return manifest_path, joblib.load(preprocessor_path), models


def run_fusion_unit(branch_input: MaterializedBranchInput, output: str | Path, language_run: str | Path, geometry_run: str | Path, fusion_config_path: str | Path, adapter_config_path: str | Path, device: str, reuse_run: str | Path | None = None) -> dict:
    if branch_input.branch_id != "fusion_head":
        raise ValueError("fusion execution requires a fusion-head materialization")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("fusion output directory must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    fusion = load_fusion_config(fusion_config_path)
    adapters = load_adapter_registry(adapter_config_path)
    adapter = adapters.adapters[branch_input.evaluation_id]
    _, language_manifest, language_ids, language_matrix = _load_embeddings(language_run, branch_input.evaluation_id, "language", branch_input.fold_index)
    _, geometry_manifest, geometry_ids, geometry_matrix = _load_embeddings(geometry_run, branch_input.evaluation_id, "geometry", branch_input.fold_index)
    fit_stage = "final_refit" if branch_input.recipe.execution_type == "final_refit" else "parameter_fit"
    fit = branch_input.stages[fit_stage]
    scored = branch_input.stages["scored_prediction"]
    language_fit = _align(language_ids, language_matrix, fit["record_id"], "language")
    language_scored = _align(language_ids, language_matrix, scored["record_id"], "language")
    geometry_fit = _align(geometry_ids, geometry_matrix, fit["record_id"], "geometry")
    geometry_scored = _align(geometry_ids, geometry_matrix, scored["record_id"], "geometry")
    dependency = None
    if branch_input.recipe.execution_type == "reuse_companion_model":
        if reuse_run is None:
            raise ValueError("R04 fusion execution requires the matching R03 fusion run")
        input_dimension = language_fit.shape[1] + geometry_fit.shape[1]
        dependency_manifest, preprocessor, models = _load_reuse(reuse_run, branch_input, fusion, input_dimension)
        dependency = {"evaluation_id": "R03", "run_manifest_sha256": hashlib.sha256(dependency_manifest.read_bytes()).hexdigest()}
    else:
        if reuse_run is not None:
            raise ValueError("new fusion training does not accept a companion run")
        quantile = adapter.winsor_quantile if adapter.winsorization else None
        preprocessor = fit_fusion_preprocessor(language_fit, geometry_fit, np.ones(len(fit), dtype=bool), "raw_standardized_concat", 260901, winsor_quantile=quantile)
        fit_features = preprocessor.transform(language_fit, geometry_fit)
        schedule = adapter_head_schedule(adapter, fusion, branch_input.fold_index, adapters.fold_seed_multiplier)
        models = [fit_fixed_epoch_fusion_head(fit_features, _labels(fit), fusion.head_parameters, epochs, fusion.gradient_clip_norm, seed, device).model for _, seed, epochs in schedule]
    scored_features = preprocessor.transform(language_scored, geometry_scored)
    members = np.column_stack([predict_fusion(model, scored_features, device) for model in models]).astype(np.float32)
    predictions = mean_member_predictions(members, fusion.expected_members)
    preprocessor_path = root / "preprocessor.joblib"
    member_path = root / "member_predictions.csv"
    joblib.dump(preprocessor, preprocessor_path)
    pd.DataFrame({"record_id": scored["record_id"].astype(str), **{f"head_{index}": members[:, index] for index in range(fusion.expected_members)}}).to_csv(member_path, index=False, lineterminator="\n")
    model_paths = []
    for member, model in zip(fusion.members, models):
        path = root / f"head_{member.member_index}.pt"
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)
        model_paths.append(path)
    artifacts = {path.name: path for path in (preprocessor_path, member_path, *model_paths)}
    details = {
        "model_identity": fusion.model_identity,
        "members": fusion.expected_members,
        "aggregation": "arithmetic_mean",
        "winsorization": adapter.winsorization,
        "language_run_manifest_sha256": hashlib.sha256(language_manifest.read_bytes()).hexdigest(),
        "geometry_run_manifest_sha256": hashlib.sha256(geometry_manifest.read_bytes()).hexdigest(),
        "device": str(device),
    }
    if dependency is not None:
        details["dependency"] = dependency
    return write_prediction_output(root, branch_input, predictions, artifacts, details)
