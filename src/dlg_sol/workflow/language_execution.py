from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..language import load_chemberta_variants
from ..language.modeling import build_chemberta_model
from ..language.training import fit_chemberta_fixed_epochs, fit_chemberta_variant, predict_chemberta
from .branch_io import MaterializedBranchInput, write_prediction_output


def _all_molecules(branch_input: MaterializedBranchInput) -> pd.DataFrame:
    frames = [frame.loc[:, ["record_id", "smiles"]] for frame in branch_input.stages.values() if len(frame)]
    frame = pd.concat(frames, ignore_index=True)
    if (frame.groupby("record_id")["smiles"].nunique() > 1).any():
        raise ValueError("materialized record identifiers map to conflicting SMILES")
    return frame.drop_duplicates("record_id", keep="first").reset_index(drop=True)


def _labels(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["logS"], errors="coerce").to_numpy(np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("language fit labels must be finite and non-empty")
    return values


def _load_dependency(path, evaluation_id, fold_index):
    root = Path(path)
    manifest_path = root / "run_manifest.json"
    model_path = root / "model.pt"
    selection_path = root / "selection.json"
    if not manifest_path.is_file() or not model_path.is_file() or not selection_path.is_file():
        raise ValueError("language dependency run is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_id") != evaluation_id or manifest.get("branch_id") != "language" or int(manifest.get("fold_index")) != int(fold_index):
        raise ValueError("language dependency run identity mismatch")
    if manifest.get("scored_labels_accessed") is not False:
        raise ValueError("language dependency violates the scored-label firewall")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return root, manifest_path, model_path, selection


def _load_profile(path, evaluation_id):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = payload.get("execution_profiles", {})
    if set(profiles) != {"R01", "R02", "R03", "R04", "R05", "R09"}:
        raise ValueError("ChemBERTa execution profiles are incomplete")
    profile = profiles[str(evaluation_id)]
    if not isinstance(profile.get("mode"), str):
        raise ValueError("ChemBERTa execution profile is malformed")
    return profile


def run_language_unit(branch_input: MaterializedBranchInput, output: str | Path, variants_path: str | Path, device: str, pretrained_model: str | Path | None = None, reuse_run: str | Path | None = None, settings_run: str | Path | None = None, seed_offset: int = 0) -> dict:
    if branch_input.branch_id != "language":
        raise ValueError("language execution requires a language materialization")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("language output directory must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    config = load_chemberta_variants(variants_path)["randomized_single_task"]
    profile = _load_profile(variants_path, branch_input.evaluation_id)
    seed = None
    if "seed_base" in profile:
        seed = int(profile["seed_base"]) + int(profile["fold_seed_multiplier"]) * int(branch_input.fold_index) + int(seed_offset)
        config = replace(config, seed=seed)
    if pretrained_model is not None:
        config = replace(config, model_identifier=str(pretrained_model), model_revision=None)
    selection = {}
    dependency = None
    if branch_input.recipe.execution_type == "reuse_companion_model":
        if reuse_run is None or settings_run is not None:
            raise ValueError("language companion reuse requires exactly one matching R03 run")
        source, dependency_manifest, model_path, selection = _load_dependency(reuse_run, "R03", branch_input.fold_index)
        model = build_chemberta_model(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
        dependency = {"evaluation_id": "R03", "run_manifest_sha256": hashlib.sha256(dependency_manifest.read_bytes()).hexdigest()}
    elif branch_input.evaluation_id == "R09":
        if settings_run is None or reuse_run is not None:
            raise ValueError("R09 language refit requires an R02 selected-settings run")
        _, dependency_manifest, _, selected = _load_dependency(settings_run, "R02", branch_input.fold_index)
        best_epoch = int(selected["best_epoch"])
        if best_epoch != int(profile["selected_epoch"]):
            raise ValueError("R09 dependency differs from the recorded R02 ChemBERTa epoch")
        fit = branch_input.stages["final_refit"]
        model = fit_chemberta_fixed_epochs(config, fit["smiles"], _labels(fit), best_epoch, device)
        selection = {"best_epoch": best_epoch, "source_evaluation_id": "R02"}
        dependency = {"evaluation_id": "R02", "run_manifest_sha256": hashlib.sha256(dependency_manifest.read_bytes()).hexdigest()}
    else:
        if reuse_run is not None or settings_run is not None:
            raise ValueError("new language training does not accept dependency runs")
        if branch_input.evaluation_id == "R02":
            best_epoch = int(profile["selected_epoch"])
            selection = {"best_epoch": best_epoch, "selection_mode": "frozen_global_preselection"}
            refit = branch_input.stages["final_refit"]
            model = fit_chemberta_fixed_epochs(config, refit["smiles"], _labels(refit), best_epoch, device)
        else:
            fit = branch_input.stages["parameter_fit"]
            score = branch_input.stages["selection_score"]
            selected = fit_chemberta_variant(config, fit["smiles"], _labels(fit), score["smiles"], _labels(score), device)
            selection = {"best_epoch": selected.best_epoch, "best_selection_rmse": selected.best_selection_rmse, "history": list(selected.history)}
            model = selected.model
    molecules = _all_molecules(branch_input)
    inferred = predict_chemberta(model, config, molecules["smiles"], device)
    index = pd.Series(np.arange(len(molecules)), index=molecules["record_id"].astype(str))
    scored_ids = branch_input.stages["scored_prediction"]["record_id"].astype(str).tolist()
    positions = index.loc[scored_ids].to_numpy()
    predictions = inferred.predictions[positions]
    model_path = root / "model.pt"
    embedding_path = root / "embeddings.npz"
    selection_path = root / "selection.json"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, model_path)
    np.savez_compressed(embedding_path, record_ids=molecules["record_id"].astype(str).to_numpy(dtype=str), embeddings=inferred.embeddings)
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {path.name: path for path in (model_path, embedding_path, selection_path)}
    details = {"variant": "randomized_single_task", "execution_profile": profile["mode"], "best_epoch": int(selection["best_epoch"]), "device": str(device), "seed": seed, "seed_offset": int(seed_offset), "embedding_dimension": int(config.embedding_dimension)}
    if dependency is not None:
        details["dependency"] = dependency
    return write_prediction_output(root, branch_input, predictions, artifacts, details)
