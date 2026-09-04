from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..geometry.config import load_geometry_config
from ..geometry.graph import build_geometry_graphs
from ..geometry.modeling import DistanceAware3DMPNN
from ..geometry.training import fit_geometry_fixed_epochs, fit_geometry_model, predict_geometry, sample_geometry_parameters
from .branch_io import MaterializedBranchInput, write_prediction_output


def _all_molecules(branch_input):
    frames = [frame.loc[:, ["record_id", "smiles"]] for frame in branch_input.stages.values() if len(frame)]
    frame = pd.concat(frames, ignore_index=True)
    if (frame.groupby("record_id")["smiles"].nunique() > 1).any():
        raise ValueError("materialized record identifiers map to conflicting SMILES")
    return frame.drop_duplicates("record_id", keep="first").reset_index(drop=True)


def _labels(frame):
    values = pd.Series(pd.to_numeric(frame["logS"], errors="coerce").to_numpy(float), index=frame["record_id"].astype(str))
    if not len(values) or not np.isfinite(values.to_numpy()).all():
        raise ValueError("geometry fit labels must be finite and non-empty")
    return values


def _graphs_for(ids, graph_map):
    result = [graph for identifier in ids for graph in graph_map.get(str(identifier), ())]
    if not result:
        raise ValueError("geometry stage has no successfully constructed graphs")
    return result


def _load_dependency(path, evaluation_id, fold_index):
    root = Path(path)
    manifest_path = root / "run_manifest.json"
    model_path = root / "model.pt"
    parameters_path = root / "model_parameters.json"
    selection_path = root / "selection.json"
    if not all(path.is_file() for path in (manifest_path, model_path, parameters_path, selection_path)):
        raise ValueError("geometry dependency run is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_id") != evaluation_id or manifest.get("branch_id") != "geometry" or int(manifest.get("fold_index")) != int(fold_index):
        raise ValueError("geometry dependency run identity mismatch")
    if manifest.get("scored_labels_accessed") is not False:
        raise ValueError("geometry dependency violates the scored-label firewall")
    return manifest_path, model_path, json.loads(parameters_path.read_text(encoding="utf-8")), json.loads(selection_path.read_text(encoding="utf-8"))


def _build_model(config, parameters):
    return DistanceAware3DMPNN(config.node_dimension, config.edge_dimension, config.auxiliary_dimension, int(parameters["hidden_dim"]), int(parameters["num_layers"]), float(parameters["dropout"]), str(parameters["pool"]))


def _load_profile(path, evaluation_id):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = payload.get("execution_profiles", {})
    if set(profiles) != {"R01", "R02", "R03", "R04", "R05", "R09"}:
        raise ValueError("geometry execution profiles are incomplete")
    profile = profiles[str(evaluation_id)]
    if not isinstance(profile.get("mode"), str) or not isinstance(profile.get("auxiliary_transform"), str):
        raise ValueError("geometry execution profile is malformed")
    return profile


def _profile_seed(profile, fold_index, seed_offset):
    return int(profile["seed_base"]) + int(profile["fold_seed_multiplier"]) * int(fold_index) + int(seed_offset)


def _stabilize_auxiliary(graph_map, fit_ids):
    identifiers = set(str(value) for value in fit_ids)
    fit_graphs = [graph for identifier in identifiers for graph in graph_map.get(identifier, ())]
    if not fit_graphs:
        raise ValueError("auxiliary transformation requires successful fit graphs")
    matrix = np.vstack([graph.aux.detach().cpu().numpy().reshape(-1) for graph in fit_graphs]).astype(np.float64)
    lower = np.quantile(matrix, 0.001, axis=0)
    upper = np.quantile(matrix, 0.999, axis=0)
    clipped = np.clip(matrix, lower, upper)
    mean = clipped.mean(axis=0)
    standard_deviation = clipped.std(axis=0)
    standard_deviation = np.where(standard_deviation < 1e-8, 1.0, standard_deviation)
    for graphs in graph_map.values():
        for graph in graphs:
            values = graph.aux.detach().cpu().numpy().reshape(-1).astype(np.float64)
            transformed = (np.clip(values, lower, upper) - mean) / standard_deviation
            graph.aux = torch.as_tensor(transformed, dtype=torch.float32).view(1, -1)
    return {
        "fit_scope": "parameter_fit",
        "quantiles": [0.001, 0.999],
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mean": mean.tolist(),
        "standard_deviation": standard_deviation.tolist(),
    }


def _infer_complete(model, graph_map, molecules, fit_ids, parameters, device):
    successful_graphs = _graphs_for(molecules["record_id"].astype(str), graph_map)
    result = predict_geometry(model, successful_graphs, int(parameters["batch_size"]), device)
    successful = {identifier: index for index, identifier in enumerate(result.record_ids)}
    dimension = result.graph_embeddings.shape[1] + result.auxiliary_features.shape[1]
    embeddings = np.full((len(molecules), dimension), np.nan, dtype=np.float32)
    predictions = np.full(len(molecules), np.nan, dtype=np.float32)
    failed = np.ones(len(molecules), dtype=bool)
    for row, identifier in enumerate(molecules["record_id"].astype(str)):
        if identifier in successful:
            index = successful[identifier]
            embeddings[row] = np.concatenate([result.graph_embeddings[index], result.auxiliary_features[index]])
            predictions[row] = result.predictions[index]
            failed[row] = False
    fit_positions = [index for index, identifier in enumerate(molecules["record_id"].astype(str)) if identifier in set(fit_ids) and not failed[index]]
    if not fit_positions:
        raise ValueError("geometry fallback requires at least one successful fit record")
    fallback = embeddings[fit_positions].mean(axis=0).astype(np.float32)
    embeddings[failed] = fallback
    graph_dimension = result.graph_embeddings.shape[1]
    model.to(device)
    model.eval()
    with torch.no_grad():
        graph_embedding = torch.as_tensor(fallback[:graph_dimension], dtype=torch.float32, device=device).view(1, -1)
        auxiliary = torch.as_tensor(fallback[graph_dimension:], dtype=torch.float32, device=device).view(1, -1)
        fallback_prediction = float(model.forward_from_embedding(graph_embedding, auxiliary).detach().cpu().reshape(-1)[0])
    predictions[failed] = fallback_prediction
    if not np.isfinite(predictions).all() or not np.isfinite(embeddings).all():
        raise ValueError("geometry fallback did not produce complete finite outputs")
    return predictions, embeddings, failed


def run_geometry_unit(branch_input: MaterializedBranchInput, output: str | Path, config_path: str | Path, device: str, seed_offset: int = 0, reuse_run: str | Path | None = None, settings_run: str | Path | None = None) -> dict:
    if branch_input.branch_id != "geometry":
        raise ValueError("geometry execution requires a geometry materialization")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("geometry output directory must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    config = load_geometry_config(config_path)
    profile = _load_profile(config_path, branch_input.evaluation_id)
    seed = _profile_seed(profile, branch_input.fold_index, seed_offset) if "seed_base" in profile else None
    molecules = _all_molecules(branch_input)
    builds = [build_geometry_graphs(row.smiles, row.record_id, config, index) for index, row in enumerate(molecules.itertuples(index=False))]
    graph_map = {item.record_id: item.graphs for item in builds}
    transform = None
    if profile["auxiliary_transform"] == "fit_partition_quantile_0.001_0.999_then_zscore":
        transform = _stabilize_auxiliary(graph_map, branch_input.stages["parameter_fit"]["record_id"].astype(str))
    elif profile["auxiliary_transform"] != "none":
        raise ValueError("unsupported geometry auxiliary transformation")
    selection = {}
    dependency = None
    if branch_input.recipe.execution_type == "reuse_companion_model":
        if reuse_run is None or settings_run is not None:
            raise ValueError("geometry companion reuse requires exactly one matching R03 run")
        dependency_manifest, model_source, parameters, selection = _load_dependency(reuse_run, "R03", branch_input.fold_index)
        model = _build_model(config, parameters)
        model.load_state_dict(torch.load(model_source, map_location="cpu", weights_only=True), strict=True)
        dependency = {"evaluation_id": "R03", "run_manifest_sha256": hashlib.sha256(dependency_manifest.read_bytes()).hexdigest()}
        fit_rows = branch_input.stages["parameter_fit"]
        seed = None
    elif branch_input.evaluation_id == "R09":
        if settings_run is None or reuse_run is not None:
            raise ValueError("R09 geometry refit requires an R02 selected-settings run")
        dependency_manifest, _, parameters, selected = _load_dependency(settings_run, "R02", branch_input.fold_index)
        best_epoch = int(selected["best_epoch"])
        if parameters != profile["parameters"] or best_epoch != int(profile["best_epoch"]):
            raise ValueError("R09 dependency differs from the recorded R02 geometry settings")
        fit_rows = branch_input.stages["final_refit"]
        fit_graphs = _graphs_for(fit_rows["record_id"].astype(str), graph_map)
        model = fit_geometry_fixed_epochs(fit_graphs, _labels(fit_rows), parameters, best_epoch, config.gradient_clip_norm, seed, device)
        selection = {"selected_trial": int(selected["selected_trial"]), "best_epoch": best_epoch, "source_evaluation_id": "R02"}
        dependency = {"evaluation_id": "R02", "run_manifest_sha256": hashlib.sha256(dependency_manifest.read_bytes()).hexdigest()}
    else:
        if reuse_run is not None or settings_run is not None:
            raise ValueError("new geometry training does not accept dependency runs")
        if branch_input.evaluation_id == "R02":
            parameters = dict(profile["parameters"])
            best_epoch = int(profile["best_epoch"])
            selection = {"selected_trial": int(profile["selected_trial"]), "best_epoch": best_epoch, "selection_mode": "frozen_global_preselection"}
            fit_rows = branch_input.stages["final_refit"]
            fit_graphs = _graphs_for(fit_rows["record_id"].astype(str), graph_map)
            model = fit_geometry_fixed_epochs(fit_graphs, _labels(fit_rows), parameters, best_epoch, config.gradient_clip_norm, seed, device)
        else:
            fit_rows = branch_input.stages["parameter_fit"]
            score_rows = branch_input.stages["selection_score"]
            fit_graphs = _graphs_for(fit_rows["record_id"].astype(str), graph_map)
            score_graphs = _graphs_for(score_rows["record_id"].astype(str), graph_map)
            candidate_count = int(profile["candidate_count"])
            if profile["mode"] == "fixed_transferred_parameters_with_checkpoint_selection":
                candidates = [(int(profile["selected_trial"]), dict(profile["parameters"]))]
            elif profile["mode"] == "four_candidate_fold_local_selection":
                candidates = [(trial, sample_geometry_parameters(seed, trial)) for trial in range(candidate_count)]
            else:
                raise ValueError("unsupported geometry training mode")
            best = None
            records = []
            for trial, candidate in candidates:
                fitted = fit_geometry_model(
                    fit_graphs,
                    _labels(fit_rows),
                    score_graphs,
                    _labels(score_rows),
                    candidate,
                    int(profile["maximum_epochs"]),
                    int(profile["patience"]),
                    config.gradient_clip_norm,
                    int(seed) + int(trial),
                    device,
                )
                record = {"trial": trial, "best_epoch": fitted.best_epoch, "selection_rmse": fitted.best_selection_rmse, "parameters": candidate}
                records.append(record)
                key = (fitted.best_selection_rmse, trial)
                if best is None or key < best[0]:
                    best = (key, record, fitted.model)
            _, selected, model = best
            parameters = selected["parameters"]
            selection = {"selected_trial": selected["trial"], "best_epoch": selected["best_epoch"], "best_selection_rmse": selected["selection_rmse"], "trials": records}
    all_predictions, embeddings, failed = _infer_complete(model, graph_map, molecules, set(fit_rows["record_id"].astype(str)), parameters, device)
    position = pd.Series(np.arange(len(molecules)), index=molecules["record_id"].astype(str))
    scored_ids = branch_input.stages["scored_prediction"]["record_id"].astype(str).tolist()
    predictions = all_predictions[position.loc[scored_ids].to_numpy()]
    model_path = root / "model.pt"
    parameters_path = root / "model_parameters.json"
    selection_path = root / "selection.json"
    embedding_path = root / "embeddings.npz"
    status_path = root / "conformer_status.csv"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, model_path)
    parameters_path.write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(embedding_path, record_ids=molecules["record_id"].astype(str).to_numpy(dtype=str), embeddings=embeddings, failed_mask=failed)
    pd.DataFrame({"record_id": molecules["record_id"].astype(str), "status": [item.status for item in builds], "conformers_retained": [item.conformers_retained for item in builds]}).to_csv(status_path, index=False, lineterminator="\n")
    artifacts = {path.name: path for path in (model_path, parameters_path, selection_path, embedding_path, status_path)}
    if transform is not None:
        transform_path = root / "auxiliary_transform.json"
        transform_path.write_text(json.dumps(transform, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts[transform_path.name] = transform_path
    details = {
        "device": str(device),
        "seed": seed,
        "seed_offset": int(seed_offset),
        "execution_profile": profile["mode"],
        "selected_trial": int(selection["selected_trial"]),
        "best_epoch": int(selection["best_epoch"]),
        "candidate_count": int(profile.get("candidate_count", 0)),
        "failed_records": int(failed.sum()),
        "embedding_dimension": int(embeddings.shape[1]),
    }
    if dependency is not None:
        details["dependency"] = dependency
    return write_prediction_output(root, branch_input, predictions, artifacts, details)
