from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..descriptors import (
    calculate_mordred_2d,
    filter_config_for,
    fit_descriptor_schema,
    fit_xgb_regressor,
    generate_candidates,
    load_descriptor_protocols,
    transform_descriptor_frame,
)
from ..descriptors.filtering import DescriptorFilterConfig, DescriptorSchema
from ..label_firewall import ensure_label_free
from ..metrics import regression_metrics
from ..training.selection import select_nested_trial
from .branch_io import MaterializedBranchInput, write_prediction_output


PROTOCOL_BY_EVALUATION = {"R01": "aqsoldbc", "R02": "complat", "R03": "tdc", "R04": "tdc", "R05": "jchem", "R09": "complat"}


def _descriptor_table(branch_input: MaterializedBranchInput, source: str | Path | None) -> pd.DataFrame:
    frames = [frame.loc[:, ["record_id", "smiles"]] for frame in branch_input.stages.values() if len(frame)]
    frames.extend(frame.loc[:, ["record_id", "smiles"]] for unit in branch_input.nested.values() for frame in unit.values())
    molecules = pd.concat(frames, ignore_index=True).drop_duplicates("record_id", keep="first")
    conflicts = pd.concat(frames, ignore_index=True).groupby("record_id")["smiles"].nunique()
    if (conflicts > 1).any():
        raise ValueError("materialized record identifiers map to conflicting SMILES")
    if source is None:
        return calculate_mordred_2d(molecules["smiles"], molecules["record_id"])
    frame = pd.read_csv(source, low_memory=False, dtype={"record_id": str})
    if "record_id" not in frame or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise ValueError("descriptor cache must contain unique non-missing record identifiers")
    ensure_label_free(frame.columns)
    required = set(molecules["record_id"].astype(str))
    if not required.issubset(set(frame["record_id"].astype(str))):
        raise ValueError("descriptor cache does not cover the materialized unit")
    return frame.loc[frame["record_id"].astype(str).isin(required)].copy()


def _index(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in frame.columns if column != "record_id"]
    if not columns:
        raise ValueError("descriptor table contains no feature columns")
    result = frame.copy()
    result["record_id"] = result["record_id"].astype(str)
    return result.set_index("record_id").loc[:, columns]


def _features(indexed: pd.DataFrame, rows: pd.DataFrame, schema) -> np.ndarray:
    identifiers = rows["record_id"].astype(str).tolist()
    missing = set(identifiers) - set(indexed.index)
    if missing:
        raise ValueError("descriptor table is missing materialized rows")
    return transform_descriptor_frame(indexed.loc[identifiers], schema)


def _labels(rows: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(rows["logS"], errors="coerce").to_numpy(float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("descriptor fit labels must be finite and non-empty")
    return values


def _candidate_parameters(protocol: str, fold: int) -> list[dict]:
    return generate_candidates(protocol, fold)


def _fit_candidate(indexed, fit_rows, selection_rows, config, parameters, seed, device, n_jobs, early_stopping):
    schema = fit_descriptor_schema(indexed.loc[fit_rows["record_id"].astype(str)], config)
    x_fit = _features(indexed, fit_rows, schema)
    x_selection = _features(indexed, selection_rows, schema)
    model = fit_xgb_regressor(
        x_fit,
        _labels(fit_rows),
        parameters,
        seed,
        device,
        n_jobs,
        x_selection if early_stopping is not None else None,
        _labels(selection_rows) if early_stopping is not None else None,
        early_stopping,
    )
    predictions = np.asarray(model.predict(x_selection), dtype=float)
    metrics = regression_metrics(_labels(selection_rows), predictions)
    best_iteration = None if early_stopping is None else int(model.best_iteration)
    return model, schema, metrics, best_iteration


def _schema_payload(schema) -> dict:
    payload = asdict(schema)
    payload["input_columns"] = list(schema.input_columns)
    payload["retained_columns"] = list(schema.retained_columns)
    payload["dropped_missing"] = list(schema.dropped_missing)
    payload["dropped_constant"] = list(schema.dropped_constant)
    payload["dropped_low_variance"] = list(schema.dropped_low_variance)
    payload["dropped_high_correlation"] = list(schema.dropped_high_correlation)
    return payload


def _schema_from_payload(payload: dict) -> DescriptorSchema:
    config = payload["config"]
    return DescriptorSchema(
        input_columns=tuple(payload["input_columns"]),
        retained_columns=tuple(payload["retained_columns"]),
        dropped_missing=tuple(payload["dropped_missing"]),
        dropped_constant=tuple(payload["dropped_constant"]),
        dropped_low_variance=tuple(payload["dropped_low_variance"]),
        dropped_high_correlation=tuple(payload["dropped_high_correlation"]),
        config=DescriptorFilterConfig(**config),
    )


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_nested(branch_input, indexed, protocol, protocol_record, config, output, device, n_jobs):
    candidates = _candidate_parameters(protocol, branch_input.fold_index)
    records = []
    parameters_by_trial = {}
    for source in candidates:
        parameters = dict(source)
        trial = int(parameters.pop("trial"))
        parameters["n_estimators"] = int(protocol_record["search"]["search_tree_budget"])
        parameters_by_trial[trial] = parameters
        for inner_fold, unit in sorted(branch_input.nested.items()):
            _, _, metrics, best_iteration = _fit_candidate(
                indexed,
                unit["selection_fit"],
                unit["selection_score"],
                config,
                parameters,
                int(protocol_record["search"]["base_seed"]) + trial,
                device,
                n_jobs,
                int(protocol_record["search"]["early_stopping_rounds"]),
            )
            n = len(unit["selection_score"])
            records.append({"trial": trial, "inner_validation_fold": inner_fold, "n": n, "sse": metrics.rmse**2 * n, "rmse": metrics.rmse, "mae": metrics.mae, "best_iteration": best_iteration})
    selected = select_nested_trial(pd.DataFrame(records))
    final_rows = branch_input.stages["final_refit"]
    schema = fit_descriptor_schema(indexed.loc[final_rows["record_id"].astype(str)], config)
    parameters = dict(parameters_by_trial[selected.trial])
    parameters["n_estimators"] = selected.final_tree_count
    scored_x = _features(indexed, branch_input.stages["scored_prediction"], schema)
    fit_x = _features(indexed, final_rows, schema)
    predictions = []
    model_paths = []
    for member, seed in enumerate(protocol_record["search"]["final_ensemble_seeds"]):
        model = fit_xgb_regressor(fit_x, _labels(final_rows), parameters, int(seed), device, n_jobs)
        predictions.append(np.asarray(model.predict(scored_x), dtype=float))
        path = output / f"model_{member}.json"
        model.save_model(path)
        model_paths.append(path)
    return np.mean(predictions, axis=0), schema, model_paths, {"selected_trial": selected.trial, "final_tree_count": selected.final_tree_count, "member_seeds": list(protocol_record["search"]["final_ensemble_seeds"]), "trial_records": records}


def _run_selected(branch_input, indexed, protocol, protocol_record, config, output, device, n_jobs):
    fit_rows = branch_input.stages["selection_fit"] if protocol == "complat" else branch_input.stages["parameter_fit"]
    selection_rows = branch_input.stages["selection_score"]
    candidates = _candidate_parameters(protocol, branch_input.fold_index)
    early = protocol_record["search"].get("early_stopping_rounds")
    tree_budget = protocol_record["search"].get("search_tree_budget")
    records = []
    best = None
    for source in candidates:
        parameters = dict(source)
        trial = int(parameters.pop("trial"))
        if tree_budget is not None:
            parameters["n_estimators"] = int(tree_budget)
        base = int(protocol_record["search"]["base_seed"])
        seed = base + trial if protocol == "complat" else base + branch_input.fold_index * 100 + trial
        model, schema, metrics, best_iteration = _fit_candidate(indexed, fit_rows, selection_rows, config, parameters, seed, device, n_jobs, early)
        record = {"trial": trial, "validation_rmse": metrics.rmse, "validation_mae": metrics.mae, "best_iteration": best_iteration, "parameters": parameters}
        records.append(record)
        key = (metrics.rmse, trial)
        if best is None or key < best[0]:
            best = (key, trial, parameters, model, schema, seed, best_iteration)
    if best is None:
        raise RuntimeError("descriptor selection produced no fitted model")
    _, selected_trial, parameters, model, schema, seed, best_iteration = best
    if protocol == "complat":
        expected = int(protocol_record["search"]["selected_trial"])
        if selected_trial != expected:
            raise ValueError("fresh ComPlat global selection differs from the recorded frozen trial")
        refit_rows = branch_input.stages["final_refit"]
        schema = fit_descriptor_schema(indexed.loc[refit_rows["record_id"].astype(str)], config)
        seed = int(protocol_record["search"]["base_seed"]) + (999 if branch_input.evaluation_id == "R09" else branch_input.fold_index)
        model = fit_xgb_regressor(_features(indexed, refit_rows, schema), _labels(refit_rows), parameters, seed, device, n_jobs)
    predictions = np.asarray(model.predict(_features(indexed, branch_input.stages["scored_prediction"], schema)), dtype=float)
    model_path = output / "model.json"
    model.save_model(model_path)
    return predictions, schema, [model_path], {"selected_trial": selected_trial, "model_seed": seed, "best_iteration": best_iteration, "trial_records": records}


def _run_reuse(branch_input, indexed, output, reuse_run):
    source = Path(reuse_run)
    manifest_path = source / "run_manifest.json"
    schema_path = source / "descriptor_schema.json"
    model_path = source / "model.json"
    if not manifest_path.is_file() or not schema_path.is_file() or not model_path.is_file():
        raise ValueError("descriptor companion run is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = branch_input.recipe.dependency
    if manifest.get("evaluation_id") != expected["evaluation_id"] or manifest.get("branch_id") != "descriptor" or int(manifest.get("fold_index")) != branch_input.fold_index:
        raise ValueError("descriptor companion run does not match the declared dependency")
    if manifest.get("scored_labels_accessed") is not False:
        raise ValueError("descriptor companion run violates the scored-label firewall")
    schema = _schema_from_payload(json.loads(schema_path.read_text(encoding="utf-8")))
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(model_path)
    predictions = np.asarray(model.predict(_features(indexed, branch_input.stages["scored_prediction"], schema)), dtype=float)
    target_model = output / "model.json"
    target_schema = output / "descriptor_schema.json"
    shutil.copy2(model_path, target_model)
    shutil.copy2(schema_path, target_schema)
    reference = output / "companion_reference.json"
    _write_json(reference, {"evaluation_id": manifest["evaluation_id"], "branch_id": "descriptor", "fold_index": branch_input.fold_index, "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
    artifacts = {path.name: path for path in (target_model, target_schema, reference)}
    details = {"protocol": "tdc", "companion_evaluation_id": manifest["evaluation_id"], "model_count": 1, "descriptor_count": len(schema.retained_columns)}
    return write_prediction_output(output, branch_input, predictions, artifacts, details)


def run_descriptor_unit(branch_input: MaterializedBranchInput, output: str | Path, descriptor_cache: str | Path | None = None, device: str = "cpu", n_jobs: int = 4, reuse_run: str | Path | None = None) -> dict:
    if branch_input.branch_id != "descriptor":
        raise ValueError("descriptor execution requires a descriptor materialization")
    protocol = PROTOCOL_BY_EVALUATION[branch_input.evaluation_id]
    if branch_input.evaluation_id == "R02" and branch_input.recipe.selection_mode != "frozen_global_preselection":
        raise ValueError("R02 descriptor execution rejects fold-local reselection")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("descriptor output directory must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    descriptors = _descriptor_table(branch_input, descriptor_cache)
    indexed = _index(descriptors)
    if branch_input.recipe.execution_type == "reuse_companion_model":
        if reuse_run is None:
            raise ValueError("companion-model reuse requires the matching R03 descriptor run")
        return _run_reuse(branch_input, indexed, root, reuse_run)
    if reuse_run is not None:
        raise ValueError("new descriptor training does not accept a companion run")
    payload = load_descriptor_protocols()
    protocol_record = payload["protocols"][protocol]
    config = filter_config_for(protocol)
    if branch_input.recipe.selection_mode == "nested_inner_cv":
        predictions, schema, model_paths, details = _run_nested(branch_input, indexed, protocol, protocol_record, config, root, device, n_jobs)
    else:
        predictions, schema, model_paths, details = _run_selected(branch_input, indexed, protocol, protocol_record, config, root, device, n_jobs)
    schema_path = root / "descriptor_schema.json"
    trials_path = root / "selection.json"
    _write_json(schema_path, _schema_payload(schema))
    _write_json(trials_path, details)
    artifacts = {path.name: path for path in [*model_paths, schema_path, trials_path]}
    run_details = {"protocol": protocol, "selected_trial": details["selected_trial"], "model_count": len(model_paths), "descriptor_count": len(schema.retained_columns), "xgboost_device": str(device), "n_jobs": int(n_jobs)}
    return write_prediction_output(root, branch_input, predictions, artifacts, run_details)
