from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .execution import TrainingExecutionRecipe, TrainingExecutionRegistry


STAGES = (
    "parameter_fit",
    "selection_fit",
    "selection_score",
    "final_refit",
    "coefficient_fit",
    "scored_prediction",
)
FORBIDDEN_LABELS = {"logs", "target", "label", "y", "observed"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MaterializedBranchInput:
    root: Path
    evaluation_id: str
    branch_id: str
    fold_index: int
    manifest_sha256: str
    stages: dict[str, pd.DataFrame]
    nested: dict[int, dict[str, pd.DataFrame]]
    recipe: TrainingExecutionRecipe


def _read_csv(path: Path, expected_columns: list[str], expected: dict) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size != int(expected["bytes"]) or sha256_file(path) != expected["sha256"]:
        raise ValueError(f"materialized file identity mismatch: {path.name}")
    frame = pd.read_csv(path)
    if list(frame.columns) != expected_columns or list(expected["columns"]) != expected_columns or len(frame) != int(expected["rows"]):
        raise ValueError(f"materialized file schema or row count mismatch: {path.name}")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).duplicated().any():
        raise ValueError(f"materialized file contains invalid record identifiers: {path.name}")
    frame["record_id"] = frame["record_id"].astype(str)
    frame["smiles"] = frame["smiles"].astype(str)
    if expected_columns[-1] == "logS":
        labels = pd.to_numeric(frame["logS"], errors="coerce").to_numpy(float)
        if not np.isfinite(labels).all():
            raise ValueError(f"materialized file contains non-finite labels: {path.name}")
        frame["logS"] = labels
    elif {str(column).lower() for column in frame.columns} & FORBIDDEN_LABELS:
        raise ValueError("scored materialization contains labels")
    return frame


def load_materialized_branch_input(
    root: str | Path,
    registry: TrainingExecutionRegistry,
) -> MaterializedBranchInput:
    base = Path(root)
    manifest_path = base / "materialization_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("materialized unit manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("redistribution_status") != "user_local_do_not_redistribute":
        raise ValueError("materialized unit manifest is unsupported")
    evaluation_id = str(manifest.get("evaluation_id"))
    branch_id = str(manifest.get("branch_id"))
    fold_index = int(manifest.get("fold_index"))
    recipe = registry.get(evaluation_id, branch_id)
    files = manifest.get("files", {})
    if set(files) != {f"{stage}.csv" for stage in STAGES}:
        raise ValueError("materialized unit file set is incomplete")
    stages = {}
    for stage in STAGES:
        columns = ["record_id", "smiles"] if stage == "scored_prediction" else ["record_id", "smiles", "logS"]
        stages[stage] = _read_csv(base / f"{stage}.csv", columns, files[f"{stage}.csv"])
    populated = tuple(stage for stage in STAGES if len(stages[stage]))
    if set(populated) != set(recipe.input_stages):
        raise ValueError("materialized stage set differs from the execution recipe")
    nested_records = manifest.get("nested_files", {})
    nested = {}
    for value, records in nested_records.items():
        inner_fold = int(value)
        expected_names = {f"nested_{inner_fold}/selection_fit.csv", f"nested_{inner_fold}/selection_score.csv"}
        if set(records) != expected_names:
            raise ValueError("nested materialized file set is incomplete")
        nested[inner_fold] = {}
        for stage in ("selection_fit", "selection_score"):
            relative = f"nested_{inner_fold}/{stage}.csv"
            nested[inner_fold][stage] = _read_csv(base / relative, ["record_id", "smiles", "logS"], records[relative])
    if recipe.selection_mode == "nested_inner_cv" and not nested:
        raise ValueError("nested execution recipe requires nested materializations")
    if recipe.selection_mode != "nested_inner_cv" and nested:
        raise ValueError("non-nested execution recipe received nested materializations")
    return MaterializedBranchInput(base, evaluation_id, branch_id, fold_index, sha256_file(manifest_path), stages, nested, recipe)


def align_matrix(record_ids, matrix, expected_ids, name: str) -> np.ndarray:
    identifiers = pd.Series(record_ids, dtype="string")
    values = np.asarray(matrix, dtype=np.float32)
    expected = tuple(str(value) for value in expected_ids)
    if identifiers.isna().any() or identifiers.duplicated().any() or values.ndim != 2 or len(values) != len(identifiers):
        raise ValueError(f"{name} embedding package is invalid")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} embeddings must be finite")
    index = pd.Series(np.arange(len(identifiers)), index=identifiers.astype(str))
    if set(index.index) != set(expected):
        raise ValueError(f"{name} embedding record set differs from the materialized stage")
    return values[index.loc[list(expected)].to_numpy()]


def write_prediction_output(
    output: str | Path,
    branch_input: MaterializedBranchInput,
    predictions,
    artifacts: dict[str, Path],
    run_details: dict,
) -> dict:
    root = Path(output)
    expected_artifacts = {str(name) for name in artifacts}
    if root.exists():
        observed = {path.name for path in root.iterdir()}
        if observed - expected_artifacts:
            raise ValueError("branch output directory contains undeclared files")
    root.mkdir(parents=True, exist_ok=True)
    scored = branch_input.stages["scored_prediction"]
    values = np.asarray(predictions, dtype=float).reshape(-1)
    if values.shape != (len(scored),) or not np.isfinite(values).all():
        raise ValueError("branch predictions must contain one finite value per scored row")
    prediction_path = root / "predictions.csv"
    pd.DataFrame({"record_id": scored["record_id"], "prediction": values}).to_csv(prediction_path, index=False, lineterminator="\n")
    artifact_records = {
        "predictions.csv": {"bytes": prediction_path.stat().st_size, "sha256": sha256_file(prediction_path)}
    }
    for name, source in artifacts.items():
        path = Path(source)
        if path.parent.resolve() != root.resolve() or not path.is_file() or name != path.name:
            raise ValueError("branch artifacts must be files written directly in the output directory")
        artifact_records[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": 1,
        "evaluation_id": branch_input.evaluation_id,
        "branch_id": branch_input.branch_id,
        "fold_index": branch_input.fold_index,
        "selection_mode": branch_input.recipe.selection_mode,
        "execution_type": branch_input.recipe.execution_type,
        "materialization_manifest_sha256": branch_input.manifest_sha256,
        "scored_labels_accessed": False,
        "historical_weight_identity_claimed": False,
        "historical_metric_identity_claimed": False,
        "redistribution_status": "user_local_do_not_redistribute",
        "run_details": run_details,
        "artifacts": artifact_records,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    lowered = text.lower()
    private_markers = ("/data/" + "koo/", "/home/" + "koo/", "agent" + "_work", "code" + "x", "clau" + "de")
    if any(marker in lowered for marker in private_markers):
        raise ValueError("branch run manifest contains private or internal identifiers")
    (root / "run_manifest.json").write_text(text, encoding="utf-8")
    return manifest
