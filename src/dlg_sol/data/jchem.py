from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


EXPECTED_RECIPE_SHA256 = "66825dd97ac60439fe63c77ac58b8e254ed53537ba1ecd0bb522d2521856215b"
DATASET_COLUMNS = ("No.", "Split1", "Split2", "Split3", "Split4", "Split5", "ID", "Name", "InChI", "InChIKey", "SMILES", "log Sw original dataset", "SD", "Occurrences", "Group", "log Sw dataset used", "log Sw correction", "Comment", "reference", "link ECHA", "New Data")
PREDICTION_COLUMNS = ("SMILES", "SExp", "Split1", "Split2", "Split3", "Split4", "Split5", "Consensus GNN", "SD")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _canonical_smiles(value: object) -> str | None:
    molecule = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def prepare_jchem_input_package(source: str | Path, output: str | Path, retrieval_date: str, recipe_path: str | Path) -> dict:
    date.fromisoformat(retrieval_date)
    recipe_file = Path(recipe_path)
    if _sha256(recipe_file) != EXPECTED_RECIPE_SHA256:
        raise ValueError("JCheM partition recipe differs from the recorded contract")
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    source_path = Path(source)
    source_contract = recipe["source"]
    if not source_path.is_file() or source_path.stat().st_size != source_contract["bytes"] or _sha256(source_path) != source_contract["sha256"]:
        raise ValueError("JCheM workbook does not match the pinned upstream object")
    RDLogger.DisableLog("rdApp.*")
    dataset = pd.read_excel(source_path, sheet_name=source_contract["dataset_sheet"]).reset_index(names="source_row")
    predictions = pd.read_excel(source_path, sheet_name=source_contract["prediction_sheet"])
    if tuple(dataset.columns[1:]) != DATASET_COLUMNS or tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("JCheM workbook schema differs from the pinned upstream object")
    if len(dataset) != source_contract["source_rows"] or len(predictions) != source_contract["prediction_rows"]:
        raise ValueError("JCheM workbook row counts differ from the pinned upstream object")
    dataset["canonical_smiles"] = dataset["SMILES"].map(_canonical_smiles)
    dataset["target"] = pd.to_numeric(dataset["log Sw dataset used"], errors="coerce")
    usable = dataset["canonical_smiles"].notna() & dataset["target"].notna()
    invalid_rows = dataset.loc[~usable, "source_row"].astype(int).tolist()
    if int(usable.sum()) != source_contract["valid_rows"] or invalid_rows != source_contract["invalid_source_rows"]:
        raise ValueError("JCheM usable-row identity differs from the recorded protocol")
    data = dataset.loc[usable].copy()
    if not np.isfinite(data["target"].to_numpy(float)).all():
        raise ValueError("JCheM targets must be finite")
    test_memberships = [set(data.loc[data[column].eq("Test"), "canonical_smiles"]) for column in recipe["partition"]["source_columns"]]
    if any(membership != test_memberships[0] for membership in test_memberships[1:]):
        raise ValueError("JCheM fixed test membership differs across the five released splits")
    predictions["canonical_smiles"] = predictions["SMILES"].map(_canonical_smiles)
    predictions["author_logS"] = pd.to_numeric(predictions["SExp"], errors="coerce")
    predictions["author_prediction"] = pd.to_numeric(predictions["Consensus GNN"], errors="coerce")
    prediction_usable = predictions["canonical_smiles"].notna() & predictions["author_logS"].notna() & predictions["author_prediction"].notna()
    if int(prediction_usable.sum()) != source_contract["valid_prediction_rows"]:
        raise ValueError("JCheM usable prediction-row count differs from the recorded protocol")
    released = predictions.loc[prediction_usable, ["canonical_smiles", "author_logS"]]
    official = data.loc[data["Split1"].eq("Test"), ["canonical_smiles", "target"]]
    if official["canonical_smiles"].duplicated().any() or released["canonical_smiles"].duplicated().any():
        raise ValueError("JCheM fixed test structures must be unique after canonicalization")
    matched = official.merge(released, on="canonical_smiles", how="outer", indicator=True, validate="one_to_one")
    if not matched["_merge"].eq("both").all():
        raise ValueError("JCheM author prediction rows do not match the fixed test structures")
    rounding_delta = np.abs(matched["target"].to_numpy(float) - matched["author_logS"].to_numpy(float))
    if not np.isfinite(rounding_delta).all() or float(rounding_delta.max()) > recipe["label_policy"]["maximum_allowed_absolute_rounding_delta"]:
        raise ValueError("JCheM released test labels differ materially between workbook sheets")
    author_labels = dict(zip(released["canonical_smiles"], released["author_logS"]))
    test_mask = data["Split1"].eq("Test")
    data.loc[test_mask, "target"] = [author_labels[value] for value in data.loc[test_mask, "canonical_smiles"]]
    record_ids = [f"jchem2025_{int(index):05d}" for index in data["source_row"]]
    role_map = recipe["partition"]["role_map"]
    if any(set(data[column].astype(str)) != set(role_map) for column in recipe["partition"]["source_columns"]):
        raise ValueError("JCheM split values differ from the recorded Train, Val, and Test roles")
    assignments = []
    for fold, column in zip(recipe["partition"]["fold_indices"], recipe["partition"]["source_columns"]):
        assignments.extend((record_id, fold, role_map[str(role)]) for record_id, role in zip(record_ids, data[column]))
    assignment_hash = hashlib.sha256("".join(f"{record_id},{fold},{role}\n" for record_id, fold, role in assignments).encode()).hexdigest()
    if assignment_hash != recipe["expected_output"]["assignment_sha256"]:
        raise ValueError("reconstructed JCheM assignment identity differs from the recorded protocol")
    observed_counts = {}
    for fold in recipe["partition"]["fold_indices"]:
        observed_counts[str(fold)] = dict(Counter(role for _, assigned_fold, role in assignments if assigned_fold == fold))
    if observed_counts != recipe["partition"]["expected_role_counts"]:
        raise ValueError("reconstructed JCheM role counts differ from the recorded protocol")
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("output directory must be absent or empty")
    output_path.mkdir(parents=True, exist_ok=True)
    molecules_path = output_path / "molecules.csv"
    labels_path = output_path / "labels.csv"
    partitions_path = output_path / "partitions.csv"
    derivation_path = output_path / "derivation.json"
    _write_csv(molecules_path, ("record_id", "smiles"), list(zip(record_ids, data["canonical_smiles"])))
    _write_csv(labels_path, ("record_id", "logS"), list(zip(record_ids, [str(value) for value in data["target"]])))
    _write_csv(partitions_path, ("record_id", "fold_index", "role"), assignments)
    observed_hashes = {"molecules_sha256": _sha256(molecules_path), "labels_sha256": _sha256(labels_path), "partitions_sha256": _sha256(partitions_path)}
    expected_hashes = {key: recipe["expected_output"][key] for key in observed_hashes}
    if observed_hashes != expected_hashes:
        raise ValueError("prepared JCheM package rows differ from the recorded identities")
    canonicalization_policy = "RDKit 2026.03.1 canonical SMILES; two invalid source structures excluded by the recorded parse rule"
    split_origin = "released dataset_split Split1-Split5; Train to fit, Val to coefficient, and fixed Test to scored"
    derivation = {
        "schema_version": 1,
        "source_objects": [{
            "source_id": "jchem_dataset",
            "locator": source_contract["locator"],
            "revision": source_contract["revision"],
            "retrieval_date": retrieval_date,
            "sha256": source_contract["sha256"],
            "bytes": source_contract["bytes"],
        }],
        "transformations": [
            {"step": 1, "operation": "verify the pinned JCheM workbook identity and required sheet schemas", "software": "DLG-Sol input preparer"},
            {"step": 2, "operation": "canonicalize dataset and prediction-sheet SMILES and exclude two unparsable dataset rows", "software": "RDKit 2026.03.1"},
            {"step": 3, "operation": "map released Split1-Split5 Train, Val, and Test roles without resplitting", "software": "DLG-Sol input preparer"},
            {"step": 4, "operation": "replace fixed-test targets with matched SExp values from predictions_test_set while retaining full-precision development targets", "software": "DLG-Sol input preparer"},
        ],
        "record_id_policy": "jchem2025_ plus the zero-based dataset_split source-row index padded to five digits",
        "canonicalization_policy": canonicalization_policy,
        "split_assignment_origin": split_origin,
    }
    derivation_path.write_text(json.dumps(derivation, indent=2) + "\n", encoding="utf-8")
    files = {}
    for role, path, columns in (
        ("molecules", molecules_path, ["record_id", "smiles"]),
        ("labels", labels_path, ["record_id", "logS"]),
        ("partitions", partitions_path, ["record_id", "fold_index", "role"]),
    ):
        files[role] = {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size, "rows": len(data) if role != "partitions" else len(assignments), "columns": columns}
    files["derivation"] = {"path": derivation_path.name, "sha256": _sha256(derivation_path), "bytes": derivation_path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "evaluation_id": "R05",
        "dataset_id": "jchem",
        "upstream_source_id": "jchem_dataset",
        "upstream_revision": source_contract["revision"],
        "retrieval_date": retrieval_date,
        "acquisition_mode": "user_supplied_local_package",
        "redistribution_status": "not_bundled_local_use_only",
        "canonicalization_policy": canonicalization_policy,
        "split_assignment_origin": split_origin,
        "files": files,
    }
    (output_path / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "evaluation_id": "R05", "rows": len(data), "partition_rows": len(assignments), "role_counts": observed_counts, "maximum_test_label_rounding_delta": float(rounding_delta.max()), "files": files}
