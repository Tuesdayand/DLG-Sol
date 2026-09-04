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
from rdkit import Chem, RDLogger, rdBase
from sklearn.model_selection import train_test_split


SOURCE_COLUMNS = ("SMILES", "Solubility")
EXPECTED_RECIPE_SHA256 = "a48e800a2a27531763e54f47fe1dcb4f6af1dcbf0646471a586f72969366e184"
EXPECTED_RDKIT_VERSION = "2023.09.6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source(path: Path, contract: dict) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size != contract["bytes"] or _sha256(path) != contract["sha256"]:
        raise ValueError("TDC runtime export does not match the pinned historical object")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("TDC runtime-export columns differ from the recorded contract")
        rows = list(reader)
    if len(rows) != contract["rows"]:
        raise ValueError("TDC runtime-export row count differs from the recorded contract")
    return rows


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def reconstruct_tdc_native_partitions(targets: list[float], test_rows: int, seed: int, target_quantiles: int, outer_folds: int, selection_fraction: float) -> list[tuple[str, int, str]]:
    values = np.asarray(targets, dtype=float)
    bins = np.asarray(pd.qcut(values, q=target_quantiles, labels=False, duplicates="drop"), dtype=int)
    random = np.random.default_rng(seed)
    outer_assignment = np.full(len(values), -1, dtype=int)
    for target_bin in np.unique(bins):
        indices = np.flatnonzero(bins == target_bin)
        random.shuffle(indices)
        outer_assignment[indices] = np.arange(len(indices)) % outer_folds
    if np.any(outer_assignment < 0):
        raise ValueError("TDC outer-fold reconstruction left rows unassigned")
    assignments: list[tuple[str, int, str]] = []
    for fold_index in range(outer_folds):
        held_out = np.flatnonzero(outer_assignment == fold_index)
        remaining = np.flatnonzero(outer_assignment != fold_index)
        fit_indices, selection_indices = train_test_split(
            remaining,
            test_size=selection_fraction,
            random_state=seed + fold_index,
            stratify=bins[remaining],
        )
        native_roles = np.full(len(values), "oof_holdout", dtype=object)
        native_roles[fit_indices] = "fit"
        native_roles[selection_indices] = "selection"
        assignments.extend((f"tdc_train_{row_index:05d}", fold_index, str(native_roles[row_index])) for row_index in range(len(values)))
        assignments.extend((f"tdc_test_{row_index:05d}", fold_index, "official_test") for row_index in range(test_rows))
        if len(held_out) != int(np.count_nonzero(native_roles == "oof_holdout")):
            raise ValueError("TDC held-out membership differs from the reconstructed native roles")
    return assignments


def prepare_tdc_input_package(train_val_source: str | Path, test_source: str | Path, output: str | Path, evaluation_id: str, retrieval_date: str, recipe_path: str | Path) -> dict:
    date.fromisoformat(retrieval_date)
    if evaluation_id not in {"R03", "R04"}:
        raise ValueError("TDC preparer supports only R03 and R04")
    recipe_file = Path(recipe_path)
    if _sha256(recipe_file) != EXPECTED_RECIPE_SHA256:
        raise ValueError("TDC partition recipe differs from the recorded contract")
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    train_rows = _read_source(Path(train_val_source), recipe["source"]["train_val"])
    test_rows = _read_source(Path(test_source), recipe["source"]["test"])
    if rdBase.rdkitVersion != EXPECTED_RDKIT_VERSION:
        raise ValueError(f"TDC reconstruction requires RDKit {EXPECTED_RDKIT_VERSION}")
    RDLogger.DisableLog("rdApp.*")
    records: list[tuple[str, str, str]] = []
    canonical_train: list[str] = []
    canonical_test: list[str] = []
    for source_name, source_rows, canonical_values in (
        ("train", train_rows, canonical_train),
        ("test", test_rows, canonical_test),
    ):
        for row_index, row in enumerate(source_rows):
            smiles = row["SMILES"].strip()
            target = row["Solubility"].strip()
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or not target or not math.isfinite(float(target)):
                raise ValueError("TDC structures and targets must be parseable, complete, and finite")
            canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
            canonical_values.append(canonical_smiles)
            records.append((f"tdc_{source_name}_{row_index:05d}", canonical_smiles, target))
    if len(set(canonical_train)) != len(canonical_train) or len(set(canonical_test)) != len(canonical_test):
        raise ValueError("TDC train or test structures are not unique after historical canonicalization")
    if set(canonical_train) & set(canonical_test):
        raise ValueError("TDC train and test structures overlap after historical canonicalization")
    partition = recipe["partition"]
    native_assignments = reconstruct_tdc_native_partitions(
        [float(row["Solubility"]) for row in train_rows],
        len(test_rows),
        partition["seed"],
        partition["target_quantiles"],
        partition["outer_folds"],
        partition["selection_fraction_of_outer_training"],
    )
    evaluation = recipe["evaluations"][evaluation_id]
    role_map = evaluation["native_role_map"]
    assignments = [(record_id, fold_index, role_map[native_role]) for record_id, fold_index, native_role in native_assignments]
    assignment_hash = hashlib.sha256("".join(f"{record_id},{fold_index},{role}\n" for record_id, fold_index, role in assignments).encode()).hexdigest()
    if assignment_hash != evaluation["assignment_sha256"]:
        raise ValueError("reconstructed TDC assignment identity differs from the recorded protocol")
    observed_counts = {}
    for fold_index in range(partition["outer_folds"]):
        observed_counts[str(fold_index)] = dict(Counter(role for _, fold, role in assignments if fold == fold_index))
    if observed_counts != evaluation["expected_role_counts"]:
        raise ValueError("reconstructed TDC role counts differ from the recorded protocol")
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("output directory must be absent or empty")
    output_path.mkdir(parents=True, exist_ok=True)
    molecules_path = output_path / "molecules.csv"
    labels_path = output_path / "labels.csv"
    partitions_path = output_path / "partitions.csv"
    derivation_path = output_path / "derivation.json"
    _write_csv(molecules_path, ("record_id", "smiles"), [(record_id, smiles) for record_id, smiles, _ in records])
    _write_csv(labels_path, ("record_id", "logS"), [(record_id, target) for record_id, _, target in records])
    _write_csv(partitions_path, ("record_id", "fold_index", "role"), assignments)
    output_hashes = {"molecules": _sha256(molecules_path), "labels": _sha256(labels_path), "partitions": _sha256(partitions_path)}
    if output_hashes != evaluation["output_sha256"]:
        raise ValueError("prepared TDC package rows differ from the recorded identities")
    canonicalization_policy = "RDKit 2023.09.6 canonical SMILES applied to the pinned historical TDC runtime exports"
    scored_description = "outer-fold holdout rows" if evaluation_id == "R03" else "the supplied TDC test rows in every outer fold"
    split_origin = f"five target-stratified outer folds with train-only selection rows; scored rows are {scored_description}"
    source_objects = [
        {
            "source_id": "tdc_dataset",
            "locator": recipe["source"][name]["locator"],
            "revision": recipe["source"]["revision"],
            "retrieval_date": retrieval_date,
            "sha256": recipe["source"][name]["sha256"],
            "bytes": recipe["source"][name]["bytes"],
        }
        for name in ("train_val", "test")
    ]
    derivation = {
        "schema_version": 1,
        "source_objects": source_objects,
        "transformations": [
            {"step": 1, "operation": "verify both pinned historical TDC runtime-export identities", "software": "DLG-Sol input preparer"},
            {"step": 2, "operation": "canonicalize SMILES and assign deterministic train/test record identifiers in source-row order", "software": "RDKit 2023.09.6 and DLG-Sol input preparer"},
            {"step": 3, "operation": "reconstruct five target-stratified outer folds and train-only selection rows from the recorded seed policy", "software": "NumPy 1.26.4, pandas 2.2.3, and scikit-learn 1.6.1"},
            {"step": 4, "operation": f"map native fit, selection, holdout, and supplied-test roles to the {evaluation_id} normalized role contract", "software": "DLG-Sol input preparer"},
        ],
        "record_id_policy": "tdc_train_ or tdc_test_ plus the zero-based source-row index padded to five digits",
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
        files[role] = {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size, "rows": len(records) if role != "partitions" else len(assignments), "columns": columns}
    files["derivation"] = {"path": derivation_path.name, "sha256": _sha256(derivation_path), "bytes": derivation_path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "dataset_id": "tdc",
        "upstream_source_id": "tdc_dataset",
        "upstream_revision": recipe["source"]["revision"],
        "retrieval_date": retrieval_date,
        "acquisition_mode": "user_supplied_local_package",
        "redistribution_status": "not_bundled_local_use_only",
        "canonicalization_policy": canonicalization_policy,
        "split_assignment_origin": split_origin,
        "files": files,
    }
    (output_path / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "evaluation_id": evaluation_id, "rows": len(records), "partition_rows": len(assignments), "role_counts": observed_counts, "files": files}
