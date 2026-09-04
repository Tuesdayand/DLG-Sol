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
from sklearn.model_selection import StratifiedKFold


SOURCE_COLUMNS = ("", "C_ID", "Name", "InChIKey", "SMILES", "CAS", "LogS", "smiles_canon")
EXPECTED_RECIPE_SHA256 = "8178c562efaf70d3e47d344024da6b02f539763540bd528d92fa874ef84f4651"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source(path: Path, contract: dict) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size != contract["bytes"] or _sha256(path) != contract["sha256"]:
        raise ValueError("ComPlat source file does not match the pinned upstream object")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("ComPlat source columns differ from the pinned upstream object")
        rows = list(reader)
    if len(rows) != contract["rows"]:
        raise ValueError("ComPlat source row count differs from the pinned upstream object")
    return rows


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def reconstruct_complat_oof_assignments(targets: list[float], n_splits: int, quantiles: int, seed: int) -> list[tuple[int, int, str]]:
    target_series = pd.Series(targets, dtype=float)
    bins = pd.qcut(target_series, q=quantiles, labels=False, duplicates="drop")
    fold_by_row = np.full(len(targets), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_index, (_, scored_indices) in enumerate(splitter.split(np.arange(len(targets)), bins)):
        fold_by_row[scored_indices] = fold_index
    return [
        (row_index, fold_index, "scored" if int(fold_by_row[row_index]) == fold_index else "fit")
        for fold_index in range(n_splits)
        for row_index in range(len(targets))
    ]


def prepare_complat_input_package(train_source: str | Path, test_source: str | Path | None, output: str | Path, evaluation_id: str, retrieval_date: str, recipe_path: str | Path) -> dict:
    date.fromisoformat(retrieval_date)
    if evaluation_id not in {"R02", "R09"}:
        raise ValueError("ComPlat preparer supports only R02 and R09")
    recipe_file = Path(recipe_path)
    if _sha256(recipe_file) != EXPECTED_RECIPE_SHA256:
        raise ValueError("ComPlat partition recipe differs from the recorded contract")
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    train_rows = _read_source(Path(train_source), recipe["source"]["train"])
    if evaluation_id == "R09":
        if test_source is None:
            raise ValueError("R09 requires the pinned ComPlat test source")
        test_rows = _read_source(Path(test_source), recipe["source"]["test"])
    else:
        test_rows = []
    all_ids = [row["C_ID"].strip() for row in train_rows + test_rows]
    if any(not value for value in all_ids) or len(set(all_ids)) != len(all_ids):
        raise ValueError("ComPlat record identifiers must be complete and unique across train and test")
    for row in train_rows + test_rows:
        values = (row["smiles_canon"].strip(), row["LogS"].strip())
        if not all(values) or not math.isfinite(float(values[1])):
            raise ValueError("ComPlat canonical SMILES and targets must be complete and finite")
    contract = recipe["evaluations"][evaluation_id]
    if evaluation_id == "R02":
        rows = train_rows
        indexed = reconstruct_complat_oof_assignments(
            [float(row["LogS"]) for row in rows],
            contract["n_splits"],
            contract["stratification_quantiles"],
            contract["random_state"],
        )
        assignments = [(rows[index]["C_ID"].strip(), fold, role) for index, fold, role in indexed]
    else:
        rows = train_rows + test_rows
        assignments = [(row["C_ID"].strip(), 0, "fit") for row in train_rows]
        assignments.extend((row["C_ID"].strip(), 0, "scored") for row in test_rows)
    assignment_hash = hashlib.sha256("".join(f"{record_id},{fold},{role}\n" for record_id, fold, role in assignments).encode()).hexdigest()
    if assignment_hash != contract["assignment_sha256"]:
        raise ValueError("reconstructed ComPlat assignment identity differs from the recorded protocol")
    observed_counts = {}
    for fold in sorted({item[1] for item in assignments}):
        observed_counts[str(fold)] = dict(Counter(role for _, assigned_fold, role in assignments if assigned_fold == fold))
    if observed_counts != contract["expected_role_counts"]:
        raise ValueError("reconstructed ComPlat role counts differ from the recorded protocol")
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("output directory must be absent or empty")
    output_path.mkdir(parents=True, exist_ok=True)
    molecules_path = output_path / "molecules.csv"
    labels_path = output_path / "labels.csv"
    partitions_path = output_path / "partitions.csv"
    derivation_path = output_path / "derivation.json"
    _write_csv(molecules_path, ("record_id", "smiles"), [(row["C_ID"].strip(), row["smiles_canon"].strip()) for row in rows])
    _write_csv(labels_path, ("record_id", "logS"), [(row["C_ID"].strip(), row["LogS"].strip()) for row in rows])
    _write_csv(partitions_path, ("record_id", "fold_index", "role"), assignments)
    output_hashes = {"molecules": _sha256(molecules_path), "labels": _sha256(labels_path), "partitions": _sha256(partitions_path)}
    if output_hashes != contract["output_sha256"]:
        raise ValueError("prepared ComPlat package rows differ from the recorded identities")
    canonicalization_policy = "author-supplied smiles_canon values copied without transformation"
    split_origin = "released ComPlat Train stratified five-fold OOF reconstruction" if evaluation_id == "R02" else "released ComPlat Train used for full refit and released Test used only for scoring"
    source_objects = [{
        "source_id": "complat",
        "locator": recipe["source"]["train"]["locator"],
        "revision": recipe["source"]["revision"],
        "retrieval_date": retrieval_date,
        "sha256": recipe["source"]["train"]["sha256"],
        "bytes": recipe["source"]["train"]["bytes"],
    }]
    if evaluation_id == "R09":
        source_objects.append({
            "source_id": "complat",
            "locator": recipe["source"]["test"]["locator"],
            "revision": recipe["source"]["revision"],
            "retrieval_date": retrieval_date,
            "sha256": recipe["source"]["test"]["sha256"],
            "bytes": recipe["source"]["test"]["bytes"],
        })
    transformations = [
        {"step": 1, "operation": "verify the pinned ComPlat source-file identities required by this evaluation", "software": "DLG-Sol input preparer"},
        {"step": 2, "operation": "copy C_ID, smiles_canon, and LogS in released row order", "software": "DLG-Sol input preparer"},
    ]
    if evaluation_id == "R02":
        transformations.append({"step": 3, "operation": "reconstruct five stratified OOF folds from 20 target quantiles and random state 260722", "software": "pandas qcut and scikit-learn StratifiedKFold"})
    else:
        transformations.append({"step": 3, "operation": "assign all released Train rows to fit and all released Test rows to scored in one full-refit fold", "software": "DLG-Sol input preparer"})
    derivation = {
        "schema_version": 1,
        "source_objects": source_objects,
        "transformations": transformations,
        "record_id_policy": "preserve the author-supplied C_ID exactly",
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
        files[role] = {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size, "rows": len(rows) if role != "partitions" else len(assignments), "columns": columns}
    files["derivation"] = {"path": derivation_path.name, "sha256": _sha256(derivation_path), "bytes": derivation_path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "dataset_id": "complat",
        "upstream_source_id": "complat",
        "upstream_revision": recipe["source"]["revision"],
        "retrieval_date": retrieval_date,
        "acquisition_mode": "user_supplied_local_package",
        "redistribution_status": "not_bundled_local_use_only",
        "canonicalization_policy": canonicalization_policy,
        "split_assignment_origin": split_origin,
        "files": files,
    }
    (output_path / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "evaluation_id": evaluation_id, "rows": len(rows), "partition_rows": len(assignments), "role_counts": observed_counts, "files": files}
