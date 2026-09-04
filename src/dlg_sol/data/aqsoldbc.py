from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np


REQUIRED_SOURCE_COLUMNS = (
    "Source", "ID", "Name", "SMILES", "SmilesCurated", "ExperimentalLogS",
    "InChI", "InChIKey", "Composition", "Origin", "Dataset", "HasError",
    "ErrorType", "AtomCount", "AlertAtoms", "DuplicateGroup", "DuplicateSD",
    "DuplicateOccurrence", "SD",
)
EXPECTED_RECIPE_SHA256 = "78e2dd21bcb96e633a1a4ce98606a2438a464649d0a8adb41b4592188a0690a9"


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


def reconstruct_aqsoldbc_partitions(row_count: int, n_splits: int, seed: int, reserved_fraction: float) -> list[tuple[int, int, str]]:
    source_indices = np.arange(row_count, dtype=np.int64)
    shuffled = source_indices.copy()
    np.random.RandomState(seed).shuffle(shuffled)
    fold_sizes = np.full(n_splits, row_count // n_splits, dtype=np.int64)
    fold_sizes[: row_count % n_splits] += 1
    assignments: list[tuple[int, int, str]] = []
    start = 0
    for fold_index, fold_size in enumerate(fold_sizes.tolist()):
        stop = start + fold_size
        scored_indices = np.sort(shuffled[start:stop])
        outer_fit_mask = np.ones(row_count, dtype=bool)
        outer_fit_mask[scored_indices] = False
        outer_fit_indices = source_indices[outer_fit_mask]
        inner_order = np.random.RandomState(seed + fold_index).permutation(len(outer_fit_indices))
        reserved_count = int(math.ceil(reserved_fraction * len(outer_fit_indices)))
        reserved_indices = set(outer_fit_indices[inner_order[:reserved_count]].tolist())
        scored_index_set = set(scored_indices.tolist())
        for row_index in source_indices.tolist():
            if row_index in scored_index_set:
                role = "scored"
            elif row_index in reserved_indices:
                role = "reserved"
            else:
                role = "fit"
            assignments.append((row_index, fold_index, role))
        start = stop
    return assignments


def prepare_aqsoldbc_input_package(source: str | Path, output: str | Path, retrieval_date: str, recipe_path: str | Path) -> dict:
    source_path = Path(source)
    output_path = Path(output)
    recipe_file = Path(recipe_path)
    if _sha256(recipe_file) != EXPECTED_RECIPE_SHA256:
        raise ValueError("AqSolDBc partition recipe differs from the recorded R01 contract")
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    date.fromisoformat(retrieval_date)
    source_contract = recipe["source"]
    if not source_path.is_file():
        raise ValueError("AqSolDBc source file is missing")
    if source_path.stat().st_size != source_contract["bytes"] or _sha256(source_path) != source_contract["sha256"]:
        raise ValueError("AqSolDBc source file does not match the pinned original object")
    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_SOURCE_COLUMNS:
            raise ValueError("AqSolDBc source columns differ from the pinned object")
        source_rows = list(reader)
    if len(source_rows) != source_contract["rows"]:
        raise ValueError("AqSolDBc source row count differs from the pinned object")
    record_ids = [row[source_contract["columns"]["record_id"]].strip() for row in source_rows]
    smiles = [row[source_contract["columns"]["smiles"]].strip() for row in source_rows]
    targets = [row[source_contract["columns"]["target"]].strip() for row in source_rows]
    if any(not value for value in record_ids + smiles + targets) or len(set(record_ids)) != len(record_ids):
        raise ValueError("AqSolDBc identifiers, curated SMILES, and targets must be complete and identifiers must be unique")
    if any(not math.isfinite(float(value)) for value in targets):
        raise ValueError("AqSolDBc targets must be finite")
    partition_contract = recipe["partition"]
    outer = partition_contract["outer"]
    assignments = reconstruct_aqsoldbc_partitions(len(source_rows), outer["n_splits"], outer["random_state"], partition_contract["inner_reserved"]["test_fraction"])
    assignment_text = "".join(f"{row_index},{fold_index},{role}\n" for row_index, fold_index, role in assignments).encode()
    if hashlib.sha256(assignment_text).hexdigest() != recipe["expected_output"]["assignment_index_sha256"]:
        raise ValueError("reconstructed AqSolDBc assignment identity differs from the recorded protocol")
    observed_counts: dict[str, dict[str, int]] = {}
    for fold_index in range(outer["n_splits"]):
        counts = Counter(role for _, fold, role in assignments if fold == fold_index)
        observed_counts[str(fold_index)] = dict(counts)
    if observed_counts != recipe["expected_role_counts"]:
        raise ValueError("reconstructed AqSolDBc role counts differ from the recorded protocol")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("output directory must be absent or empty")
    output_path.mkdir(parents=True, exist_ok=True)
    molecules_path = output_path / "molecules.csv"
    labels_path = output_path / "labels.csv"
    partitions_path = output_path / "partitions.csv"
    derivation_path = output_path / "derivation.json"
    _write_csv(molecules_path, ("record_id", "smiles"), list(zip(record_ids, smiles)))
    _write_csv(labels_path, ("record_id", "logS"), list(zip(record_ids, targets)))
    _write_csv(partitions_path, ("record_id", "fold_index", "role"), [(record_ids[index], fold, role) for index, fold, role in assignments])
    expected_output = recipe["expected_output"]
    observed_output = {
        "molecules_sha256": _sha256(molecules_path),
        "labels_sha256": _sha256(labels_path),
        "partitions_sha256": _sha256(partitions_path),
    }
    if observed_output != {key: expected_output[key] for key in observed_output}:
        raise ValueError("prepared AqSolDBc package rows differ from the recorded R01 identities")
    canonicalization_policy = "author-supplied SmilesCurated values copied without transformation; downstream model canonicalization is stage-specific"
    split_assignment_origin = "recorded R01 KFold and inner ShuffleSplit recipe in configs/aqsoldbc_partition_recipe.json"
    derivation = {
        "schema_version": 1,
        "source_objects": [{
            "source_id": "llompart_aqsoldbc_dataset",
            "locator": source_contract["persistent_file_id"],
            "revision": source_contract["revision"],
            "retrieval_date": retrieval_date,
            "sha256": source_contract["sha256"],
            "bytes": source_contract["bytes"],
        }],
        "transformations": [
            {"step": 1, "operation": "verify the original AqSolDBc file hash, byte size, schema, and row count", "software": "DLG-Sol input preparer"},
            {"step": 2, "operation": "copy ID, SmilesCurated, and ExperimentalLogS in original author row order", "software": "DLG-Sol input preparer"},
            {"step": 3, "operation": "reconstruct five outer folds and fold-specific inner reserved rows from the recorded random states", "software": "NumPy legacy RandomState"},
        ],
        "record_id_policy": "preserve the author-supplied AqSolDBc ID exactly",
        "canonicalization_policy": canonicalization_policy,
        "split_assignment_origin": split_assignment_origin,
    }
    derivation_path.write_text(json.dumps(derivation, indent=2) + "\n", encoding="utf-8")
    files = {}
    for role, path, columns in (
        ("molecules", molecules_path, ["record_id", "smiles"]),
        ("labels", labels_path, ["record_id", "logS"]),
        ("partitions", partitions_path, ["record_id", "fold_index", "role"]),
    ):
        files[role] = {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size, "rows": len(source_rows) if role != "partitions" else len(assignments), "columns": columns}
    files["derivation"] = {"path": derivation_path.name, "sha256": _sha256(derivation_path), "bytes": derivation_path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "evaluation_id": "R01",
        "dataset_id": "aqsoldbc",
        "upstream_source_id": "llompart_aqsoldbc_dataset",
        "upstream_revision": source_contract["revision"],
        "retrieval_date": retrieval_date,
        "acquisition_mode": "user_supplied_local_package",
        "redistribution_status": "not_bundled_local_use_only",
        "canonicalization_policy": canonicalization_policy,
        "split_assignment_origin": split_assignment_origin,
        "files": files,
    }
    (output_path / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "evaluation_id": "R01", "rows": len(source_rows), "partition_rows": len(assignments), "role_counts": observed_counts, "files": files}
