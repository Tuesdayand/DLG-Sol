from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from ..evaluation.package import AcquisitionRegistry


@dataclass(frozen=True)
class BenchmarkInputContract:
    evaluation_id: str
    dataset_id: str
    upstream_source_id: str
    fold_indices: tuple[int, ...]
    required_roles: tuple[str, ...]
    scored_assignment: str
    expected_scored_rows: int


@dataclass(frozen=True)
class InputContractRegistry:
    files: dict[str, dict]
    contracts: dict[str, BenchmarkInputContract]
    redistribution_status: str


@dataclass(frozen=True)
class BenchmarkInputFile:
    role: str
    path: Path
    sha256: str
    bytes: int
    rows: int | None
    columns: tuple[str, ...] | None


@dataclass(frozen=True)
class BenchmarkInputPackage:
    root: Path
    evaluation_id: str
    dataset_id: str
    upstream_source_id: str
    upstream_revision: str
    retrieval_date: str
    canonicalization_policy: str
    split_assignment_origin: str
    files: dict[str, BenchmarkInputFile]
    molecule_rows: int
    partition_rows: int
    scored_rows: int


EXPECTED_CONTRACTS = {
    "R01": ("aqsoldbc", "llompart_aqsoldbc_dataset", (0, 1, 2, 3, 4), ("fit", "reserved", "scored"), "one_held_out_prediction", 8047),
    "R02": ("complat", "complat", (0, 1, 2, 3, 4), ("fit", "scored"), "one_held_out_prediction", 17937),
    "R03": ("tdc", "tdc_dataset", (0, 1, 2, 3, 4), ("fit", "reserved", "scored"), "one_held_out_prediction", 7985),
    "R04": ("tdc", "tdc_dataset", (0, 1, 2, 3, 4), ("fit", "reserved", "scored"), "same_rows_in_every_fold", 1997),
    "R05": ("jchem", "jchem_dataset", (1, 2, 3, 4, 5), ("fit", "coefficient", "scored"), "same_rows_in_every_fold", 980),
    "R09": ("complat", "complat", (0,), ("fit", "scored"), "single_full_refit", 1282),
}
EXPECTED_FILES = {
    "molecules": {"path": "molecules.csv", "columns": ["record_id", "smiles"]},
    "labels": {"path": "labels.csv", "columns": ["record_id", "logS"]},
    "partitions": {"path": "partitions.csv", "columns": ["record_id", "fold_index", "role"]},
    "derivation": {"path": "derivation.json"},
}
ALLOWED_ROLES = {"fit", "coefficient", "scored", "reserved"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("input-package paths must be normalized relative paths")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents if parent != root.parent):
        raise ValueError("input-package files cannot traverse symbolic links")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("input-package file escapes its package root") from error
    if not resolved.is_file():
        raise ValueError(f"input-package file is missing: {relative}")
    return resolved


def _read_csv(path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"input CSV schema differs from the contract: {path.name}")
        return list(reader)


def _validate_text(value: object, field: str) -> str:
    text = str(value).strip()
    if not text or "\x00" in text:
        raise ValueError(f"derivation field must be non-empty text: {field}")
    return text


def load_input_contract_registry(path: str | Path) -> InputContractRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("network_default") != "deny":
        raise ValueError("unsupported or unsafe benchmark-input configuration")
    if payload.get("redistribution_status") != "not_bundled_local_use_only" or payload.get("files") != EXPECTED_FILES:
        raise ValueError("benchmark-input file or redistribution contract mismatch")
    contracts = {}
    for item in payload.get("evaluation_contracts", []):
        contract = BenchmarkInputContract(
            str(item["evaluation_id"]),
            str(item["dataset_id"]),
            str(item["upstream_source_id"]),
            tuple(int(value) for value in item["fold_indices"]),
            tuple(str(value) for value in item["required_roles"]),
            str(item["scored_assignment"]),
            int(item["expected_scored_rows"]),
        )
        if contract.evaluation_id in contracts:
            raise ValueError("benchmark-input evaluation identifiers must be unique")
        contracts[contract.evaluation_id] = contract
    if set(contracts) != set(EXPECTED_CONTRACTS):
        raise ValueError("benchmark-input registry does not contain the primary evaluation set")
    for evaluation_id, expected in EXPECTED_CONTRACTS.items():
        observed = contracts[evaluation_id]
        values = (observed.dataset_id, observed.upstream_source_id, observed.fold_indices, observed.required_roles, observed.scored_assignment, observed.expected_scored_rows)
        if values != expected:
            raise ValueError(f"benchmark-input contract mismatch: {evaluation_id}")
    evidence = payload.get("source_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"llompart_aqsoldbc_dataset", "complat", "tdc_dataset", "jchem_dataset"}:
        raise ValueError("benchmark-input source evidence set mismatch")
    return InputContractRegistry(dict(payload["files"]), contracts, str(payload["redistribution_status"]))


def _load_declared_file(root: Path, role: str, declaration: object, registry: InputContractRegistry) -> BenchmarkInputFile:
    expected = registry.files[role]
    required = {"path", "sha256", "bytes"} if role == "derivation" else {"path", "sha256", "bytes", "rows", "columns"}
    if not isinstance(declaration, dict) or set(declaration) != required:
        raise ValueError(f"input-package file declaration is malformed: {role}")
    if declaration["path"] != expected["path"]:
        raise ValueError(f"input-package file name differs from the contract: {role}")
    path = _safe_file(root, str(declaration["path"]))
    declared_hash = str(declaration["sha256"])
    declared_bytes = int(declaration["bytes"])
    if not _valid_sha256(declared_hash) or declared_bytes != path.stat().st_size or _sha256(path) != declared_hash:
        raise ValueError(f"input-package hash or size mismatch: {role}")
    if role == "derivation":
        return BenchmarkInputFile(role, path, declared_hash, declared_bytes, None, None)
    rows = int(declaration["rows"])
    columns = tuple(str(value) for value in declaration["columns"])
    if rows < 1 or columns != tuple(expected["columns"]):
        raise ValueError(f"input-package row or column declaration mismatch: {role}")
    observed_rows = _read_csv(path, columns)
    if len(observed_rows) != rows:
        raise ValueError(f"input-package row count mismatch: {role}")
    return BenchmarkInputFile(role, path, declared_hash, declared_bytes, rows, columns)


def _validate_derivation(path: Path, package: dict, acquisition: AcquisitionRegistry) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "source_objects", "transformations", "record_id_policy", "canonicalization_policy", "split_assignment_origin"}
    if not isinstance(payload, dict) or set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("derivation record schema differs from the contract")
    if _validate_text(payload["canonicalization_policy"], "canonicalization_policy") != package["canonicalization_policy"]:
        raise ValueError("canonicalization policy differs between package and derivation record")
    if _validate_text(payload["split_assignment_origin"], "split_assignment_origin") != package["split_assignment_origin"]:
        raise ValueError("split-assignment origin differs between package and derivation record")
    _validate_text(payload["record_id_policy"], "record_id_policy")
    objects = payload["source_objects"]
    if not isinstance(objects, list) or not objects:
        raise ValueError("derivation record must identify at least one hashed source object")
    object_fields = {"source_id", "locator", "revision", "retrieval_date", "sha256", "bytes"}
    source_ids = set()
    for item in objects:
        if not isinstance(item, dict) or set(item) != object_fields:
            raise ValueError("derivation source-object declaration is malformed")
        source_id = _validate_text(item["source_id"], "source_id")
        source_ids.add(source_id)
        if source_id not in acquisition.sources or item["revision"] != acquisition.sources[source_id].revision:
            raise ValueError("derivation source object is not tied to a pinned acquisition source")
        locator = _validate_text(item["locator"], "locator")
        parsed = urlparse(locator)
        invalid_url = bool(parsed.scheme) and (parsed.scheme not in {"http", "https"} or not parsed.netloc)
        invalid_relative = not parsed.scheme and (PurePosixPath(locator).is_absolute() or ".." in PurePosixPath(locator).parts)
        if invalid_url or invalid_relative:
            raise ValueError("derivation source locator cannot expose or traverse a local absolute path")
        try:
            date.fromisoformat(str(item["retrieval_date"]))
        except ValueError as error:
            raise ValueError("derivation retrieval date must use ISO format") from error
        if not _valid_sha256(str(item["sha256"])) or int(item["bytes"]) < 1:
            raise ValueError("derivation source object requires a SHA-256 and positive byte size")
    if package["upstream_source_id"] not in source_ids:
        raise ValueError("derivation record does not bind the package's declared upstream source")
    transformations = payload["transformations"]
    if not isinstance(transformations, list) or not transformations:
        raise ValueError("derivation record must contain an ordered transformation history")
    for expected_step, item in enumerate(transformations, start=1):
        if not isinstance(item, dict) or set(item) != {"step", "operation", "software"} or item["step"] != expected_step:
            raise ValueError("derivation transformations must be consecutively numbered")
        _validate_text(item["operation"], "operation")
        _validate_text(item["software"], "software")


def _validate_rows(files: dict[str, BenchmarkInputFile], contract: BenchmarkInputContract) -> tuple[int, int, int]:
    molecules = _read_csv(files["molecules"].path, files["molecules"].columns or ())
    labels = _read_csv(files["labels"].path, files["labels"].columns or ())
    partitions = _read_csv(files["partitions"].path, files["partitions"].columns or ())
    molecule_ids = []
    for row in molecules:
        record_id = row["record_id"].strip()
        smiles = row["smiles"].strip()
        if not record_id or not smiles:
            raise ValueError("molecule record identifiers and SMILES must be non-empty")
        molecule_ids.append(record_id)
    if len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError("molecule record identifiers must be unique")
    label_ids = []
    for row in labels:
        record_id = row["record_id"].strip()
        try:
            value = float(row["logS"])
        except ValueError as error:
            raise ValueError("logS labels must be numeric") from error
        if not record_id or not math.isfinite(value):
            raise ValueError("label identifiers must be non-empty and logS must be finite")
        label_ids.append(record_id)
    if len(set(label_ids)) != len(label_ids) or set(label_ids) != set(molecule_ids):
        raise ValueError("labels must match the unique molecule identifiers exactly")
    molecule_id_set = set(molecule_ids)
    assignments = set()
    assigned_ids = set()
    role_counts = defaultdict(Counter)
    scored_counts = Counter()
    fold_set = set(contract.fold_indices)
    for row in partitions:
        record_id = row["record_id"].strip()
        role = row["role"].strip()
        try:
            fold = int(row["fold_index"])
        except ValueError as error:
            raise ValueError("partition fold indices must be integers") from error
        key = (record_id, fold)
        if record_id not in molecule_id_set or fold not in fold_set or role not in ALLOWED_ROLES:
            raise ValueError("partition row contains an unknown record, fold, or role")
        if key in assignments:
            raise ValueError("each record can have only one role within a fold")
        assignments.add(key)
        assigned_ids.add(record_id)
        role_counts[fold][role] += 1
        if role == "scored":
            scored_counts[record_id] += 1
    if assigned_ids != molecule_id_set:
        raise ValueError("every molecule must have at least one partition assignment")
    required_roles = set(contract.required_roles)
    for fold in contract.fold_indices:
        present = {role for role, count in role_counts[fold].items() if count > 0}
        if present != required_roles:
            raise ValueError(f"partition roles differ from the evaluation contract in fold {fold}")
    if len(scored_counts) != contract.expected_scored_rows:
        raise ValueError("unique scored-row count differs from the evaluation contract")
    if contract.scored_assignment in {"one_held_out_prediction", "single_full_refit"}:
        expected_repetitions = 1
    elif contract.scored_assignment == "same_rows_in_every_fold":
        expected_repetitions = len(contract.fold_indices)
    else:
        raise ValueError("unsupported scored-assignment contract")
    if set(scored_counts.values()) != {expected_repetitions}:
        raise ValueError("scored rows do not follow the required fold-assignment pattern")
    return len(molecules), len(partitions), len(scored_counts)


def load_benchmark_input_package(root: str | Path, evaluation_id: str, registry: InputContractRegistry, acquisition: AcquisitionRegistry) -> BenchmarkInputPackage:
    package_root = Path(root)
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError("benchmark-input package root must be a real directory")
    manifest_path = package_root / "package_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("benchmark-input package manifest is missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema_version", "evaluation_id", "dataset_id", "upstream_source_id", "upstream_revision", "retrieval_date", "acquisition_mode", "redistribution_status", "canonicalization_policy", "split_assignment_origin", "files"}
    if not isinstance(payload, dict) or set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("benchmark-input package manifest schema differs from the contract")
    if evaluation_id not in registry.contracts or payload["evaluation_id"] != evaluation_id:
        raise ValueError("benchmark-input evaluation identity mismatch")
    contract = registry.contracts[evaluation_id]
    source_id = str(payload["upstream_source_id"])
    if payload["dataset_id"] != contract.dataset_id or source_id != contract.upstream_source_id:
        raise ValueError("benchmark-input dataset or source identity mismatch")
    if source_id not in acquisition.sources:
        raise ValueError("benchmark-input source is absent from the acquisition registry")
    source = acquisition.sources[source_id]
    if payload["upstream_revision"] != source.revision or payload["acquisition_mode"] != source.acquisition_mode:
        raise ValueError("benchmark-input acquisition mode or revision mismatch")
    if payload["redistribution_status"] != registry.redistribution_status:
        raise ValueError("benchmark-input redistribution declaration mismatch")
    try:
        date.fromisoformat(str(payload["retrieval_date"]))
    except ValueError as error:
        raise ValueError("benchmark-input retrieval date must use ISO format") from error
    canonicalization_policy = _validate_text(payload["canonicalization_policy"], "canonicalization_policy")
    split_origin = _validate_text(payload["split_assignment_origin"], "split_assignment_origin")
    declarations = payload["files"]
    if not isinstance(declarations, dict) or set(declarations) != set(EXPECTED_FILES):
        raise ValueError("benchmark-input package file set differs from the contract")
    files = {role: _load_declared_file(package_root, role, declarations[role], registry) for role in EXPECTED_FILES}
    allowed_names = {"package_manifest.json", *(item.path.name for item in files.values())}
    actual_names = {path.name for path in package_root.iterdir()}
    if actual_names != allowed_names:
        raise ValueError("benchmark-input package contains undeclared or missing top-level files")
    _validate_derivation(files["derivation"].path, payload, acquisition)
    molecule_rows, partition_rows, scored_rows = _validate_rows(files, contract)
    return BenchmarkInputPackage(
        package_root.resolve(), evaluation_id, contract.dataset_id, source_id, source.revision,
        str(payload["retrieval_date"]), canonicalization_policy, split_origin, files,
        molecule_rows, partition_rows, scored_rows,
    )
