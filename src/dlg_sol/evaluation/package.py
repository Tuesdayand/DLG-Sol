from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath

from ..fusion.adapters import EvaluationAdapter


@dataclass(frozen=True)
class AcquisitionSource:
    source_id: str
    url: str
    revision: str
    licence_status: str
    acquisition_mode: str
    automatic_download: bool


@dataclass(frozen=True)
class AcquisitionRegistry:
    sources: dict[str, AcquisitionSource]
    evaluation_source_map: dict[str, str]


@dataclass(frozen=True)
class PackageFile:
    role: str
    path: Path
    sha256: str
    bytes: int
    rows: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationPackage:
    root: Path
    evaluation_id: str
    dataset_id: str
    model_identity: str
    acquisition_mode: str
    upstream_source_id: str
    upstream_revision: str
    retrieval_date: str
    derivation_record: str
    redistribution_status: str
    files: dict[str, PackageFile]
    scored_labels_content_accessed: bool


EXPECTED_SOURCES = {
    "aqsoldb": ("https://github.com/mcsorkun/AqSolDB", "8e02b548fd9a78778ff89a5aa9a460d1a289cc3a", "upstream_data_marked_cc0_1_0", "manual_pinned_upstream_acquisition", False),
    "llompart_aqsoldbc_dataset": ("https://doi.org/10.57745/CZVZIA", "2.0", "etalab_2_0", "user_supplied_local_package", False),
    "complat": ("https://github.com/ComPlat/water-solubility-prediction", "d4feda24b4bcb9efee9953605ba56af0e52dba4d", "readme_declares_mit_no_license_file_dataset_rights_not_established", "user_supplied_local_package", False),
    "tdc_dataset": ("https://tdcommons.ai/", "c310c35f27e3f506411018ac43d97b8ba23ca652", "software_mit_dataset_rights_separate", "user_supplied_local_package", False),
    "jchem_dataset": ("https://github.com/nadinulrich/log_Sw_prediction", "3da09dc96acde6c5c51d784307559067864ca779", "repository_mit_dataset_included_no_separate_data_notice", "user_supplied_local_package", False),
    "chemberta_zinc_base_v1": ("https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1", "761d6a18cf99db371e0b43baf3e2d21b3e865a20", "no_explicit_licence_in_retrieved_metadata", "identifier_only_no_bundled_weights", False),
}
EXPECTED_EVALUATION_SOURCES = {"R01": "llompart_aqsoldbc_dataset", "R02": "complat", "R03": "tdc_dataset", "R04": "tdc_dataset", "R05": "jchem_dataset", "R09": "complat"}
EXPECTED_FILE_NAMES = {
    "scored_component_predictions": "scored_component_predictions.csv",
    "coefficient_component_predictions": "coefficient_component_predictions.csv",
    "coefficient_labels": "coefficient_labels.csv",
    "scored_labels": "scored_labels.csv",
}
EXPECTED_COLUMNS = {
    "scored_component_predictions": ("record_id", "fold_index", "descriptor_prediction", "aug2_head4"),
    "coefficient_component_predictions": ("deployment_fold", "record_id", "descriptor_prediction", "aug2_head4"),
    "coefficient_labels": ("deployment_fold", "record_id", "logS"),
    "scored_labels": ("record_id", "logS"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_shape(path: Path) -> tuple[int, tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            columns = tuple(next(reader))
        except StopIteration as error:
            raise ValueError("package CSV cannot be empty") from error
        rows = sum(1 for _ in reader)
    return rows, columns


def load_acquisition_registry(path: str | Path) -> AcquisitionRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("network_default") != "deny":
        raise ValueError("unsupported or unsafe acquisition configuration")
    sources = {}
    for item in payload.get("resources", []):
        source = AcquisitionSource(str(item["source_id"]), str(item["url"]), str(item["revision"]), str(item["licence_status"]), str(item["acquisition_mode"]), bool(item["automatic_download"]))
        if source.source_id in sources:
            raise ValueError("acquisition source identifiers must be unique")
        sources[source.source_id] = source
    if set(sources) != set(EXPECTED_SOURCES):
        raise ValueError("acquisition registry does not contain the recorded source set")
    for source_id, expected in EXPECTED_SOURCES.items():
        source = sources[source_id]
        if (source.url, source.revision, source.licence_status, source.acquisition_mode, source.automatic_download) != expected:
            raise ValueError(f"acquisition contract mismatch: {source_id}")
    mapping = {str(key): str(value) for key, value in payload.get("evaluation_source_map", {}).items()}
    if mapping != EXPECTED_EVALUATION_SOURCES:
        raise ValueError("evaluation acquisition-source mapping differs from the recorded contract")
    return AcquisitionRegistry(sources, mapping)


def _safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("package file paths must be normalized relative paths")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents if parent != root.parent):
        raise ValueError("package files cannot traverse symbolic links")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("package file escapes its package root") from error
    if not resolved.is_file():
        raise ValueError(f"package file is missing: {relative}")
    return resolved


def _package_file(root: Path, role: str, item: dict, inspect_content: bool) -> PackageFile:
    if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes", "rows", "columns"}:
        raise ValueError(f"package file declaration is malformed: {role}")
    relative = str(item["path"])
    if relative != EXPECTED_FILE_NAMES[role]:
        raise ValueError(f"package file name differs from the contract: {role}")
    path = _safe_file(root, relative)
    declared_hash = str(item["sha256"])
    declared_bytes = int(item["bytes"])
    declared_rows = int(item["rows"])
    declared_columns = tuple(str(value) for value in item["columns"])
    if len(declared_hash) != 64 or any(value not in "0123456789abcdef" for value in declared_hash):
        raise ValueError(f"package SHA-256 is malformed: {role}")
    if declared_bytes != path.stat().st_size or declared_rows < 1 or declared_columns != EXPECTED_COLUMNS[role]:
        raise ValueError(f"package file metadata differs from the contract: {role}")
    if inspect_content:
        rows, columns = _csv_shape(path)
        if _sha256(path) != declared_hash or rows != declared_rows or columns != declared_columns:
            raise ValueError(f"package file content differs from its declaration: {role}")
    return PackageFile(role, path, declared_hash, declared_bytes, declared_rows, declared_columns)


def load_evaluation_package(root: str | Path, adapter: EvaluationAdapter, acquisition: AcquisitionRegistry, phase: str) -> EvaluationPackage:
    if phase not in {"assemble", "score"}:
        raise ValueError("package phase must be assemble or score")
    package_root = Path(root)
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError("evaluation package root must be a real directory")
    manifest_path = package_root / "package_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("evaluation package manifest is missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema_version", "evaluation_id", "dataset_id", "model_identity", "acquisition_mode", "upstream_source_id", "upstream_revision", "retrieval_date", "derivation_record", "contains_scored_labels", "redistribution_status", "files"}
    if set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("evaluation package manifest schema differs from the contract")
    evaluation_id = str(payload["evaluation_id"])
    source_id = str(payload["upstream_source_id"])
    if evaluation_id != adapter.evaluation_id or str(payload["dataset_id"]) != adapter.dataset_id or payload["model_identity"] != "aug2_head4":
        raise ValueError("package evaluation, dataset, or model identity mismatch")
    if acquisition.evaluation_source_map.get(evaluation_id) != source_id or source_id not in acquisition.sources:
        raise ValueError("package upstream source does not match its evaluation")
    source = acquisition.sources[source_id]
    if payload["acquisition_mode"] != source.acquisition_mode or payload["upstream_revision"] != source.revision:
        raise ValueError("package acquisition mode or upstream revision mismatch")
    try:
        date.fromisoformat(str(payload["retrieval_date"]))
    except ValueError as error:
        raise ValueError("package retrieval date must use ISO format") from error
    derivation = str(payload["derivation_record"])
    if not derivation or Path(derivation).is_absolute() or ".." in PurePosixPath(derivation).parts:
        raise ValueError("package derivation record must be a non-empty relative identifier")
    if payload["contains_scored_labels"] is not True or payload["redistribution_status"] != "not_bundled_local_use_only":
        raise ValueError("package label or redistribution declaration mismatch")
    expected_roles = {"scored_component_predictions", "scored_labels"}
    if adapter.coefficient_policy != "prespecified_0.5":
        expected_roles |= {"coefficient_component_predictions", "coefficient_labels"}
    files_payload = payload["files"]
    if not isinstance(files_payload, dict) or set(files_payload) != expected_roles:
        raise ValueError("package file set differs from its evaluation contract")
    files = {}
    for role in sorted(expected_roles):
        inspect = not (phase == "assemble" and role == "scored_labels")
        files[role] = _package_file(package_root, role, files_payload[role], inspect)
    return EvaluationPackage(package_root.resolve(), evaluation_id, adapter.dataset_id, "aug2_head4", source.acquisition_mode, source_id, source.revision, str(payload["retrieval_date"]), derivation, str(payload["redistribution_status"]), files, phase == "score")
